# EPEX day-ahead price prediction

This is a simple statistical model to predict EPEX day-ahead prices based on various parameters.
It works to a reasonably good degree. Better than many of the commercial solutions.
This repository includes
- The self-training prediction model itself
- A FastAPI app that serves the latest cached forecasts
- A background worker that refreshes forecasts and stores issue-time snapshots
- A Docker compose file to have it running wherever

Supported Countries:
- Germany (default)
- Austria
- Belgium
- Netherlands
- Sweden (SE1-SE4)
- Denmark (DK1-DK2)
- Finland (FI)
- Others can be added relatively easily, if there is interest


## Lookout
- Maybe package it directly as a Home Assistant Add-on

## The Model
We sample multiple locations distributed across each region. We fetch [Weather data from Open-Meteo.com](https://open-meteo.com/) for those locations for the past n days (default n=120).
This serves as the main data source.

Price data is provided under CC BY 4.0 by smartd.de, retrieved via [api.energy-charts.info](https://api.energy-charts.info/) and [ENTSO-E transparency platform](https://transparency.entsoe.eu/).

Grid load data is provided by [ENTSO-E transparency platform](https://transparency.entsoe.eu/).

### Features

Weather features (per sample location):
- Wind speed at 80m
- Temperature at 2m
- Global tilted irradiance (solar)
- Air pressure at mean sea level
- Relative humidity

Time features:
- Azimuth of the sun as indicator of time of day
- Elevation of the sun
- Day of the week (Monday to Saturday)
- Holiday/Sunday indicator (regional holidays weighted by fraction of regions, e.g. 0.5 if half the regions have the holiday)
- Sunrise influence: how many minutes between sunrise and the current time slot
- Sunset influence: how many minutes between sunset and the current time slot

Other:
- Entso-E load forecast (optional, but highly recommended, especially for DE and AT)
- Natural gas day-ahead-price, forward filled (select regions only)
- Conservative FI-only post-model blend against yesterday's same-slot price
- FI market-state features from ENTSO-E and optional Fingrid open data
  - load, generation, wind and solar forecasts
  - imbalance and cross-border flow state summaries
  - coupled-market shadow price state for SE1, SE3, EE and NO4
  - derived residual-load and import-headroom features

Output:
- Electricity price
- Internal quantile forecasts and issue-time feature snapshots for evaluation

## How it works
The runtime now has two parts:
- `epexpredictor` serves the last good cached forecast and only falls back to request-time retraining if needed
- `forecast-worker` refreshes data every few hours, retrains models, writes forecast artifacts and stores issue-time snapshots

The core predictor stays conservative and tree-based:
- a main LightGBM point model
- FI quantile models (`q10`, `q50`, `q90`) for forecast distribution tracking
- a spike classifier and spike-uplift regressor for volatile regimes
- a small dynamic FI-only blend against yesterday's same-slot price

## Model performance
For performance testing, see `predictor/performance_testing.py`.
For detailed offline evaluation against actual prices, use `python -m predictor.evaluate_predictions --region FI --eval-start 2026-02-01T00:00:00Z --eval-end 2026-04-07T23:45:00Z --source backtest`.
The evaluator reports overall metrics, lead buckets, volatility buckets and generated-at window slices. Use `--primary-window-only` to focus on the configured morning issue window for a region.
For a visual comparison against actual prices, use `python -m predictor.plot_snapshot_evaluation`.
- For production-faithful live evaluation, use `--source snapshots`.
- For immediate retrospective visuals, use `--source backtest`.
- Example: `python -m predictor.plot_snapshot_evaluation --region FI --eval-start 2026-04-01T00:00:00Z --eval-end 2026-04-07T23:45:00Z --source backtest --selection latest --output-file ./data/fi_backtest_eval.png`.
The plot command selects one forecast per target timestamp, plots it against actual prices and baseline series, and is a better decision tool than `/eval_plot`.

Remarks:
- Tests were run in early 2026, with data from 2025-01-24 to 2026-01-24. The model is tuned for 15 minute pricing. Since data before 2025-10-01 were using hourly pricing, actual performance might be slightly better
- The model uses a 120-day rolling training window
- Tests were done with historical weather data. If the weather forecast is wrong, performance might be slightly worse in practice
- Offline weather evaluation currently uses Open-Meteo historical forecast data, which is useful for backtests but not a perfect proxy for live forecast quality. Production-faithful analysis should prefer snapshot history first, and later move to Open-Meteo Previous Runs for day-ahead realism.

Results (1-day ahead prediction):
| Region | MAE (ct/kWh) | RMSE (ct/kWh) |
|--------|--------------|---------------|
| DE_LU  | 1.57         | 2.38          |
| AT     | 1.8          | 2.77          |
| BE     | 1.74         | 2.53          |
| NL     | 1.66         | 2.53          |
| SE_1   | 1.37         | 2.48          |
| SE_2   | 1.29         | 2.44          |
| SE_3   | 2.11         | 3.1           |
| SE_4   | 2.39         | 3.35          |
| DK_1   | 1.99         | 2.92          |
| DK_2   | 2.19         | 3.2           |

Some observations:
- At night, predictions are typically within 0.5 ct/kWh
- Morning/Evening peaks are typically within 1-1.5 ct/kWh
- Extreme peaks due to "Dunkelflaute" are correctly detected, but estimation of the exact price is a challenge (e.g. the model might predict 75ct while reality is 60ct or vice versa)
- High PV noons are usually correctly detected with good accuracy


### Current forecast (DE)
![image](https://epexpredictor.batzill.com/eval_plot?region=DE&transparent=false&width=1024&height=512)


Feel free to generate your own plot for other time ranges or regions [here](https://epexpredictor.batzill.com/docs#/default/generate_evaluation_plot_eval_plot_get).


# Public API
You can find a freely accessible installment of this software [here](https://epexpredictor.batzill.com/).
Get a glimpse of the current prediction [here](https://epexpredictor.batzill.com/prices).
There is also a lightweight inspector UI at `/ui` to compare cached prediction data, actual prices and selected source series.
For FI, `/ui` also exposes a snapshot explainability workspace that loads issue-time forecast runs, grouped driver contributions, and a limited what-if sensitivity view from saved model bundles.

There are no guarantees given whatsoever - it might work for you or not.
I might stop or block this service at any time. Fair use is expected!

# Self Hosting
You can easily self-host this software. For easy deployment, check out the docker compose file.
You will probably want to register with Entso-E and request an API key.
For FI market-feature enrichment, you can optionally add `EPEXPREDICTOR_FINGRID_API_KEY`.
Without Entso-E API access
- some parameters are missing and the model will perform significantly worse, especially for DE and AT
- Some regions will not be available (e.g. SE1-4)

# Home Assistant integration
At some point, I might create a HA addon to run everything locally.
For now, you have to either use my server, or run it yourself.

Note: Home Assistant only supports a limited amount of data in state attributes. Therefore, we use the "short format" output, and limit the time to 120 hours.
If you need more, you will have to be more creative.
Personally, I provide the data as a HA "service" (now "action") using pyscript, and then call this service to work with the data.



### Configuration:
```yaml
# Make sure you change the parameters region, surcharge and taxPercent according to your electricity plan
sensor:
  - platform: rest
    resource: "https://epexpredictor.batzill.com/prices_short?region=DE&surcharge=13.70084&taxPercent=19&unit=EUR_PER_KWH&hours=120"
    method: GET
    unique_id: epex_price_prediction
    name: "EPEX Price Prediction"
    unit_of_measurement: €/kWh
    value_template: "{{ value_json.t[0] }}"
    json_attributes:
      - s
      - t

  # If you want to evaluate performance in real time, you can add another sensor like this
  # and plot it in the same diagram as the actual prediction sensor

  #- platform: rest
  #  resource: "https://epexpredictor.batzill.com/prices_short?region=DE&surcharge=13.70084&taxPercent=19&evaluation=true&unit=EUR_PER_KWH&hours=120"
  #  method: GET
  #  unique_id: epex_price_prediction_evaluation
  #  name: "EPEX Price Prediction Evaluation"
  #  unit_of_measurement: €/kWh
  #  value_template: "{{ value_json.t[0] }}"
  #  json_attributes:
  #    - s
  #    - t
```

### Display, e.g. via Plotly Graph Card:
```yaml
type: custom:plotly-graph
time_offset: 26h
layout:
  yaxis9:
    fixedrange: true
    visible: false
    minallowed: 0
    maxallowed: 1
entities:
  - entity: sensor.epex_price_prediction
    name: EPEX Price Prediction
    unit_of_measurement: ct/kWh
    texttemplate: "%{y:.0f}"
    mode: lines+text
    textposition: top right
    filters:
      - fn: |-
          ({xs, ys, meta}) => {
            return {
              xs: xs.concat(meta.s.map(s => s*1000)),
              ys: ys.concat(meta.t).map(t => +t*100)
            }
          }
  - entity: ""
    name: Now
    yaxis: y9
    showlegend: false
    line:
      width: 1
      dash: dot
      color: orange
    x: $ex [Date.now(), Date.now()]
    "y":
      - 0
      - 1
hours_to_show: 30
refresh_interval: 10
```

# evcc integration

[evcc](https://evcc.io/) is an open-source EV charging controller that can optimize charging based on electricity prices.
It now has native support for EpexPredictor, see [docs](https://docs.evcc.io/docs/tariffs#epex-predictor-predicted-epex-spot-prices)
