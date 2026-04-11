(function () {
  const palette = ["#005f73", "#9b2226", "#ca6702", "#6a4c93", "#386641", "#8d5a97"];
  const regionSelect = document.getElementById("regionSelect");
  const startInput = document.getElementById("startInput");
  const endInput = document.getElementById("endInput");
  const sourcePicker = document.getElementById("sourcePicker");
  const sourcePickerRow = document.getElementById("sourcePickerRow");
  const reloadButton = document.getElementById("reloadButton");
  const preset48h = document.getElementById("preset48h");
  const preset7d = document.getElementById("preset7d");
  const liveTabButton = document.getElementById("liveTabButton");
  const snapshotTabButton = document.getElementById("snapshotTabButton");
  const modeCopy = document.getElementById("modeCopy");
  const liveWorkspace = document.getElementById("liveWorkspace");
  const snapshotWorkspace = document.getElementById("snapshotWorkspace");

  const priceChart = document.getElementById("priceChart");
  const priceSummary = document.getElementById("priceSummary");
  const sourceCharts = document.getElementById("sourceCharts");
  const sourceSummary = document.getElementById("sourceSummary");
  const dataTable = document.getElementById("dataTable");
  const tableSummary = document.getElementById("tableSummary");
  const statusRegion = document.getElementById("statusRegion");
  const statusGenerated = document.getElementById("statusGenerated");
  const statusKnownUntil = document.getElementById("statusKnownUntil");
  const statusOverlap = document.getElementById("statusOverlap");

  const generatedAtSelect = document.getElementById("generatedAtSelect");
  const targetTimeSelect = document.getElementById("targetTimeSelect");
  const snapshotSelectionSummary = document.getElementById("snapshotSelectionSummary");
  const snapshotRunMeta = document.getElementById("snapshotRunMeta");
  const snapshotSummary = document.getElementById("snapshotSummary");
  const snapshotChart = document.getElementById("snapshotChart");
  const localExplainSummary = document.getElementById("localExplainSummary");
  const localMetrics = document.getElementById("localMetrics");
  const groupContributionBars = document.getElementById("groupContributionBars");
  const localAdjustments = document.getElementById("localAdjustments");
  const featureContributionTable = document.getElementById("featureContributionTable");
  const historicalSummary = document.getElementById("historicalSummary");
  const summaryGroupBars = document.getElementById("summaryGroupBars");
  const summaryFeatureTable = document.getElementById("summaryFeatureTable");
  const errorSliceCards = document.getElementById("errorSliceCards");
  const scenarioForm = document.getElementById("scenarioForm");
  const scenarioRunButton = document.getElementById("scenarioRunButton");
  const scenarioSummary = document.getElementById("scenarioSummary");
  const scenarioMetrics = document.getElementById("scenarioMetrics");
  const scenarioGroupBars = document.getElementById("scenarioGroupBars");
  const scenarioAppliedInputs = document.getElementById("scenarioAppliedInputs");
  const scenarioChangedFeaturesTable = document.getElementById("scenarioChangedFeaturesTable");
  const sourceGroupTemplate = document.getElementById("sourceGroupTemplate");

  const scenarioInputs = {
    loadForecastPct: document.getElementById("scenarioLoadPct"),
    windForecastPct: document.getElementById("scenarioWindPct"),
    solarForecastPct: document.getElementById("scenarioSolarPct"),
    importHeadroomMw: document.getElementById("scenarioImportMw"),
    crossBorderFlowMw: document.getElementById("scenarioFlowMw"),
    coupledMarketSpreadDelta: document.getElementById("scenarioSpreadDelta"),
  };

  const state = {
    activeMode: "live",
    latestPayload: null,
    selectedSourceColumns: [],
    snapshotCatalog: null,
    snapshotSummary: null,
    localExplanation: null,
    scenarioResult: null,
  };

  function formatLocal(isoString) {
    if (!isoString) return "-";
    const value = new Date(isoString);
    return new Intl.DateTimeFormat(undefined, {
      year: "numeric",
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    }).format(value);
  }

  function toDatetimeLocal(isoString) {
    const value = new Date(isoString);
    const year = value.getFullYear();
    const month = String(value.getMonth() + 1).padStart(2, "0");
    const day = String(value.getDate()).padStart(2, "0");
    const hour = String(value.getHours()).padStart(2, "0");
    const minute = String(value.getMinutes()).padStart(2, "0");
    return `${year}-${month}-${day}T${hour}:${minute}`;
  }

  function fromDatetimeLocal(value) {
    if (!value) return null;
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? null : date.toISOString();
  }

  function formatNumber(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) return "–";
    return Number(value).toFixed(digits);
  }

  function formatSigned(value, digits = 2) {
    if (value === null || value === undefined || Number.isNaN(value)) return "–";
    const number = Number(value);
    return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}`;
  }

  function normalizeRegionValue(region) {
    if (!region) return "FI";
    return region.replace("_LU", "") === "DE" ? "DE" : region.replaceAll("_", "");
  }

  function buildLiveUrl() {
    const params = new URLSearchParams();
    params.set("region", regionSelect.value);
    const startIso = fromDatetimeLocal(startInput.value);
    const endIso = fromDatetimeLocal(endInput.value);
    if (startIso) params.set("startTs", startIso);
    if (endIso) params.set("endTs", endIso);
    for (const column of state.selectedSourceColumns) {
      params.append("sourceColumns", column);
    }
    return `/ui/data?${params.toString()}`;
  }

  function buildSnapshotBaseParams() {
    const params = new URLSearchParams();
    params.set("region", regionSelect.value);
    const startIso = fromDatetimeLocal(startInput.value);
    const endIso = fromDatetimeLocal(endInput.value);
    if (startIso) params.set("startTs", startIso);
    if (endIso) params.set("endTs", endIso);
    return params;
  }

  async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Request failed" }));
      throw new Error(error.detail || "Request failed");
    }
    return response.json();
  }

  function setActiveMode(mode) {
    state.activeMode = mode;
    const isSnapshot = mode === "snapshot";
    liveTabButton.classList.toggle("active", !isSnapshot);
    snapshotTabButton.classList.toggle("active", isSnapshot);
    liveWorkspace.classList.toggle("active", !isSnapshot);
    snapshotWorkspace.classList.toggle("active", isSnapshot);
    sourcePickerRow.classList.toggle("hidden", isSnapshot);
    modeCopy.textContent = isSnapshot
      ? "Snapshot explainability is FI-only and uses saved issue-time feature rows plus saved model bundles."
      : "Live inspector uses current cached forecast data and raw source series.";
  }

  async function loadAllData() {
    reloadButton.disabled = true;
    reloadButton.textContent = "Loading…";
    try {
      await loadLiveData();
      await loadSnapshotData();
    } finally {
      reloadButton.disabled = false;
      reloadButton.textContent = "Load View";
    }
  }

  async function loadLiveData() {
    try {
      const payload = await fetchJson(buildLiveUrl());
      state.latestPayload = payload;
      regionSelect.value = normalizeRegionValue(payload.region);
      if (!startInput.value) startInput.value = toDatetimeLocal(payload.range.default_start_utc);
      if (!endInput.value) endInput.value = toDatetimeLocal(payload.range.default_end_utc);
      if (!state.selectedSourceColumns.length) {
        state.selectedSourceColumns = payload.selected_source_columns.slice();
      }
      renderLive();
    } catch (error) {
      priceSummary.textContent = error.message;
      sourceSummary.textContent = error.message;
      tableSummary.textContent = error.message;
    }
  }

  async function loadSnapshotData() {
    if (regionSelect.value !== "FI") {
      clearSnapshotUI("Snapshot explainability is available only for FI in this iteration.");
      return;
    }

    try {
      const catalogPayload = await fetchJson(`/ui/api/snapshots?${buildSnapshotBaseParams().toString()}`);
      state.snapshotCatalog = catalogPayload;
      populateGeneratedAtSelect(catalogPayload.runs || []);
      await loadSnapshotSummary();
    } catch (error) {
      clearSnapshotUI(error.message);
    }
  }

  async function loadSnapshotSummary() {
    if (regionSelect.value !== "FI") {
      return;
    }

    const params = buildSnapshotBaseParams();
    const selectedRun = generatedAtSelect.value || "__latest__";
    if (selectedRun === "__earliest__") {
      params.set("selection", "earliest");
    } else if (selectedRun !== "__latest__") {
      params.set("generatedAtUtc", selectedRun);
    }

    const payload = await fetchJson(`/ui/api/explanation-summary?${params.toString()}`);
    state.snapshotSummary = payload;
    renderSnapshotSummary();
  }

  async function loadLocalExplanation() {
    const selected = getSelectedSnapshotRow();
    if (!selected) {
      renderLocalExplanation(null);
      return;
    }

    try {
      const params = new URLSearchParams({
        region: regionSelect.value,
        generatedAtUtc: selected.generated_at_utc,
        targetTimeUtc: selected.time_utc,
      });
      state.localExplanation = await fetchJson(`/ui/api/explanation?${params.toString()}`);
      renderLocalExplanation(state.localExplanation);
    } catch (error) {
      renderLocalExplanation({ explainable: false, reason: error.message });
    }
  }

  async function runScenario(event) {
    event.preventDefault();
    const selected = getSelectedSnapshotRow();
    if (!selected) {
      return;
    }

    scenarioRunButton.disabled = true;
    scenarioRunButton.textContent = "Running…";
    try {
      const payload = {
        region: regionSelect.value,
        generatedAtUtc: selected.generated_at_utc,
        targetTimeUtc: selected.time_utc,
        loadForecastPct: Number(scenarioInputs.loadForecastPct.value || 0),
        windForecastPct: Number(scenarioInputs.windForecastPct.value || 0),
        solarForecastPct: Number(scenarioInputs.solarForecastPct.value || 0),
        importHeadroomMw: Number(scenarioInputs.importHeadroomMw.value || 0),
        crossBorderFlowMw: Number(scenarioInputs.crossBorderFlowMw.value || 0),
        coupledMarketSpreadDelta: Number(scenarioInputs.coupledMarketSpreadDelta.value || 0),
      };
      state.scenarioResult = await fetchJson("/ui/api/scenario", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      renderScenarioResult(state.scenarioResult);
    } catch (error) {
      renderScenarioError(error.message);
    } finally {
      scenarioRunButton.disabled = false;
      scenarioRunButton.textContent = "Run Scenario";
    }
  }

  function setPreset(hours) {
    if (!state.latestPayload) return;
    const end = new Date(state.latestPayload.known_until_utc || state.latestPayload.range.default_end_utc);
    const start = new Date(end.getTime() - hours * 60 * 60 * 1000);
    startInput.value = toDatetimeLocal(start.toISOString());
    endInput.value = toDatetimeLocal(end.toISOString());
  }

  function renderLive() {
    if (!state.latestPayload) return;
    renderStatus();
    renderSourcePicker();
    renderPriceChart();
    renderSourceCharts();
    renderTable();
  }

  function renderStatus() {
    statusRegion.textContent = state.latestPayload.region;
    statusGenerated.textContent = formatLocal(state.latestPayload.generated_at_utc);
    statusKnownUntil.textContent = formatLocal(state.latestPayload.known_until_utc);
    statusOverlap.textContent = String(state.latestPayload.summary.actual_overlap_rows);
    priceSummary.textContent = `${state.latestPayload.summary.price_rows} rows, ${state.latestPayload.summary.actual_overlap_rows} with predicted/actual overlap`;
    sourceSummary.textContent = `${state.latestPayload.summary.source_rows} source rows across ${state.selectedSourceColumns.length} selected series`;
    tableSummary.textContent = `${state.latestPayload.summary.table_rows} merged rows in the selected time window`;
  }

  function renderSourcePicker() {
    sourcePicker.innerHTML = "";
    for (const group of state.latestPayload.source_groups) {
      const fragment = sourceGroupTemplate.content.cloneNode(true);
      const details = fragment.querySelector(".source-group");
      const summary = fragment.querySelector("summary");
      const options = fragment.querySelector(".source-options");
      summary.textContent = `${group.label} (${group.columns.length})`;
      details.open = group.id === "weather" || group.id === "market";

      for (const column of group.columns) {
        const label = document.createElement("label");
        label.className = "source-option";
        const input = document.createElement("input");
        input.type = "checkbox";
        input.value = column.name;
        input.checked = state.selectedSourceColumns.includes(column.name);
        input.addEventListener("change", () => {
          state.selectedSourceColumns = Array.from(sourcePicker.querySelectorAll("input:checked")).map((node) => node.value);
        });
        const text = document.createElement("span");
        text.textContent = column.label;
        label.append(input, text);
        options.appendChild(label);
      }
      sourcePicker.appendChild(fragment);
    }
  }

  function buildSeriesFromRows(rows, keys, colors = palette) {
    return keys.map((key, index) => ({
      key,
      color: colors[index % colors.length],
      values: rows.map((row) => ({
        time: new Date(row.time_utc).getTime(),
        value: row[key],
      })),
    }));
  }

  function renderLegend(container, definitions) {
    let legend = container.querySelector(".chart-legend");
    if (legend) legend.remove();
    legend = document.createElement("div");
    legend.className = "chart-legend";
    for (const definition of definitions) {
      const item = document.createElement("div");
      item.className = "legend-item";
      const swatch = document.createElement("span");
      swatch.className = "legend-swatch";
      swatch.style.background = definition.color;
      const label = document.createElement("span");
      label.textContent = definition.label;
      item.append(swatch, label);
      legend.appendChild(item);
    }
    container.prepend(legend);
  }

  function renderLineChart(svg, series, options = {}) {
    svg.innerHTML = "";
    const width = 1100;
    const height = Number(svg.getAttribute("viewBox").split(" ")[3] || 360);
    const padding = { top: 18, right: 18, bottom: 32, left: 54 };
    const innerWidth = width - padding.left - padding.right;
    const innerHeight = height - padding.top - padding.bottom;
    const values = series.flatMap((entry) => entry.values.map((item) => item.value)).filter((value) => value !== null && value !== undefined && !Number.isNaN(value));
    const times = series.flatMap((entry) => entry.values.map((item) => item.time)).filter(Boolean);
    if (!values.length || !times.length) {
      const empty = document.createElementNS("http://www.w3.org/2000/svg", "text");
      empty.setAttribute("x", "24");
      empty.setAttribute("y", "48");
      empty.setAttribute("class", "chart-label");
      empty.textContent = options.emptyLabel || "No rows available in the selected range.";
      svg.appendChild(empty);
      return;
    }

    let minValue = Math.min(...values);
    let maxValue = Math.max(...values);
    if (minValue === maxValue) {
      minValue -= 1;
      maxValue += 1;
    }

    const minTime = Math.min(...times);
    const maxTime = Math.max(...times);
    const xScale = (time) => padding.left + ((time - minTime) / Math.max(1, maxTime - minTime)) * innerWidth;
    const yScale = (value) => padding.top + (1 - (value - minValue) / (maxValue - minValue)) * innerHeight;

    for (let i = 0; i < 5; i += 1) {
      const y = padding.top + (innerHeight / 4) * i;
      const grid = document.createElementNS("http://www.w3.org/2000/svg", "line");
      grid.setAttribute("x1", String(padding.left));
      grid.setAttribute("x2", String(width - padding.right));
      grid.setAttribute("y1", String(y));
      grid.setAttribute("y2", String(y));
      grid.setAttribute("class", "chart-grid");
      svg.appendChild(grid);

      const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
      label.setAttribute("x", "6");
      label.setAttribute("y", String(y + 4));
      label.setAttribute("class", "chart-label");
      label.textContent = formatNumber(maxValue - ((maxValue - minValue) / 4) * i);
      svg.appendChild(label);
    }

    const axis = document.createElementNS("http://www.w3.org/2000/svg", "line");
    axis.setAttribute("x1", String(padding.left));
    axis.setAttribute("x2", String(width - padding.right));
    axis.setAttribute("y1", String(height - padding.bottom));
    axis.setAttribute("y2", String(height - padding.bottom));
    axis.setAttribute("class", "chart-axis");
    svg.appendChild(axis);

    if (options.focusTime) {
      const focus = document.createElementNS("http://www.w3.org/2000/svg", "line");
      const x = xScale(options.focusTime);
      focus.setAttribute("x1", String(x));
      focus.setAttribute("x2", String(x));
      focus.setAttribute("y1", String(padding.top));
      focus.setAttribute("y2", String(height - padding.bottom));
      focus.setAttribute("class", "chart-focus");
      svg.appendChild(focus);
    }

    for (const entry of series) {
      const points = entry.values.filter((item) => item.value !== null && item.value !== undefined && !Number.isNaN(item.value));
      if (!points.length) continue;
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      const d = points.map((point, pointIndex) => `${pointIndex === 0 ? "M" : "L"} ${xScale(point.time)} ${yScale(point.value)}`).join(" ");
      path.setAttribute("d", d);
      path.setAttribute("stroke", entry.color);
      path.setAttribute("class", "chart-path");
      svg.appendChild(path);
    }

    const startLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    startLabel.setAttribute("x", String(padding.left));
    startLabel.setAttribute("y", String(height - 8));
    startLabel.setAttribute("class", "chart-label");
    startLabel.textContent = formatLocal(new Date(minTime).toISOString());
    svg.appendChild(startLabel);

    const endLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    endLabel.setAttribute("x", String(width - padding.right));
    endLabel.setAttribute("y", String(height - 8));
    endLabel.setAttribute("text-anchor", "end");
    endLabel.setAttribute("class", "chart-label");
    endLabel.textContent = formatLocal(new Date(maxTime).toISOString());
    svg.appendChild(endLabel);
  }

  function renderPriceChart() {
    const rows = state.latestPayload.price_rows || [];
    const definitions = [
      { key: "predicted_price", label: "Predicted", color: "#0a6c74" },
      { key: "served_price", label: "Served", color: "#9b2226" },
      { key: "actual_price", label: "Actual", color: "#111111" },
    ];
    renderLegend(priceChart.parentElement, definitions);
    renderLineChart(
      priceChart,
      definitions.map((definition) => ({
        key: definition.key,
        color: definition.color,
        values: rows.map((row) => ({
          time: new Date(row.time_utc).getTime(),
          value: row[definition.key],
        })),
      })),
      { emptyLabel: "No price rows available in the selected range." },
    );
  }

  function renderSourceCharts() {
    sourceCharts.innerHTML = "";
    if (!state.selectedSourceColumns.length) {
      sourceCharts.appendChild(buildEmptyState("Select at least one source series."));
      return;
    }
    const rows = state.latestPayload.source_rows || [];
    for (const [index, column] of state.selectedSourceColumns.entries()) {
      const block = document.createElement("article");
      block.className = "mini-chart";
      const title = document.createElement("h3");
      title.textContent = column.replaceAll("_", " ");
      const subtitle = document.createElement("p");
      subtitle.textContent = "Raw live source series";
      const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.setAttribute("viewBox", "0 0 520 180");
      svg.setAttribute("preserveAspectRatio", "none");
      block.append(title, subtitle, svg);
      sourceCharts.appendChild(block);
      renderLineChart(
        svg,
        buildSeriesFromRows(rows, [column], [palette[index % palette.length]]),
        { emptyLabel: "No source rows in range." },
      );
    }
  }

  function renderTable() {
    const rows = state.latestPayload.table_rows || [];
    renderSimpleTable(dataTable, ["time_utc", "predicted_price", "served_price", "actual_price", ...state.selectedSourceColumns], rows, {
      headers: {
        time_utc: "Time",
        predicted_price: "Predicted",
        served_price: "Served",
        actual_price: "Actual",
      },
      formatters: {
        time_utc: formatLocal,
      },
      limit: 400,
    });
  }

  function populateGeneratedAtSelect(runs) {
    const previous = generatedAtSelect.value || "__latest__";
    generatedAtSelect.innerHTML = "";
    generatedAtSelect.appendChild(createOption("__latest__", "Latest per target"));
    for (const run of runs) {
      const label = `${formatLocal(run.generated_at_utc)} (${run.explainable_row_count}/${run.row_count} explainable)`;
      generatedAtSelect.appendChild(createOption(run.generated_at_utc, label));
    }
    if (Array.from(generatedAtSelect.options).some((option) => option.value === previous)) {
      generatedAtSelect.value = previous;
    } else {
      generatedAtSelect.value = "__latest__";
    }
  }

  function renderSnapshotSummary() {
    const payload = state.snapshotSummary;
    if (!payload) {
      clearSnapshotUI("No snapshot rows available for this window.");
      return;
    }

    snapshotSelectionSummary.textContent = `${payload.rows.length} selected rows, ${payload.explainable_rows} explainable with actual prices`;
    snapshotSummary.textContent = `${payload.rows_evaluated} rows with actual prices included in historical summary`;
    renderRunMeta(payload);
    populateTargetTimeSelect(payload.rows || []);
    renderSnapshotChart(payload.rows || []);
    renderHistoricalSummary(payload);
    loadLocalExplanation();
  }

  function renderRunMeta(payload) {
    snapshotRunMeta.innerHTML = "";
    const items = [
      { label: "Selection", value: payload.selected_generated_at_utc ? formatLocal(payload.selected_generated_at_utc) : "Latest per target" },
      { label: "Rows", value: String((payload.rows || []).length) },
      { label: "Explainable", value: String(payload.explainable_rows || 0) },
      { label: "Runs", value: String((payload.runs || []).length) },
    ];
    for (const item of items) {
      snapshotRunMeta.appendChild(buildMetricCard(item.label, item.value));
    }
  }

  function populateTargetTimeSelect(rows) {
    const previous = targetTimeSelect.value;
    targetTimeSelect.innerHTML = "";
    for (const row of rows) {
      const suffix = row.generated_at_utc ? ` | run ${formatLocal(row.generated_at_utc)}` : "";
      const label = `${formatLocal(row.time_utc)}${suffix}${row.explainable ? "" : " | not explainable"}`;
      targetTimeSelect.appendChild(createOption(snapshotRowKey(row), label));
    }
    if (previous && Array.from(targetTimeSelect.options).some((option) => option.value === previous)) {
      targetTimeSelect.value = previous;
      return;
    }
    const explainable = rows.find((row) => row.explainable);
    targetTimeSelect.value = explainable ? snapshotRowKey(explainable) : rows[0] ? snapshotRowKey(rows[0]) : "";
  }

  function renderSnapshotChart(rows) {
    const definitions = [
      { key: "predicted_price", label: "Selected snapshot", color: "#0a6c74" },
      { key: "actual_price", label: "Actual", color: "#111111" },
    ];
    renderLegend(snapshotChart.parentElement, definitions);
    const selected = getSelectedSnapshotRow();
    renderLineChart(
      snapshotChart,
      definitions.map((definition) => ({
        key: definition.key,
        color: definition.color,
        values: rows.map((row) => ({
          time: new Date(row.time_utc).getTime(),
          value: row[definition.key],
        })),
      })),
      {
        emptyLabel: "No snapshot rows available in the selected range.",
        focusTime: selected ? new Date(selected.time_utc).getTime() : null,
      },
    );
  }

  function renderLocalExplanation(payload) {
    localMetrics.innerHTML = "";
    groupContributionBars.innerHTML = "";
    localAdjustments.innerHTML = "";
    if (!payload || !payload.explainable) {
      localExplainSummary.textContent = payload && payload.reason ? payload.reason : "Choose an explainable snapshot row.";
      featureContributionTable.querySelector("thead").innerHTML = "";
      featureContributionTable.querySelector("tbody").innerHTML = "";
      localMetrics.appendChild(buildMetricCard("Status", "Unavailable"));
      localAdjustments.appendChild(buildEmptyState(payload && payload.reason ? payload.reason : "No explainable row selected."));
      scenarioRunButton.disabled = true;
      return;
    }

    localExplainSummary.textContent = `${formatLocal(payload.target_time_utc)} from run ${formatLocal(payload.generated_at_utc)}`;
    localMetrics.appendChild(buildMetricCard("Predicted", formatNumber(payload.predicted_price)));
    localMetrics.appendChild(buildMetricCard("Actual", formatNumber(payload.actual_price)));
    localMetrics.appendChild(buildMetricCard("Base Value", formatNumber(payload.base_value)));
    localMetrics.appendChild(buildMetricCard("Reconstructed", formatNumber(payload.reconstructed_prediction)));

    renderBarList(groupContributionBars, payload.group_contributions || [], {
      labelKey: "group_label",
      valueKey: "signed_contribution",
      absolute: false,
    });
    renderAdjustmentList(localAdjustments, payload.adjustments || []);
    renderSimpleTable(featureContributionTable, ["absolute_rank", "feature_name", "group_label", "feature_value", "signed_contribution"], payload.feature_contributions || [], {
      headers: {
        absolute_rank: "Rank",
        feature_name: "Feature",
        group_label: "Group",
        feature_value: "Value",
        signed_contribution: "Contribution",
      },
      formatters: {
        feature_value: (value) => formatNumber(value, 3),
        signed_contribution: (value) => formatSigned(value, 3),
      },
      limit: 40,
    });
    scenarioRunButton.disabled = false;
  }

  function renderHistoricalSummary(payload) {
    historicalSummary.textContent = `${payload.rows_evaluated} explainable rows with actual prices in the selected window`;
    renderBarList(summaryGroupBars, payload.group_summary || [], {
      labelKey: "group_label",
      valueKey: "mean_abs_contribution",
      absolute: true,
    });
    renderSimpleTable(summaryFeatureTable, ["name", "group_label", "mean_abs_contribution", "mean_signed_contribution"], payload.feature_summary || [], {
      headers: {
        name: "Feature",
        group_label: "Group",
        mean_abs_contribution: "Mean Abs",
        mean_signed_contribution: "Mean Signed",
      },
      formatters: {
        mean_abs_contribution: (value) => formatNumber(value, 3),
        mean_signed_contribution: (value) => formatSigned(value, 3),
      },
      limit: 20,
    });

    errorSliceCards.innerHTML = "";
    const slices = payload.error_slices || {};
    const names = ["low", "medium", "high"];
    for (const name of names) {
      if (!slices[name]) continue;
      const card = document.createElement("article");
      card.className = "slice-card";
      const title = document.createElement("h3");
      title.textContent = `${name.replace("_", " ")} error`;
      const body = document.createElement("p");
      body.textContent = `${slices[name].rows} rows. Strongest average group drivers in this slice.`;
      card.append(title, body);
      for (const item of slices[name].group_summary || []) {
        const line = document.createElement("div");
        line.className = "adjustment-item";
        const label = document.createElement("span");
        label.className = "label";
        label.textContent = item.group_label;
        const value = document.createElement("span");
        value.className = "value";
        value.textContent = formatNumber(item.mean_abs_contribution, 3);
        line.append(label, value);
        card.appendChild(line);
      }
      errorSliceCards.appendChild(card);
    }
    if (!errorSliceCards.children.length) {
      errorSliceCards.appendChild(buildEmptyState("No error-slice summary available yet."));
    }
  }

  function renderScenarioResult(payload) {
    if (!payload || !payload.explainable) {
      renderScenarioError(payload && payload.reason ? payload.reason : "Scenario unavailable for this row.");
      return;
    }
    scenarioSummary.textContent = "Sensitivity analysis for the selected snapshot row.";
    scenarioMetrics.innerHTML = "";
    scenarioMetrics.appendChild(buildMetricCard("Baseline", formatNumber(payload.baseline_prediction)));
    scenarioMetrics.appendChild(buildMetricCard("Scenario", formatNumber(payload.scenario_prediction)));
    scenarioMetrics.appendChild(buildMetricCard("Delta", formatSigned(payload.prediction_delta, 3)));

    renderBarList(scenarioGroupBars, payload.group_deltas || [], {
      labelKey: "label",
      valueKey: "delta",
      absolute: false,
      limit: 8,
    });
    renderAdjustmentList(scenarioAppliedInputs, payload.applied_inputs || [], {
      labelKey: "label",
      valueKey: "value",
      formatter: (item) => `${item.applied ? "Applied" : "Skipped"}${item.reason ? ` | ${item.reason}` : ""}`,
    });
    renderSimpleTable(scenarioChangedFeaturesTable, ["feature_name", "group_label", "before", "after", "delta"], payload.changed_features || [], {
      headers: {
        feature_name: "Feature",
        group_label: "Group",
        before: "Before",
        after: "After",
        delta: "Delta",
      },
      formatters: {
        before: (value) => formatNumber(value, 3),
        after: (value) => formatNumber(value, 3),
        delta: (value) => formatSigned(value, 3),
      },
      limit: 20,
    });
  }

  function renderScenarioError(message) {
    scenarioSummary.textContent = message;
    scenarioMetrics.innerHTML = "";
    scenarioGroupBars.innerHTML = "";
    scenarioAppliedInputs.innerHTML = "";
    scenarioChangedFeaturesTable.querySelector("thead").innerHTML = "";
    scenarioChangedFeaturesTable.querySelector("tbody").innerHTML = "";
    scenarioMetrics.appendChild(buildMetricCard("Scenario", "Unavailable"));
    scenarioGroupBars.appendChild(buildEmptyState(message));
  }

  function clearSnapshotUI(message) {
    state.snapshotCatalog = null;
    state.snapshotSummary = null;
    state.localExplanation = null;
    state.scenarioResult = null;
    snapshotSelectionSummary.textContent = message;
    snapshotSummary.textContent = message;
    snapshotRunMeta.innerHTML = "";
    generatedAtSelect.innerHTML = "";
    targetTimeSelect.innerHTML = "";
    snapshotChart.innerHTML = "";
    renderLocalExplanation({ explainable: false, reason: message });
    historicalSummary.textContent = message;
    summaryGroupBars.innerHTML = "";
    errorSliceCards.innerHTML = "";
    summaryFeatureTable.querySelector("thead").innerHTML = "";
    summaryFeatureTable.querySelector("tbody").innerHTML = "";
    renderScenarioError(message);
  }

  function getSelectedSnapshotRow() {
    if (!state.snapshotSummary || !state.snapshotSummary.rows) return null;
    const selectedKey = targetTimeSelect.value;
    return state.snapshotSummary.rows.find((row) => snapshotRowKey(row) === selectedKey) || null;
  }

  function snapshotRowKey(row) {
    return `${row.generated_at_utc}__${row.time_utc}`;
  }

  function renderBarList(container, rows, options = {}) {
    container.innerHTML = "";
    const limitedRows = (rows || []).slice(0, options.limit || 12);
    if (!limitedRows.length) {
      container.appendChild(buildEmptyState("No contribution data available."));
      return;
    }
    const values = limitedRows.map((row) => Math.abs(Number(row[options.valueKey] || 0)));
    const maxAbs = Math.max(...values, 1);
    for (const row of limitedRows) {
      const entry = document.createElement("div");
      entry.className = "bar-row";
      const label = document.createElement("div");
      label.className = "bar-row-label";
      label.textContent = row[options.labelKey];
      const track = document.createElement("div");
      track.className = `bar-track${options.absolute ? " absolute" : ""}`;
      const fill = document.createElement("div");
      const value = Number(row[options.valueKey] || 0);
      const width = `${Math.max((Math.abs(value) / maxAbs) * (options.absolute ? 100 : 50), 1.5)}%`;
      fill.className = `bar-fill ${value < 0 && !options.absolute ? "negative" : "positive"}`;
      fill.style.width = width;
      track.appendChild(fill);
      const text = document.createElement("div");
      text.className = "bar-value";
      text.textContent = options.absolute ? formatNumber(value, 3) : formatSigned(value, 3);
      entry.append(label, track, text);
      container.appendChild(entry);
    }
  }

  function renderAdjustmentList(container, rows, options = {}) {
    container.innerHTML = "";
    if (!rows || !rows.length) {
      container.appendChild(buildEmptyState("No adjustments to show."));
      return;
    }
    for (const row of rows) {
      const item = document.createElement("div");
      item.className = "adjustment-item";
      const label = document.createElement("span");
      label.className = "label";
      label.textContent = row[options.labelKey || "label"];
      const value = document.createElement("span");
      value.className = "value";
      if (options.formatter) {
        value.textContent = options.formatter(row);
      } else if (typeof row.signed_contribution === "number") {
        value.textContent = formatSigned(row.signed_contribution, 3);
      } else {
        value.textContent = formatNumber(row[options.valueKey || "value"], 3);
      }
      item.append(label, value);
      container.appendChild(item);
    }
  }

  function renderSimpleTable(table, columns, rows, options = {}) {
    const head = table.querySelector("thead");
    const body = table.querySelector("tbody");
    head.innerHTML = "";
    body.innerHTML = "";
    const headerRow = document.createElement("tr");
    for (const column of columns) {
      const th = document.createElement("th");
      th.textContent = options.headers && options.headers[column] ? options.headers[column] : column.replaceAll("_", " ");
      headerRow.appendChild(th);
    }
    head.appendChild(headerRow);

    const limitedRows = (rows || []).slice(0, options.limit || 400);
    for (const row of limitedRows) {
      const tr = document.createElement("tr");
      for (const column of columns) {
        const td = document.createElement("td");
        const formatter = options.formatters && options.formatters[column];
        const rawValue = row[column];
        if (formatter) {
          td.textContent = formatter(rawValue);
        } else if (typeof rawValue === "number") {
          td.textContent = formatNumber(rawValue);
        } else {
          td.textContent = rawValue === null || rawValue === undefined ? "–" : String(rawValue);
        }
        tr.appendChild(td);
      }
      body.appendChild(tr);
    }
  }

  function buildMetricCard(label, value) {
    const card = document.createElement("div");
    card.className = "metric-card";
    const span = document.createElement("span");
    span.textContent = label;
    const strong = document.createElement("strong");
    strong.textContent = value;
    card.append(span, strong);
    return card;
  }

  function buildEmptyState(text) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = text;
    return empty;
  }

  function createOption(value, label) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    return option;
  }

  reloadButton.addEventListener("click", loadAllData);
  preset48h.addEventListener("click", () => {
    setPreset(48);
    loadAllData();
  });
  preset7d.addEventListener("click", () => {
    setPreset(24 * 7);
    loadAllData();
  });
  regionSelect.addEventListener("change", () => {
    startInput.value = "";
    endInput.value = "";
    state.selectedSourceColumns = [];
    loadAllData();
  });
  liveTabButton.addEventListener("click", () => setActiveMode("live"));
  snapshotTabButton.addEventListener("click", () => setActiveMode("snapshot"));
  generatedAtSelect.addEventListener("change", loadSnapshotSummary);
  targetTimeSelect.addEventListener("change", () => {
    renderSnapshotChart(state.snapshotSummary ? state.snapshotSummary.rows || [] : []);
    loadLocalExplanation();
  });
  scenarioForm.addEventListener("submit", runScenario);

  const initialRegion = new URLSearchParams(window.location.search).get("region");
  if (initialRegion) regionSelect.value = initialRegion;
  setActiveMode("live");
  loadAllData();
})();
