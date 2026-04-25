import logging
import os

import duckdb

log = logging.getLogger(__name__)

class DatabaseManager:
    """Manages the DuckDB connection and structural schemas for the snapshot-based forecast pipeline."""
    
    def __init__(self, db_path: str = "data/forecasts.duckdb"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        # Establish connection. Note: in a multi-process environment (like uvicorn workers), 
        # DuckDB should be connected in read_only mode or a single central writer process should be used.
        # For this prototype we assume standard single-writer access.
        self.conn = duckdb.connect(self.db_path)
        self._initialize_schema()

    def _initialize_schema(self):
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_source_snapshots")
        
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS source_snapshots (
            id BIGINT DEFAULT nextval('seq_source_snapshots'),
            source VARCHAR,
            dataset_id VARCHAR,
            as_of_timestamp TIMESTAMPTZ NOT NULL,
            delivery_start_utc TIMESTAMPTZ NOT NULL,
            area VARCHAR,
            value DOUBLE,
            unit VARCHAR,
            payload_hash VARCHAR
        );
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_as_of ON source_snapshots (dataset_id, delivery_start_utc, as_of_timestamp DESC);")

        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS market_prices (
            area VARCHAR,
            delivery_start_utc TIMESTAMPTZ,
            market_day_cet DATE,
            resolution_min INTEGER, 
            price_eur_mwh DOUBLE,
            published_at TIMESTAMPTZ,
            source VARCHAR,
            PRIMARY KEY (area, delivery_start_utc)
        );
        """)

        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS features_fi_dayahead (
            forecast_as_of TIMESTAMPTZ,
            delivery_start_utc TIMESTAMPTZ,
            qh_index INTEGER CHECK (qh_index >= 1 AND qh_index <= 96),
            
            -- History
            fi_price_d1_q DOUBLE, fi_price_d2_q DOUBLE, fi_price_d7_q DOUBLE,
            fi_price_d1_qm1 DOUBLE, fi_price_d1_qp1 DOUBLE,
            fi_minus_se1_d1_q DOUBLE, fi_minus_se3_d1_q DOUBLE, fi_minus_ee_d1_q DOUBLE,
            fi_price_d1_max DOUBLE, fi_price_d1_min DOUBLE, fi_price_d1_mean DOUBLE,
            
            -- Forecasts
            fi_load_fc_q DOUBLE, fi_wind_fc_q DOUBLE, fi_solar_fc_q DOUBLE, net_load_fc_q DOUBLE,
            load_ramp_1q DOUBLE, wind_ramp_1q DOUBLE,
            
            -- Coupled Spatials
            flow_se1_d1_q DOUBLE, flow_se3_d1_q DOUBLE, flow_ee_d1_q DOUBLE,
            cap_fi_se1_q DOUBLE, cap_se1_fi_q DOUBLE, 
            cap_balance_se1_q DOUBLE, cap_balance_se3_q DOUBLE, cap_balance_ee_q DOUBLE,
            
            -- Operational
            ol3_protection_fc_q DOUBLE,
            
            -- Calendar
            day_of_week INTEGER, is_weekend BOOLEAN, month INTEGER, is_fi_holiday BOOLEAN,
            
            PRIMARY KEY (forecast_as_of, delivery_start_utc)
        );
        """)
        
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS model_runs (
            model_name VARCHAR,
            trained_at TIMESTAMPTZ,
            train_start TIMESTAMPTZ,
            train_end TIMESTAMPTZ,
            params_json VARCHAR,
            artifact_path VARCHAR,
            PRIMARY KEY (model_name, trained_at)
        );
        """)

        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS forecasts (
            model_name VARCHAR,
            forecast_as_of TIMESTAMPTZ,
            delivery_start_utc TIMESTAMPTZ,
            p50 DOUBLE,
            p10 DOUBLE,
            p90 DOUBLE,
            version VARCHAR,
            PRIMARY KEY (model_name, forecast_as_of, delivery_start_utc)
        );
        """)

        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_results (
            model_name VARCHAR,
            run_id VARCHAR,
            metric_name VARCHAR,
            metric_value DOUBLE,
            slice_json VARCHAR
        );
        """)
        log.info("DuckDB schema initialized at %s", self.db_path)

    def close(self):
        self.conn.close()

# Shared global access if needed
_db_instance = None

def get_db() -> DatabaseManager:
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance
