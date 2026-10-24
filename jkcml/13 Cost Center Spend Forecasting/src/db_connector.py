"""
Database Connector Module - Use Case 13: Cost Center Spend Forecasting (Time Series)
Handles connection and data extraction from PostgreSQL for SAP HANA Controlling & Financial
Accounting Cost Center Spend Data across 2025 (Training Horizon) and 2026 (Forecast Horizon).
Extracts Cost Center Document Items (RSEG/RBCO), Invoice Headers & Dates (RBKP),
Account Assignments (EKKN), Cost Center Master (CSKS), G/L Account Master & Texts (SKAT, SKA1),
Plant Maintenance Orders (AUFK), Plant Master (T001W), Company Code Master (T001), and Vendor Master (LFA1).
"""
import os
import yaml
import psycopg2
import pandas as pd
from sqlalchemy import create_engine


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config", "db_config.yaml")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_db_engine(config=None):
    """Create SQLAlchemy engine using loaded configuration."""
    if config is None:
        config = load_config()
    db = config["database"]
    uri = f"postgresql+psycopg2://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['dbname']}"
    return create_engine(uri)


def get_db_connection(config=None):
    """Create raw psycopg2 connection."""
    if config is None:
        config = load_config()
    db = config["database"]
    return psycopg2.connect(
        host=db["host"],
        port=db["port"],
        dbname=db["dbname"],
        user=db["user"],
        password=db["password"]
    )


def extract_cost_center_spend_data(config=None):
    """
    Extracts SAP Cost Center Spend Time-Series Data spanning 2025 (1-year history) and 2026 (forecast horizon).
    Stratifies monthly spend from RSEG and RBKP joined with EKKN and CSKS to ensure complete temporal continuity across:
    - 2025 Training Set: 2025-01 through 2025-12 (~16,400 records)
    - 2026 Forecast Horizon: 2026-01 through 2026-07 (~6,800 records)
    Enriched with:
    - CSKS: Cost Center Name, Description, Department, Cost Center Category
    - SKAT & SKA1: G/L short/long descriptions and account group/type
    - EKKN: PO Account assignment, commitment values, and distinct orders
    - AUFK: Maintenance & internal order count per Cost Center
    - T001W, T001, LFA1: Plant, Company Code, and Vendor metadata
    """
    if config is None:
        config = load_config()
    schema = config["database"].get("schema", "HANA_P3")
    engine = get_db_engine(config)

    ts_params = config.get("time_series_parameters", {})
    train_year = str(ts_params.get("train_year", 2025))
    forecast_year = str(ts_params.get("forecast_year", 2026))
    limit_train = ts_params.get("sample_size_per_month_2025", 1600)
    limit_forecast = ts_params.get("sample_size_per_month_2026", 1600)

    query = f"""
    WITH ekkn_clean AS (
        SELECT DISTINCT ON ("EBELN", "EBELP")
            "EBELN", "EBELP", 
            TRIM("KOSTL") AS kostl_key, 
            TRIM("SAKTO") AS sakto_key, 
            TRIM("AUFNR") AS aufnr_key, 
            "PRCTR"
        FROM "{schema}"."EKKN"
        WHERE "KOSTL" IS NOT NULL AND TRIM("KOSTL") != ''
        ORDER BY "EBELN", "EBELP", "ZEKKN" ASC
    ),
    csks_clean AS (
        SELECT DISTINCT ON ("KOSTL")
            TRIM("KOSTL") AS kostl_key,
            "Name" AS cost_center_name,
            "Description" AS cost_center_desc,
            "ABTEI" AS department_code,
            "KOSAR" AS cc_category,
            "BUKRS" AS bukrs_key
        FROM "{schema}"."CSKS"
        ORDER BY "KOSTL", "DATBI" DESC
    ),
    skat_clean AS (
        SELECT DISTINCT ON ("SAKNR")
            TRIM("SAKNR") AS saknr_key,
            "TXT20" AS gl_short_text,
            "TXT50" AS gl_long_text
        FROM "{schema}"."SKAT"
        WHERE "SPRAS" = 'E'
        ORDER BY "SAKNR"
    ),
    ska1_clean AS (
        SELECT DISTINCT ON ("SAKNR")
            TRIM("SAKNR") AS saknr_key,
            "KTOKS" AS gl_account_group,
            "GLACCOUNT_TYPE" AS gl_account_type,
            "FUNC_AREA" AS gl_func_area
        FROM "{schema}"."SKA1"
        ORDER BY "SAKNR"
    ),
    aufk_clean AS (
        SELECT DISTINCT ON ("AUFNR")
            TRIM("AUFNR") AS aufnr_key,
            "KTEXT" AS order_desc,
            "AUART" AS order_type
        FROM "{schema}"."AUFK"
        ORDER BY "AUFNR"
    ),
    t001w_clean AS (
        SELECT DISTINCT ON ("WERKS")
            TRIM("WERKS") AS werks_key,
            "NAME1" AS plant_name,
            "ORT01" AS plant_city
        FROM "{schema}"."T001W"
        ORDER BY "WERKS"
    ),
    lfa1_clean AS (
        SELECT DISTINCT ON ("LIFNR")
            TRIM("LIFNR") AS lifnr_key,
            "NAME1" AS vendor_name
        FROM "{schema}"."LFA1"
        ORDER BY "LIFNR"
    ),
    t001_clean AS (
        SELECT DISTINCT ON ("BUKRS")
            TRIM("BUKRS") AS bukrs_key,
            "BUTXT" AS company_name,
            "ORT01" AS company_city,
            "WAERS" AS company_currency
        FROM "{schema}"."T001"
        ORDER BY "BUKRS"
    ),
    ekkn_commitments AS (
        SELECT 
            TRIM("KOSTL") AS kostl_key,
            COUNT(*) AS po_commitment_items,
            COUNT(DISTINCT "EBELN") AS po_distinct_orders,
            SUM(COALESCE(NULLIF(REGEXP_REPLACE("NETWR"::text, '[^0-9.-]', '', 'g'), '')::numeric, 0.0)) AS po_commitment_value_inr
        FROM "{schema}"."EKKN"
        WHERE "KOSTL" IS NOT NULL AND TRIM("KOSTL") != ''
        GROUP BY TRIM("KOSTL")
    ),
    aufk_summary AS (
        SELECT 
            TRIM(COALESCE(NULLIF("KOSTL", ''), "KOSTV")) AS kostl_key,
            COUNT(*) AS pm_order_count,
            COUNT(DISTINCT "AUART") AS pm_order_types_count
        FROM "{schema}"."AUFK"
        WHERE (COALESCE(NULLIF("KOSTL", ''), "KOSTV") IS NOT NULL) 
          AND TRIM(COALESCE(NULLIF("KOSTL", ''), "KOSTV")) != ''
        GROUP BY TRIM(COALESCE(NULLIF("KOSTL", ''), "KOSTV"))
    ),
    rseg_ekkn AS (
        SELECT 
            r."BELNR" AS doc_number,
            r."GJAHR" AS fiscal_year,
            COALESCE(r."BUZEI"::text, '1') AS line_item,
            r."EBELN" AS po_number,
            r."EBELP" AS po_item,
            r."WERKS" AS plant,
            r."SHKZG" AS debit_credit_ind,
            COALESCE(NULLIF(REGEXP_REPLACE(r."WRBTR", '[^0-9.-]', '', 'g'), '')::numeric, 0.0) AS spend_amount_raw,
            k."BUDAT" AS posting_date,
            k."BLDAT" AS document_date,
            SUBSTRING(k."BUDAT", 1, 7) AS posting_month,
            k."LIFNR" AS vendor_code,
            COALESCE(k."WAERS", 'INR') AS currency,
            COALESCE(k."XBLNR", '') AS invoice_ref,
            COALESCE(k."BUKRS", '1000') AS company_code,
            e.kostl_key,
            e.sakto_key,
            e.aufnr_key,
            ROW_NUMBER() OVER (PARTITION BY SUBSTRING(k."BUDAT", 1, 7) ORDER BY r."BELNR") AS month_rank
        FROM "{schema}"."RSEG" r
        JOIN "{schema}"."RBKP" k ON r."BELNR" = k."BELNR" AND r."GJAHR" = k."GJAHR"
        JOIN ekkn_clean e ON r."EBELN" = e."EBELN" AND r."EBELP" = e."EBELP"
        WHERE r."GJAHR" IN ('{train_year}', '{forecast_year}')
          AND k."BUDAT" IS NOT NULL
          AND NULLIF(REGEXP_REPLACE(r."WRBTR", '[^0-9.-]', '', 'g'), '')::numeric > 0
    )
    SELECT 
        p.doc_number,
        p.fiscal_year,
        p.line_item,
        p.po_number,
        p.po_item,
        p.kostl_key AS cost_center,
        COALESCE(c.cost_center_name, c.cost_center_desc, 'Cost Center ' || p.kostl_key) AS cost_center_name,
        COALESCE(c.department_code, c.cc_category, 'GEN') AS department_code,
        COALESCE(p.sakto_key, '') AS gl_account,
        COALESCE(sk.gl_short_text, '') AS gl_short_text,
        COALESCE(sk.gl_long_text, sk.gl_short_text, 'G/L ' || p.sakto_key) AS gl_description,
        COALESCE(sa.gl_account_group, '') AS gl_account_group,
        COALESCE(sa.gl_account_type, '') AS gl_account_type,
        COALESCE(p.plant, '1000') AS plant,
        COALESCE(w.plant_name, 'Plant ' || p.plant) AS plant_name,
        COALESCE(w.plant_city, '') AS plant_city,
        COALESCE(comp.company_name, 'J.K. CEMENT LIMITED') AS company_name,
        COALESCE(comp.company_city, 'Kanpur') AS company_city,
        p.debit_credit_ind,
        p.spend_amount_raw,
        p.posting_date,
        p.document_date,
        p.posting_month,
        COALESCE(p.vendor_code, '') AS vendor_code,
        COALESCE(l.vendor_name, 'Vendor ' || p.vendor_code) AS vendor_name,
        p.invoice_ref,
        p.company_code,
        p.currency,
        p.aufnr_key AS order_id,
        COALESCE(a.order_desc, '') AS order_desc,
        COALESCE(ekc.po_commitment_items, 0) AS cc_po_commitment_items,
        COALESCE(ekc.po_distinct_orders, 0) AS cc_po_distinct_orders,
        COALESCE(ekc.po_commitment_value_inr, 0.0) AS cc_po_commitment_value_inr,
        COALESCE(ak.pm_order_count, 0) AS cc_pm_order_count,
        COALESCE(ak.pm_order_types_count, 0) AS cc_pm_order_types_count
    FROM rseg_ekkn p
    LEFT JOIN csks_clean c ON p.kostl_key = c.kostl_key
    LEFT JOIN skat_clean sk ON p.sakto_key = sk.saknr_key
    LEFT JOIN ska1_clean sa ON p.sakto_key = sa.saknr_key
    LEFT JOIN aufk_clean a ON p.aufnr_key = a.aufnr_key
    LEFT JOIN t001w_clean w ON TRIM(p.plant) = w.werks_key
    LEFT JOIN lfa1_clean l ON TRIM(p.vendor_code) = l.lifnr_key
    LEFT JOIN t001_clean comp ON TRIM(p.company_code) = comp.bukrs_key
    LEFT JOIN ekkn_commitments ekc ON p.kostl_key = ekc.kostl_key
    LEFT JOIN aufk_summary ak ON p.kostl_key = ak.kostl_key
    WHERE (p.fiscal_year = '{train_year}' AND p.month_rank <= {limit_train})
       OR (p.fiscal_year = '{forecast_year}' AND p.month_rank <= {limit_forecast})
    ORDER BY p.posting_date ASC, p.doc_number ASC;
    """

    print(f"[INFO] Extracting SAP Cost Center Spend Time-Series Data ({train_year} Training Set + {forecast_year} Forecast Horizon) from {schema}...")
    df = pd.read_sql(query, engine)
    print(f"[SUCCESS] Extracted {len(df):,} Cost Center Spend records across {df['posting_date'].nunique()} posting dates directly from server.")
    return df


# Aliases for seamless compatibility
extract_departmental_spend_data = extract_cost_center_spend_data
extract_cost_center_timeseries_data = extract_cost_center_spend_data
