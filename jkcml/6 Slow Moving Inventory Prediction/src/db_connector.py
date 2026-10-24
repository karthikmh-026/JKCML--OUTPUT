"""
Database Connector Module - Use Case 6: Slow Moving Inventory Prediction
Handles robust and high-speed data extraction directly from PostgreSQL (192.168.1.44:5432 / JKC_Input).
Extracts material valuation (MBEW), material master (MARA), material descriptions (MAKT),
plant material parameters (MARC), plant master (T001W), and aggregated goods movements (MSEG)
across BOTH schemas: HANA_P2 and HANA_P3 with NO row limits.
Implements local persistent caching in data/cache/ for high-speed repeatability.
"""
import os
import yaml
import time
import psycopg2
import pandas as pd
from sqlalchemy import create_engine


def load_config(config_path: str = None) -> dict:
    """Load database configuration from YAML file."""
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config", "db_config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_db_engine(config: dict = None):
    """Create SQLAlchemy engine using loaded configuration."""
    if config is None:
        config = load_config()
    db = config["database"]
    uri = f"postgresql+psycopg2://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['dbname']}"
    return create_engine(uri)


def get_db_connection(config: dict = None):
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


def extract_slow_moving_data(
    limit: int = None,
    config: dict = None,
    use_cache: bool = True,
    cache_path: str = None
) -> pd.DataFrame:
    """
    Extracts SAP Material Inventory, Valuation, and Movement history from BOTH HANA_P2 and HANA_P3 schemas.
    Applies high-performance aggregation and deduplicated joins on MBEW, MARA, MAKT, MARC, T001W, and MSEG.
    If limit is None, extracts ALL records without limitation.
    Caches extracted data to data/cache/slow_moving_inventory_raw_data.csv.
    """
    if cache_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_path = os.path.join(base_dir, "data", "cache", "slow_moving_inventory_raw_data.csv")

    if use_cache and os.path.exists(cache_path):
        print(f"[CACHE] Loading cached SAP inventory dataset from {cache_path}...")
        df = pd.read_csv(cache_path, low_memory=False)
        if limit is None or len(df) >= (limit or 0):
            print(f"[CACHE] Loaded {len(df):,} records successfully from cache.")
            return df

    print("[DB] Connecting to PostgreSQL (192.168.1.44:5432 / JKC_Input / HANA_P2 & HANA_P3)...")
    t0 = time.time()
    conn = get_db_connection(config)
    cur = conn.cursor()

    limit_clause = f"LIMIT {limit}" if limit is not None else ""

    query = f"""
    WITH p2_mseg AS (
        SELECT 
            "MATNR",
            "WERKS",
            MAX(CASE WHEN "BWART" IN ('601', '261', '201', '221', '641') AND "SHKZG" = 'H' THEN "BUDAT_MKPF" END) AS last_issue_date,
            MAX(CASE WHEN "BWART" IN ('101', '105') AND "SHKZG" = 'S' THEN "BUDAT_MKPF" END) AS last_receipt_date,
            MAX("BUDAT_MKPF") AS last_movement_date,
            SUM(CASE WHEN "BWART" IN ('601', '261', '201', '221', '641') AND "SHKZG" = 'H' THEN CAST("MENGE" AS NUMERIC) ELSE 0 END) AS total_issue_qty,
            SUM(CASE WHEN "BWART" IN ('601', '261', '201', '221', '641') AND "SHKZG" = 'H' THEN CAST("DMBTR" AS NUMERIC) ELSE 0 END) AS total_issue_val,
            SUM(CASE WHEN "BWART" IN ('101', '105') AND "SHKZG" = 'S' THEN CAST("MENGE" AS NUMERIC) ELSE 0 END) AS total_receipt_qty,
            SUM(CASE WHEN "BWART" IN ('101', '105') AND "SHKZG" = 'S' THEN CAST("DMBTR" AS NUMERIC) ELSE 0 END) AS total_receipt_val,
            COUNT(CASE WHEN "BWART" IN ('601', '261', '201', '221', '641') AND "SHKZG" = 'H' THEN 1 END) AS issue_txn_cnt,
            COUNT(CASE WHEN "BWART" IN ('101', '105') AND "SHKZG" = 'S' THEN 1 END) AS receipt_txn_cnt,
            COUNT(*) AS total_movement_cnt
        FROM "HANA_P2"."MSEG_SJPR8"
        WHERE "BUDAT_MKPF" IS NOT NULL
        GROUP BY "MATNR", "WERKS"
    ),
    p3_marc AS (
        SELECT DISTINCT ON ("MATNR", "WERKS")
            "MATNR", "WERKS", "MAABC", "DISMM", "DISPO", "PLIFZ", "MINBE", "EISBE", "DISLS", "BESKZ"
        FROM "HANA_P3"."MARC"
    ),
    p3_t001w AS (
        SELECT DISTINCT ON ("WERKS")
            "WERKS", "NAME1"
        FROM "HANA_P3"."T001W"
    ),
    p3_mseg AS (
        SELECT 
            "MATNR",
            "WERKS",
            MAX(CASE WHEN "BWART" IN ('601', '261', '201', '221', '641') AND "SHKZG" = 'H' 
                     THEN CASE WHEN "BUDAT_MKPF" ~ '^[0-9]{{2}}\\.[0-9]{{2}}\\.[0-9]{{4}}$' THEN TO_DATE("BUDAT_MKPF", 'DD.MM.YYYY') END END) AS last_issue_date,
            MAX(CASE WHEN "BWART" IN ('101', '105') AND "SHKZG" = 'S' 
                     THEN CASE WHEN "BUDAT_MKPF" ~ '^[0-9]{{2}}\\.[0-9]{{2}}\\.[0-9]{{4}}$' THEN TO_DATE("BUDAT_MKPF", 'DD.MM.YYYY') END END) AS last_receipt_date,
            MAX(CASE WHEN "BUDAT_MKPF" ~ '^[0-9]{{2}}\\.[0-9]{{2}}\\.[0-9]{{4}}$' THEN TO_DATE("BUDAT_MKPF", 'DD.MM.YYYY') END) AS last_movement_date,
            SUM(CASE WHEN "BWART" IN ('601', '261', '201', '221', '641') AND "SHKZG" = 'H' 
                     THEN CAST(NULLIF(REPLACE("MENGE", ',', ''), '') AS NUMERIC) ELSE 0 END) AS total_issue_qty,
            SUM(CASE WHEN "BWART" IN ('601', '261', '201', '221', '641') AND "SHKZG" = 'H' 
                     THEN CAST(NULLIF(REPLACE("DMBTR", ',', ''), '') AS NUMERIC) ELSE 0 END) AS total_issue_val,
            SUM(CASE WHEN "BWART" IN ('101', '105') AND "SHKZG" = 'S' 
                     THEN CAST(NULLIF(REPLACE("MENGE", ',', ''), '') AS NUMERIC) ELSE 0 END) AS total_receipt_qty,
            SUM(CASE WHEN "BWART" IN ('101', '105') AND "SHKZG" = 'S' 
                     THEN CAST(NULLIF(REPLACE("DMBTR", ',', ''), '') AS NUMERIC) ELSE 0 END) AS total_receipt_val,
            COUNT(CASE WHEN "BWART" IN ('601', '261', '201', '221', '641') AND "SHKZG" = 'H' THEN 1 END) AS issue_txn_cnt,
            COUNT(CASE WHEN "BWART" IN ('101', '105') AND "SHKZG" = 'S' THEN 1 END) AS receipt_txn_cnt,
            COUNT(*) AS total_movement_cnt
        FROM "HANA_P3"."MSEG"
        WHERE "BUDAT_MKPF" IS NOT NULL
        GROUP BY "MATNR", "WERKS"
    )
    -- HANA_P2 dataset
    SELECT 
        'HANA_P2' AS source_schema,
        b."MATNR" AS matnr,
        b."BWKEY" AS plant,
        CAST(b."LBKUM" AS NUMERIC) AS current_stock_qty,
        CAST(b."SALK3" AS NUMERIC) AS current_stock_val,
        CAST(b."VERPR" AS NUMERIC) AS moving_avg_price,
        CAST(b."STPRS" AS NUMERIC) AS standard_price,
        b."VPRSV" AS price_control,
        b."BKLAS" AS valuation_class,
        CAST(b."VMKUM" AS NUMERIC) AS prev_period_stock_qty,
        CAST(b."VMSAL" AS NUMERIC) AS prev_period_stock_val,
        CAST(b."VJKUM" AS NUMERIC) AS prev_year_stock_qty,
        CAST(b."VJSAL" AS NUMERIC) AS prev_year_stock_val,
        m."MTART" AS material_type,
        m."MATKL" AS material_group,
        m."MEINS" AS base_uom,
        CAST(m."NTGEW" AS NUMERIC) AS net_weight,
        CAST(m."BRGEW" AS NUMERIC) AS gross_weight,
        CAST(m."ERSDA" AS TEXT) AS material_created_on,
        CAST(m."LAEDA" AS TEXT) AS last_change_date,
        t."MAKTX" AS material_name,
        c."MAABC" AS abc_indicator,
        c."DISMM" as mrp_type,
        c."DISPO" AS mrp_controller,
        CAST(c."PLIFZ" AS NUMERIC) AS planned_deliv_days,
        CAST(c."MINBE" AS NUMERIC) AS reorder_point,
        CAST(c."EISBE" AS NUMERIC) AS safety_stock,
        c."DISLS" AS lot_size_key,
        c."BESKZ" AS procurement_type,
        w."NAME1" AS plant_name,
        CAST(s.last_issue_date AS TEXT) AS last_issue_date,
        CAST(s.last_receipt_date AS TEXT) AS last_receipt_date,
        CAST(s.last_movement_date AS TEXT) AS last_movement_date,
        COALESCE(s.total_issue_qty, 0) AS total_issue_qty,
        COALESCE(s.total_issue_val, 0) AS total_issue_val,
        COALESCE(s.total_receipt_qty, 0) AS total_receipt_qty,
        COALESCE(s.total_receipt_val, 0) AS total_receipt_val,
        COALESCE(s.issue_txn_cnt, 0) AS issue_txn_cnt,
        COALESCE(s.receipt_txn_cnt, 0) AS receipt_txn_cnt,
        COALESCE(s.total_movement_cnt, 0) AS total_movement_cnt
    FROM "HANA_P2"."MBEW" b
    JOIN "HANA_P2"."MARA" m ON b."MATNR" = m."MATNR"
    LEFT JOIN "HANA_P2"."MAKT" t ON b."MATNR" = t."MATNR" AND t."SPRAS" = 'E'
    LEFT JOIN "HANA_P2"."MARC" c ON b."MATNR" = c."MATNR" AND b."BWKEY" = c."WERKS"
    LEFT JOIN "HANA_P2"."T001W" w ON b."BWKEY" = w."WERKS"
    LEFT JOIN p2_mseg s ON b."MATNR" = s."MATNR" AND b."BWKEY" = s."WERKS"
    WHERE b."BWTAR" IS NULL AND CAST(b."LBKUM" AS NUMERIC) > 0

    UNION ALL

    -- HANA_P3 dataset
    SELECT 
        'HANA_P3' AS source_schema,
        b."MATNR" AS matnr,
        b."BWKEY" AS plant,
        CAST(NULLIF(REPLACE(b."LBKUM", ',', ''), '') AS NUMERIC) AS current_stock_qty,
        CAST(NULLIF(REPLACE(b."SALK3", ',', ''), '') AS NUMERIC) AS current_stock_val,
        CAST(NULLIF(REPLACE(b."VERPR", ',', ''), '') AS NUMERIC) AS moving_avg_price,
        CAST(NULLIF(REPLACE(b."STPRS", ',', ''), '') AS NUMERIC) AS standard_price,
        b."VPRSV" AS price_control,
        b."BKLAS" AS valuation_class,
        CAST(NULLIF(REPLACE(b."VMKUM", ',', ''), '') AS NUMERIC) AS prev_period_stock_qty,
        CAST(NULLIF(REPLACE(b."VMSAL", ',', ''), '') AS NUMERIC) AS prev_period_stock_val,
        CAST(NULLIF(REPLACE(b."VJKUM", ',', ''), '') AS NUMERIC) AS prev_year_stock_qty,
        CAST(NULLIF(REPLACE(b."VJSAL", ',', ''), '') AS NUMERIC) AS prev_year_stock_val,
        m."MTART" AS material_type,
        m."MATKL" AS material_group,
        m."MEINS" AS base_uom,
        CAST(NULLIF(REPLACE(m."NTGEW", ',', ''), '') AS NUMERIC) AS net_weight,
        CAST(NULLIF(REPLACE(m."BRGEW", ',', ''), '') AS NUMERIC) AS gross_weight,
        CAST(m."ERSDA" AS TEXT) AS material_created_on,
        CAST(m."LAEDA" AS TEXT) AS last_change_date,
        t."MAKTX" AS material_name,
        c."MAABC" AS abc_indicator,
        c."DISMM" as mrp_type,
        c."DISPO" AS mrp_controller,
        CAST(NULLIF(REPLACE(c."PLIFZ", ',', ''), '') AS NUMERIC) AS planned_deliv_days,
        CAST(NULLIF(REPLACE(c."MINBE", ',', ''), '') AS NUMERIC) AS reorder_point,
        CAST(NULLIF(REPLACE(c."EISBE", ',', ''), '') AS NUMERIC) AS safety_stock,
        c."DISLS" AS lot_size_key,
        c."BESKZ" AS procurement_type,
        w."NAME1" AS plant_name,
        CAST(s.last_issue_date AS TEXT) AS last_issue_date,
        CAST(s.last_receipt_date AS TEXT) AS last_receipt_date,
        CAST(s.last_movement_date AS TEXT) AS last_movement_date,
        COALESCE(s.total_issue_qty, 0) AS total_issue_qty,
        COALESCE(s.total_issue_val, 0) AS total_issue_val,
        COALESCE(s.total_receipt_qty, 0) AS total_receipt_qty,
        COALESCE(s.total_receipt_val, 0) AS total_receipt_val,
        COALESCE(s.issue_txn_cnt, 0) AS issue_txn_cnt,
        COALESCE(s.receipt_txn_cnt, 0) AS receipt_txn_cnt,
        COALESCE(s.total_movement_cnt, 0) AS total_movement_cnt
    FROM "HANA_P3"."MBEW" b
    JOIN "HANA_P3"."MARA" m ON b."MATNR" = m."MATNR"
    LEFT JOIN "HANA_P3"."MAKT" t ON b."MATNR" = t."MATNR" AND t."SPRAS" = 'E'
    LEFT JOIN p3_marc c ON b."MATNR" = c."MATNR" AND b."BWKEY" = c."WERKS"
    LEFT JOIN p3_t001w w ON b."BWKEY" = w."WERKS"
    LEFT JOIN p3_mseg s ON b."MATNR" = s."MATNR" AND b."BWKEY" = s."WERKS"
    WHERE b."BWTAR" IS NULL 
      AND b."LBKUM" IS NOT NULL AND b."LBKUM" != '' AND b."LBKUM" != 'None'
      AND CAST(NULLIF(REPLACE(b."LBKUM", ',', ''), '') AS NUMERIC) > 0
    {limit_clause};
    """

    print("[DB] Executing staged inventory query across HANA_P2 and HANA_P3 without row limits...")
    cur.execute(query)
    cols = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    cur.close()
    conn.close()

    df = pd.DataFrame(rows, columns=cols)
    print(f"[DB] Successfully extracted {len(df):,} records with {len(cols)} columns in {time.time() - t0:.2f}s.")

    # Save to local cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"[CACHE] Saved full raw dataset to {cache_path}.")

    return df
