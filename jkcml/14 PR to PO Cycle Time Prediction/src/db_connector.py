"""
Database Connector Module - Use Case 14: PR to PO Cycle Time Prediction
Category: Regression
Objective: Estimate procurement processing time.

Handles connection and data extraction from PostgreSQL for SAP HANA Procurement tables:
- EBAN: Purchase Requisition Header & Item
- MARA: General Material Data
- MAKT: Material Descriptions
- LFA1: Vendor Master General Data
- T001W: Plant Master
- T001: Company Code Master

Supports multi-schema extraction across HANA_P2 and HANA_P3 with no limits.
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


def build_schema_query(schema: str, limit: int = 0) -> str:
    """
    Constructs high-performance 1:1 joined query for SAP PRs from the specified schema.
    Uses deduplicated dimension subqueries to prevent Cartesian product row duplication.
    """
    limit_clause = f"LIMIT {limit}" if limit and limit > 0 else ""

    query = f"""
    WITH makt_dedup AS (
        SELECT DISTINCT ON ("MATNR") "MATNR", "MAKTX" 
        FROM "{schema}"."MAKT" 
        WHERE "SPRAS" = 'EN'
    ),
    t001w_dedup AS (
        SELECT DISTINCT ON ("WERKS") "WERKS", "NAME1", "ORT01" 
        FROM "{schema}"."T001W"
    ),
    lfa1_dedup AS (
        SELECT DISTINCT ON ("LIFNR") "LIFNR", "NAME1", "ORT01", "REGIO", "LAND1" 
        FROM "{schema}"."LFA1"
    ),
    mara_dedup AS (
        SELECT DISTINCT ON ("MATNR") "MATNR", "MTART", "MATKL" 
        FROM "{schema}"."MARA"
    )
    SELECT 
        cp."BANFN" AS pr_number,
        cp."BNFPO" AS pr_item,
        cp."BSART" AS doc_type,
        cp."STATU" AS processing_status,
        cp."LOEKZ" AS deletion_indicator,
        cp."FRGKZ" AS release_indicator,
        cp."FRGZU" AS release_state,
        cp."FRGST" AS release_strategy,
        cp."FRGGR" AS release_group,
        cp."EKGRP" AS purchasing_group,
        cp."ERDAT" AS creation_date,
        cp."BADAT" AS requisition_date,
        cp."LFDAT" AS delivery_date,
        cp."FRGDT" AS release_date,
        cp."BEDAT" AS po_date,
        cp."EBELN" AS po_number,
        cp."EBELP" AS po_item,
        cp."AFNAM" AS requisitioner,
        cp."KNTTP" AS acct_assignment_cat,
        cp."PSTYP" AS item_category,
        cp."PLIFZ" AS planned_delivery_time,
        cp."MENGE" AS quantity,
        cp."MEINS" AS unit_of_measure,
        cp."PREIS" AS unit_price,
        cp."PEINH" AS price_unit,
        cp."WAERS" AS currency,
        cp."WERKS" AS plant,
        w."NAME1" AS plant_name,
        w."ORT01" AS plant_city,
        '1000' AS company_code,
        'J.K. CEMENT LIMITED' AS company_name,
        'Kanpur' AS company_city,
        'IN' AS company_country,
        'INR' AS company_currency,
        cp."MATNR" AS material_id,
        mkt."MAKTX" AS material_desc,
        m."MTART" AS material_type,
        COALESCE(NULLIF(cp."MATKL", ''), m."MATKL") AS material_group,
        cp."LIFNR" AS vendor_id,
        v."NAME1" AS vendor_name,
        v."ORT01" AS vendor_city,
        v."REGIO" AS vendor_region,
        v."LAND1" AS vendor_country,
        '{schema}' AS schema_source
    FROM "{schema}"."EBAN" cp
    LEFT JOIN t001w_dedup w ON cp."WERKS" = w."WERKS"
    LEFT JOIN mara_dedup m ON cp."MATNR" = m."MATNR"
    LEFT JOIN makt_dedup mkt ON cp."MATNR" = mkt."MATNR"
    LEFT JOIN lfa1_dedup v ON cp."LIFNR" = v."LIFNR"
    {limit_clause}
    """
    return query


def extract_pr_cycle_data(limit: int = 0, config=None) -> pd.DataFrame:
    """
    Extracts Purchase Requisitions and procurement cycle data from all configured SAP schemas:
    HANA_P2 (165,008 records) and HANA_P3 (363,993 records).
    If limit <= 0, extracts ALL available data without limits.
    """
    if config is None:
        config = load_config()

    schemas = config["database"].get("schemas", ["HANA_P2", "HANA_P3"])
    if isinstance(schemas, str):
        schemas = [schemas]

    conn = get_db_connection(config)
    dfs = []

    for s in schemas:
        print(f"[INFO] Extracting PR procurement data from schema: {s} (limit={'UNLIMITED' if limit <= 0 else limit})...")
        query = build_schema_query(schema=s, limit=limit)
        df_s = pd.read_sql(query, conn)
        print(f"[INFO] Extracted {len(df_s):,} records from {s}.")
        dfs.append(df_s)

    conn.close()

    df_combined = pd.concat(dfs, ignore_index=True)
    if limit and limit > 0 and len(df_combined) > limit:
        df_combined = df_combined.head(limit)

    print(f"[SUCCESS] Total PR records extracted across {schemas}: {len(df_combined):,} records.")
    return df_combined
