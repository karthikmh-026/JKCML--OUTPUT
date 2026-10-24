"""
Database Connector Module
Handles connection and data extraction from PostgreSQL for SAP HANA tables.
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


def extract_procurement_data(limit=None, config=None):
    """
    Extracts procurement transaction and master data from SAP tables:
    - EKPO: Purchasing Document Item
    - EKKO: Purchasing Document Header
    - T001: Company Code Master
    - T001W: Plants
    - LFA1: Vendor Master
    - MARA: General Material Data
    - MAKT: Material Descriptions
    - EINA: Purchasing Info Record (General)
    - EINE: Purchasing Info Record (Purchasing Org)
    """
    if config is None:
        config = load_config()
    schema = config["database"].get("schema", "HANA_P2")
    conn = get_db_connection(config)

    # Determine if limit should be applied
    limit_clause = ""
    if limit is not None:
        try:
            limit_int = int(limit)
            if limit_int > 0:
                limit_clause = f"LIMIT {limit_int}"
        except (ValueError, TypeError):
            limit_clause = ""

    # Extract valid material purchase orders directly from server tables
    # Uses window partitioning on EINA/EINE so every PO item maps cleanly to its primary active info record
    query = f"""
    SELECT 
        k."BUKRS" AS company_code,
        c."BUTXT" AS company_name,
        c."ORT01" AS company_city,
        c."LAND1" AS company_country,
        c."WAERS" AS company_currency,
        k."LIFNR" AS vendor_id,
        v."NAME1" AS vendor_name,
        v."ORT01" AS vendor_city,
        v."REGIO" AS vendor_region,
        v."LAND1" AS vendor_country,
        p."WERKS" AS plant,
        w."NAME1" AS plant_name,
        w."ORT01" AS plant_city,
        p."MATNR" AS material_id,
        COALESCE(m."MAKTX", p."TXZ01") AS material_desc,
        ra."MTART" AS material_type,
        p."MATKL" AS material_group,
        p."EBELN" AS po_number,
        p."EBELP" AS po_item,
        k."BSART" AS doc_type,
        k."AEDAT"::text AS po_date,
        p."MENGE"::double precision AS order_quantity,
        p."MEINS" AS unit_of_measure,
        p."NETPR"::double precision AS po_unit_price,
        COALESCE(NULLIF(p."PEINH"::text, '')::double precision, 1.0) AS po_price_unit,
        k."WAERS" AS currency,
        p."NETWR"::double precision AS po_net_value,
        COALESCE(NULLIF(p."INFNR", ''), a."INFNR") AS info_record_id,
        e."NETPR"::double precision AS info_rec_price,
        COALESCE(NULLIF(e."PEINH"::text, '')::double precision, 1.0) AS info_rec_price_unit
    FROM "{schema}"."EKPO" p
    JOIN "{schema}"."EKKO" k ON p."EBELN" = k."EBELN"
    LEFT JOIN "{schema}"."T001" c ON k."BUKRS" = c."BUKRS"
    LEFT JOIN "{schema}"."T001W" w ON p."WERKS" = w."WERKS"
    LEFT JOIN "{schema}"."LFA1" v ON k."LIFNR" = v."LIFNR"
    LEFT JOIN "{schema}"."MAKT" m ON p."MATNR" = m."MATNR" AND m."SPRAS" = 'EN'
    LEFT JOIN "{schema}"."MARA" ra ON p."MATNR" = ra."MATNR"
    LEFT JOIN (
        SELECT "MATNR", "LIFNR", "INFNR",
               ROW_NUMBER() OVER (PARTITION BY "MATNR", "LIFNR" ORDER BY "INFNR" DESC) as rn
        FROM "{schema}"."EINA"
    ) a ON p."MATNR" = a."MATNR" AND k."LIFNR" = a."LIFNR" AND a.rn = 1
    LEFT JOIN (
        SELECT "INFNR", "NETPR", "PEINH",
               ROW_NUMBER() OVER (PARTITION BY "INFNR" ORDER BY "NETPR" DESC) as rn
        FROM "{schema}"."EINE"
        WHERE "NETPR" IS NOT NULL AND "NETPR" > 0
    ) e ON COALESCE(NULLIF(p."INFNR", ''), a."INFNR") = e."INFNR" AND e.rn = 1
    WHERE p."MATNR" IS NOT NULL AND p."MATNR" != ''
      AND k."LIFNR" IS NOT NULL AND k."LIFNR" != ''
      AND p."NETPR" IS NOT NULL AND p."NETPR"::double precision > 0
      AND p."MENGE" IS NOT NULL AND p."MENGE"::double precision > 0
      AND p."NETWR" IS NOT NULL AND p."NETWR"::double precision > 0
      AND (p."LOEKZ" IS NULL OR p."LOEKZ" = '')
    ORDER BY k."AEDAT" DESC NULLS LAST
    {limit_clause};
    """
    
    limit_msg = f"limit={limit_int}" if limit_clause else "ALL records (NO limit)"
    print(f"[INFO] Extracting records from schema '{schema}' ({limit_msg})...")
    df = pd.read_sql(query, conn)
    conn.close()
    print(f"[SUCCESS] Extracted {len(df):,} records with {len(df.columns)} features directly from server.")
    return df
