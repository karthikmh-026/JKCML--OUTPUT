"""
Database Connector Module - Use Case 15: Order Value Prediction
Connects to PostgreSQL (192.168.1.44:5432 / JKC_Input / HANA_P2 & HANA_P3)
and extracts live SAP Sales Orders (VBAK), Line Items (VBAP), and Customer Master (KNA1, KNVV)
without artificial row limits.
"""
import os
import yaml
import pandas as pd
from sqlalchemy import create_engine


def load_config(config_path: str = "config/db_config.yaml") -> dict:
    """Loads database and model configuration from YAML."""
    if not os.path.exists(config_path):
        return {
            "database": {
                "host": "192.168.1.44",
                "port": 5432,
                "dbname": "JKC_Input",
                "schema": "HANA_P3",
                "fallback_schema": "HANA_P2",
                "username": "postgres",
                "password": "postgres"
            },
            "query": {
                "default_limit": None,
                "excel_export_limit": 100000
            }
        }
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_db_engine(config: dict = None):
    """Creates an SQLAlchemy engine for the PostgreSQL database."""
    if config is None:
        config = load_config()
    db = config["database"]
    uri = f"postgresql+psycopg2://{db['username']}:{db['password']}@{db['host']}:{db['port']}/{db['dbname']}"
    return create_engine(uri)


def extract_order_value_data(limit: int = None, config: dict = None) -> pd.DataFrame:
    """
    Extracts live sales orders joined with customer master and aggregated line item data
    from SAP tables across schemas HANA_P3 and HANA_P2.
    When limit is None or <= 0, extracts the complete multi-million dataset with NO limit.
    """
    if config is None:
        config = load_config()

    schema = config["database"].get("schema", "HANA_P3")
    engine = get_db_engine(config)

    print(f"[INFO] Connecting to PostgreSQL at {config['database']['host']}:{config['database']['port']} ({config['database']['dbname']})...")
    print(f"[INFO] Querying schema '{schema}' for Sales Orders (VBAK), Item Metrics (VBAP), and Customers (KNA1)...")
    if limit is None or limit <= 0:
        print("[INFO] Row limit is REMOVED: Extracting all available sales order data from database server.")
    else:
        print(f"[INFO] Row extraction limit: {limit:,} records.")

    limit_clause = f"LIMIT {limit}" if (limit is not None and limit > 0) else ""

    # Check schema column typing for NETWR and numeric conversions
    is_p3 = (schema.upper() == "HANA_P3")
    if is_p3:
        netwr_expr = "NULLIF(REGEXP_REPLACE(vbak.\"NETWR\"::text, '[^0-9.]', '', 'g'), '')::double precision"
        kwmeng_expr = "COALESCE(NULLIF(REGEXP_REPLACE(\"KWMENG\"::text, '[^0-9.]', '', 'g'), '')::double precision, 0)"
        netpr_expr = "COALESCE(NULLIF(REGEXP_REPLACE(\"NETPR\"::text, '[^0-9.]', '', 'g'), '')::double precision, 0)"
        vbap_netwr_expr = "COALESCE(NULLIF(REGEXP_REPLACE(\"NETWR\"::text, '[^0-9.]', '', 'g'), '')::double precision, 0)"
        brgew_expr = "COALESCE(NULLIF(REGEXP_REPLACE(\"BRGEW\"::text, '[^0-9.]', '', 'g'), '')::double precision, 0)"
        valid_netwr_filter = """
            vbak."NETWR" IS NOT NULL 
            AND vbak."NETWR" != '' 
            AND vbak."NETWR" != '0'
            AND vbak."NETWR" ~ '^[0-9]+(\\.[0-9]+)?$'
        """
    else:
        netwr_expr = "vbak.\"NETWR\"::double precision"
        kwmeng_expr = "COALESCE(\"KWMENG\", 0)"
        netpr_expr = "COALESCE(\"NETPR\", 0)"
        vbap_netwr_expr = "COALESCE(\"NETWR\", 0)"
        brgew_expr = "COALESCE(\"BRGEW\", 0)"
        valid_netwr_filter = 'vbak."NETWR" IS NOT NULL AND vbak."NETWR" > 0'

    query = f"""
    WITH orders AS (
        SELECT 
            vbak."VBELN"::text AS sales_order_id,
            vbak."ERDAT"::text AS order_date,
            vbak."AUART"::text AS order_type,
            {netwr_expr} AS order_net_value,
            vbak."WAERK"::text AS currency,
            vbak."VKORG"::text AS sales_org,
            vbak."VTWEG"::text AS dist_channel,
            vbak."SPART"::text AS division,
            vbak."VKBUR"::text AS sales_office,
            vbak."VKGRP"::text AS sales_group,
            vbak."KUNNR"::text AS customer_id,
            vbak."LIFSK"::text AS delivery_block,
            vbak."FAKSK"::text AS billing_block,
            vbak."CMGST"::text AS credit_status,
            vbak."GBSTK"::text AS overall_status,
            vbak."ABSTK"::text AS rejection_status,
            kna1."NAME1"::text AS customer_name,
            kna1."ORT01"::text AS customer_city,
            kna1."REGIO"::text AS customer_region,
            kna1."LAND1"::text AS customer_country,
            kna1."KTOKD"::text AS account_group
        FROM "{schema}"."VBAK" vbak
        LEFT JOIN "{schema}"."KNA1" kna1 ON vbak."KUNNR" = kna1."KUNNR"
        WHERE {valid_netwr_filter}
        ORDER BY vbak."ERDAT" DESC, vbak."VBELN" DESC
        {limit_clause}
    ),
    vbap_agg AS (
        SELECT 
            "VBELN"::text AS sales_order_id,
            COUNT("POSNR") AS total_items,
            SUM({kwmeng_expr}) AS total_quantity,
            COUNT(DISTINCT "MATNR") AS distinct_materials,
            AVG({netpr_expr}) AS avg_unit_price,
            MAX({vbap_netwr_expr}) AS max_item_value,
            SUM({brgew_expr}) AS total_gross_weight,
            MAX("WERKS"::text) AS plant
        FROM "{schema}"."VBAP"
        WHERE "VBELN" IN (SELECT sales_order_id FROM orders)
        GROUP BY "VBELN"
    )
    SELECT 
        o.*,
        COALESCE(v.total_items, 1) AS total_items,
        COALESCE(v.total_quantity, 1.0) AS total_quantity,
        COALESCE(v.distinct_materials, 1) AS distinct_materials,
        COALESCE(v.avg_unit_price, o.order_net_value) AS avg_unit_price,
        COALESCE(v.max_item_value, o.order_net_value) AS max_item_value,
        COALESCE(v.total_gross_weight, 0.0) AS total_gross_weight,
        COALESCE(v.plant, '1000') AS plant
    FROM orders o
    LEFT JOIN vbap_agg v ON o.sales_order_id = v.sales_order_id;
    """

    df = pd.read_sql(query, engine)
    print(f"[INFO] Successfully fetched {len(df):,} SAP sales order records from {schema}.")
    return df
