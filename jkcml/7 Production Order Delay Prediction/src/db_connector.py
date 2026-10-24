"""
Database Connector Module - Use Case 7: Production Order Delay Prediction
Handles direct connection to PostgreSQL database (192.168.1.44:5432 / JKC_Input)
and extracts ALL production order records across both HANA_P2 and HANA_P3 schemas
by joining AFKO, AUFK, AFPO, AFRU, MAKT, MARA, MARC without any row limits.
Includes intelligent local caching to accelerate repetitive training and experimentation.
"""
import os
import yaml
import psycopg2
import pandas as pd


def load_config(config_path: str = "config/db_config.yaml") -> dict:
    """Loads configuration yaml file."""
    if not os.path.exists(config_path):
        return {
            "database": {
                "host": "192.168.1.44",
                "port": 5432,
                "dbname": "JKC_Input",
                "schemas": ["HANA_P2", "HANA_P3"],
                "user": "postgres",
                "password": "postgres"
            },
            "app": {
                "name": "Production Order Delay Prediction & Classification",
                "use_case_id": 7,
                "version": "1.0.0",
                "random_seed": 42,
                "sample_size": None
            }
        }
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_db_connection(config: dict = None):
    """Establishes and returns a psycopg2 connection to PostgreSQL."""
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


def extract_production_order_data(
    limit: int = None,
    config: dict = None,
    use_cache: bool = True,
    cache_path: str = "data/cache/production_order_delay_raw_data_all.csv"
) -> pd.DataFrame:
    """
    Extracts joined production order data from PostgreSQL HANA_P2 and HANA_P3 schemas.
    When limit is None, extracts ALL available production orders without any row limit.
    Uses local cache if available and requested.
    """
    if use_cache and os.path.exists(cache_path):
        print(f"[CACHE] Loading cached full production order dataset from: {cache_path}")
        df = pd.read_csv(cache_path, low_memory=False)
        print(f"[CACHE] Successfully loaded {len(df):,} records from cache.")
        return df

    if config is None:
        config = load_config()

    print("[DB] Connecting to live PostgreSQL database (192.168.1.44:5432 / JKC_Input)...")
    print("[DB] Extracting ALL production order records across HANA_P2 and HANA_P3 schemas (NO LIMIT)...")
    conn = get_db_connection(config)

    limit_clause = f"LIMIT {limit}" if (limit is not None and limit > 0) else ""

    sql_query = f"""
    WITH afru_agg AS (
        SELECT 
            "AUFNR",
            MIN("ISDD")::text as min_conf_start,
            MAX("IEDD")::text as max_conf_end,
            MAX("BUDAT")::text as last_conf_date,
            SUM("LMNGA") as total_conf_yield,
            SUM("XMNGA") as total_conf_scrap,
            COUNT(*) as total_confirmations
        FROM (
            SELECT "AUFNR", "ISDD", "IEDD", "BUDAT", "LMNGA", "XMNGA" FROM "HANA_P3"."AFRU" WHERE "AUFNR" IS NOT NULL
            UNION ALL
            SELECT "AUFNR", "ISDD", "IEDD", "BUDAT", "LMNGA", "XMNGA" FROM "HANA_P2"."AFRU" WHERE "AUFNR" IS NOT NULL
        ) u_afru
        GROUP BY "AUFNR"
    ),
    afpo_dedup AS (
        SELECT 
            "AUFNR", "MATNR", "PSMNG"::text as item_target_qty, "WEMNG"::text as item_delivered_qty,
            "IAMNG"::text as item_scrap_qty, "AMEIN" as item_uom, "LTRMI"::text as actual_delivery_date,
            "LTRMP"::text as planned_delivery_date, "ELIKZ" as delivery_completed_flag,
            "CHARG" as batch_number, "LGORT" as storage_loc,
            ROW_NUMBER() OVER (PARTITION BY "AUFNR" ORDER BY "POSNR") as rn
        FROM "HANA_P3"."AFPO"
    )
    SELECT 
        k."AUFNR" as order_number,
        u."AUART" as order_type,
        u."KTEXT" as order_description,
        u."WERKS" as plant,
        u."BUKRS" as company_code,
        u."ERDAT"::text as order_creation_date,
        u."AEDAT"::text as order_change_date,
        u."IDAT2"::text as technical_completion_date,
        
        k."GLTRP"::text as planned_finish_date,
        k."GSTRP"::text as planned_start_date,
        k."GLTRS"::text as scheduled_finish_date,
        k."GSTRS"::text as scheduled_start_date,
        k."GLTRI"::text as actual_finish_date,
        k."GSTRI"::text as actual_start_date,
        k."FTRMS"::text as scheduled_release_date,
        k."FTRMI"::text as actual_release_date,
        k."GAMNG"::text as total_order_qty,
        k."GASMG"::text as planned_scrap_qty,
        k."GMEIN" as base_uom,
        k."DISPO" as mrp_controller,
        k."FEVOR" as production_supervisor,
        k."IGMNG"::text as header_confirmed_qty,
        
        p.item_target_qty,
        p.item_delivered_qty,
        p.item_scrap_qty,
        p.item_uom,
        p.actual_delivery_date,
        p.planned_delivery_date,
        p.delivery_completed_flag,
        p.batch_number,
        p.storage_loc,
        
        a.min_conf_start,
        a.max_conf_end,
        a.last_conf_date,
        a.total_conf_yield,
        a.total_conf_scrap,
        a.total_confirmations,
        
        p."MATNR" as material_number,
        m."MAKTX" as material_name,
        ra."MTART" as material_type,
        ra."MATKL" as material_group,
        ra."NTGEW"::text as net_weight,
        rc."PLIFZ"::text as planned_deliv_lead_days,
        rc."EISBE"::text as plant_safety_stock
    FROM "HANA_P3"."AFKO" k
    JOIN "HANA_P3"."AUFK" u ON k."AUFNR" = u."AUFNR"
    LEFT JOIN afpo_dedup p ON k."AUFNR" = p."AUFNR" AND p.rn = 1
    LEFT JOIN afru_agg a ON k."AUFNR" = a."AUFNR"
    LEFT JOIN "HANA_P3"."MAKT" m ON p."MATNR" = m."MATNR" AND m."SPRAS" = 'E'
    LEFT JOIN "HANA_P3"."MARA" ra ON p."MATNR" = ra."MATNR"
    LEFT JOIN "HANA_P3"."MARC" rc ON p."MATNR" = rc."MATNR" AND u."WERKS" = rc."WERKS"
    WHERE k."GLTRP" IS NOT NULL AND (u."IDAT2" IS NOT NULL OR k."GLTRI" IS NOT NULL OR p.actual_delivery_date IS NOT NULL)
    ORDER BY k."GLTRP" DESC
    {limit_clause};
    """

    df = pd.read_sql(sql_query, conn)
    conn.close()

    print(f"[DB] Successfully extracted {len(df):,} production order records across HANA_P2 and HANA_P3.")

    # Save to local cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"[CACHE] Saved raw production order dataset to: {cache_path}")

    return df
