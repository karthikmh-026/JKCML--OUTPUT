"""
Database Connector Module - Use Case 8: Quality Inspection Failure Prediction
Handles direct connection to PostgreSQL database (192.168.1.44:5432 / JKC_Input)
and extracts joined inspection lot records across HANA_P3 and HANA_P2:
Tables: QALS, QAVE, MAKT, MARA, MARC, MBEW, LFA1 without any row limits.
Includes local caching to accelerate repeatable training and experimentation.
"""
import os
import yaml
import psycopg2
import pandas as pd


def load_config(config_path: str = "config/db_config.yaml") -> dict:
    """Loads configuration yaml file or returns default settings."""
    if not os.path.exists(config_path):
        return {
            "database": {
                "host": "192.168.1.44",
                "port": 5432,
                "dbname": "JKC_Input",
                "schemas": ["HANA_P3", "HANA_P2"],
                "user": "postgres",
                "password": "postgres"
            },
            "app": {
                "name": "Quality Inspection Failure Prediction & Classification",
                "use_case_id": 8,
                "version": "2.0.0",
                "random_seed": 42,
                "sample_size": None,
                "contamination": 0.05
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


def extract_quality_inspection_data(
    limit: int = None,
    config: dict = None,
    use_cache: bool = True,
    cache_path: str = "data/cache/qm_inspection_lot_raw_data.pkl",
    csv_fallback_path: str = "data/cache/qm_inspection_lot_raw_data.csv"
) -> pd.DataFrame:
    """
    Extracts joined quality inspection lot records from PostgreSQL HANA_P3 & HANA_P2 schemas.
    Uses local cache if available and requested; otherwise queries live server and caches.
    When limit is None, extracts 100% of available data with no limit.
    """
    if use_cache:
        if os.path.exists(cache_path):
            print(f"[CACHE] Loading cached quality inspection dataset from: {cache_path}")
            df = pd.read_pickle(cache_path)
            if limit is not None and limit > 0 and len(df) > limit:
                df = df.head(limit)
            print(f"[CACHE] Successfully loaded {len(df):,} records from cache.")
            return df
        elif os.path.exists(csv_fallback_path):
            print(f"[CACHE] Loading cached quality inspection dataset from fallback: {csv_fallback_path}")
            df = pd.read_csv(csv_fallback_path, low_memory=False)
            if limit is not None and limit > 0 and len(df) > limit:
                df = df.head(limit)
            print(f"[CACHE] Successfully loaded {len(df):,} records from CSV cache.")
            return df

    if config is None:
        config = load_config()

    print("[DB] Connecting to live PostgreSQL database (192.168.1.44:5432 / JKC_Input / HANA_P3 & HANA_P2)...")
    conn = get_db_connection(config)

    limit_clause = f"LIMIT {limit}" if limit is not None and limit > 0 else ""

    sql_query = f"""
    WITH qals_combined AS (
        SELECT 
            "PRUEFLOS"::text as inspection_lot,
            "WERK"::text as plant,
            "ART"::text as inspection_type,
            "HERKUNFT"::text as inspection_origin,
            "MATNR"::text as material_number,
            "LIFNR"::text as vendor_id,
            "CHARG"::text as batch_number,
            "LOSMENGE"::text as lot_quantity,
            "MENGENEINH"::text as quantity_uom,
            "LMENGEZUB"::text as sample_quantity,
            "LMENGEPR"::text as destroyed_sample_qty,
            "LMENGE01"::text as unrestricted_qty,
            "LMENGE02"::text as scrap_qty,
            "LMENGE03"::text as sample_stock_qty,
            "LMENGE04"::text as blocked_qty,
            "ENSTEHDAT"::text as creation_date,
            "PASTRTERM"::text as scheduled_start_date,
            "PAENDTERM"::text as scheduled_end_date,
            "STAT01"::text as status_created,
            "STAT02"::text as status_sample_drawn,
            "STAT26"::text as status_results_confirmed,
            "STAT35"::text as status_ud_made,
            "EBELN"::text as po_number,
            "EBELP"::text as po_item,
            "AUFNR"::text as order_number,
            'HANA_P3' as source_schema
        FROM "HANA_P3"."QALS"
        UNION ALL
        SELECT 
            p2."PRUEFLOS"::text as inspection_lot,
            p2."WERK"::text as plant,
            p2."ART"::text as inspection_type,
            p2."HERKUNFT"::text as inspection_origin,
            p2."MATNR"::text as material_number,
            p2."LIFNR"::text as vendor_id,
            p2."CHARG"::text as batch_number,
            p2."LOSMENGE"::text as lot_quantity,
            p2."MENGENEINH"::text as quantity_uom,
            p2."LMENGEZUB"::text as sample_quantity,
            p2."LMENGEPR"::text as destroyed_sample_qty,
            p2."LMENGE01"::text as unrestricted_qty,
            p2."LMENGE02"::text as scrap_qty,
            p2."LMENGE03"::text as sample_stock_qty,
            p2."LMENGE04"::text as blocked_qty,
            p2."ENSTEHDAT"::text as creation_date,
            p2."PASTRTERM"::text as scheduled_start_date,
            p2."PAENDTERM"::text as scheduled_end_date,
            p2."STAT01"::text as status_created,
            p2."STAT02"::text as status_sample_drawn,
            p2."STAT26"::text as status_results_confirmed,
            p2."STAT35"::text as status_ud_made,
            p2."EBELN"::text as po_number,
            p2."EBELP"::text as po_item,
            p2."AUFNR"::text as order_number,
            'HANA_P2' as source_schema
        FROM "HANA_P2"."QALS" p2
        WHERE NOT EXISTS (
            SELECT 1 FROM "HANA_P3"."QALS" p3 WHERE p3."PRUEFLOS"::text = p2."PRUEFLOS"::text
        )
    ),
    makt_dedup AS (
        SELECT 
            "MATNR",
            "MAKTX",
            ROW_NUMBER() OVER (PARTITION BY "MATNR" ORDER BY "SPRAS" DESC) as rn
        FROM "HANA_P3"."MAKT"
        WHERE "SPRAS" IN ('EN', 'E')
    ),
    marc_dedup AS (
        SELECT 
            "MATNR",
            "WERKS",
            "PLIFZ",
            "EISBE",
            ROW_NUMBER() OVER (PARTITION BY "MATNR", "WERKS" ORDER BY "PLIFZ" DESC) as rn
        FROM "HANA_P3"."MARC"
    ),
    mbew_dedup AS (
        SELECT 
            "MATNR",
            "BWKEY",
            MAX("STPRS") as standard_price,
            MAX("VERPR") as moving_avg_price,
            MAX("PEINH") as price_unit
        FROM "HANA_P3"."MBEW"
        GROUP BY "MATNR", "BWKEY"
    )
    SELECT 
        q.inspection_lot,
        q.plant,
        q.inspection_type,
        q.inspection_origin,
        q.material_number,
        m."MAKTX" as material_name,
        ra."MTART" as material_type,
        ra."MATKL" as material_group,
        ra."MEINS" as base_uom,
        ra."NTGEW" as net_weight,
        
        q.vendor_id,
        l."NAME1" as vendor_name,
        l."LAND1" as vendor_country,
        q.batch_number,
        
        q.lot_quantity,
        q.quantity_uom,
        q.sample_quantity,
        q.destroyed_sample_qty,
        q.unrestricted_qty,
        q.scrap_qty,
        q.sample_stock_qty,
        q.blocked_qty,
        
        q.creation_date,
        q.scheduled_start_date,
        q.scheduled_end_date,
        
        q.status_created,
        q.status_sample_drawn,
        q.status_results_confirmed,
        q.status_ud_made,
        
        q.po_number,
        q.po_item,
        q.order_number,
        
        v."VBEWERTUNG" as ud_valuation,
        v."VCODE" as ud_code,
        v."VDATUM" as ud_date,
        
        rc."PLIFZ" as planned_delivery_days,
        rc."EISBE" as plant_safety_stock,
        
        mb.standard_price,
        mb.moving_avg_price,
        mb.price_unit,
        q.source_schema
        
    FROM qals_combined q
    LEFT JOIN "HANA_P3"."QAVE" v ON q.inspection_lot = v."PRUEFLOS"::text
    LEFT JOIN makt_dedup m ON q.material_number = m."MATNR" AND m.rn = 1
    LEFT JOIN "HANA_P3"."MARA" ra ON q.material_number = ra."MATNR"
    LEFT JOIN marc_dedup rc ON q.material_number = rc."MATNR" AND q.plant = rc."WERKS" AND rc.rn = 1
    LEFT JOIN mbew_dedup mb ON q.material_number = mb."MATNR" AND q.plant = mb."BWKEY"
    LEFT JOIN "HANA_P3"."LFA1" l ON q.vendor_id = l."LIFNR"
    ORDER BY q.creation_date DESC NULLS LAST
    {limit_clause};
    """

    desc = f"Limit: {limit:,}" if limit is not None and limit > 0 else "ALL DATA (No Limit)"
    print(f"[DB] Executing live extraction query across HANA_P3 & HANA_P2 ({desc})...")
    df = pd.read_sql(sql_query, conn)
    conn.close()

    print(f"[DB] Successfully extracted {len(df):,} unique inspection lot records.")

    # Save to local cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_pickle(cache_path)
    print(f"[CACHE] Saved raw quality inspection dataset to: {cache_path}")

    return df
