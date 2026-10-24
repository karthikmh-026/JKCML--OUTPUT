"""
Database Connector Module - Use Case 24: Material Auto Classification (NLP)
Handles connection and data extraction from PostgreSQL for SAP HANA Material Master,
Material Descriptions (MAKT), Plant/MRP (MARC), Valuation (MBEW), Purchase Order Short Text (EKPO),
Units of Measure (MARM), Purchasing Info Records (EINA), and Reference tables (T134T, T001W, T001)
across all target schemas (HANA_P2, HANA_P3) with zero row limits.
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

    with open(config_path, "r", encoding="utf-8") as f:
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


def build_schema_query(schema: str, limit: int = None) -> str:
    """
    Builds the extraction query for a given schema (HANA_P2 or HANA_P3) with safe casting
    to support both numeric-typed and text-typed SAP columns without schema datatype errors.
    """
    limit_clause = f"LIMIT {limit}" if limit and limit > 0 else ""

    return f"""
    WITH mara_pool AS (
        SELECT
            a."MATNR", a."ERSDA", a."ERNAM", a."LAEDA", a."AENAM",
            a."VPSTA", a."PSTAT", a."LVORM", a."MTART", a."MBRSH",
            a."MATKL", a."BISMT", a."MEINS", a."BSTME", a."SPART",
            a."PRDHA", a."GROES", a."WRKST", a."NORMT", a."TRAGR",
            a."XCHPF", a."MSTAE",
            CASE WHEN CAST(a."BRGEW" AS VARCHAR) ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN CAST(a."BRGEW" AS VARCHAR)::double precision ELSE 0.0 END AS brgew_num,
            CASE WHEN CAST(a."NTGEW" AS VARCHAR) ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN CAST(a."NTGEW" AS VARCHAR)::double precision ELSE 0.0 END AS ntgew_num,
            a."GEWEI",
            CASE WHEN CAST(a."VOLUM" AS VARCHAR) ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN CAST(a."VOLUM" AS VARCHAR)::double precision ELSE 0.0 END AS volum_num,
            a."VOLEH"
        FROM "{schema}"."MARA" a
        WHERE a."MATNR" IS NOT NULL AND a."MATNR" != ''
        {limit_clause}
    ),
    marc_primary AS (
        SELECT DISTINCT ON (c."MATNR")
            c."MATNR", c."WERKS",
            c."PSTAT" AS marc_pstat,
            c."LVORM" AS marc_lvorm,
            c."MMSTA" AS plant_mat_status,
            c."MAABC" AS abc_indicator,
            c."EKGRP" AS purchasing_group,
            c."DISMM" AS mrp_type,
            c."DISPO" AS mrp_controller,
            CASE WHEN CAST(c."PLIFZ" AS VARCHAR) ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN CAST(c."PLIFZ" AS VARCHAR)::integer ELSE 0 END AS planned_deliv_days,
            CASE WHEN CAST(c."WEBAZ" AS VARCHAR) ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN CAST(c."WEBAZ" AS VARCHAR)::integer ELSE 0 END AS gr_processing_days,
            c."BESKZ" AS procurement_type,
            c."PRCTR" AS profit_center,
            c."LADGR" AS loading_group
        FROM "{schema}"."MARC" c
        JOIN mara_pool mp ON c."MATNR" = mp."MATNR"
        ORDER BY c."MATNR", c."WERKS" ASC
    ),
    ekpo_agg AS (
        SELECT
            p."MATNR",
            COUNT(*) AS po_line_count,
            SUM(CASE WHEN CAST(p."NETWR" AS VARCHAR) ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN CAST(p."NETWR" AS VARCHAR)::double precision ELSE 0.0 END) AS total_po_spend,
            AVG(CASE WHEN CAST(p."NETPR" AS VARCHAR) ~ '^-?[0-9]+(\\.[0-9]+)?$' AND CAST(p."NETPR" AS VARCHAR)::double precision > 0 THEN CAST(p."NETPR" AS VARCHAR)::double precision ELSE NULL END) AS avg_po_net_price,
            MAX(p."TXZ01") AS sample_po_short_text,
            MAX(p."MATKL") AS po_material_group,
            COUNT(DISTINCT p."MATKL") AS distinct_po_matkl_cnt
        FROM "{schema}"."EKPO" p
        JOIN mara_pool mp ON p."MATNR" = mp."MATNR"
        WHERE p."MATNR" IS NOT NULL AND p."MATNR" != ''
          AND (p."LOEKZ" IS NULL OR p."LOEKZ" = '')
        GROUP BY p."MATNR"
    ),
    marm_cnt AS (
        SELECT u."MATNR", COUNT(*) AS alt_uom_count
        FROM "{schema}"."MARM" u
        JOIN mara_pool mp ON u."MATNR" = mp."MATNR"
        GROUP BY u."MATNR"
    ),
    eina_cnt AS (
        SELECT e."MATNR", COUNT(*) AS info_record_count
        FROM "{schema}"."EINA" e
        JOIN mara_pool mp ON e."MATNR" = mp."MATNR"
        WHERE e."LOEKZ" IS NULL OR e."LOEKZ" = ''
        GROUP BY e."MATNR"
    ),
    makt_primary AS (
        SELECT DISTINCT ON (m."MATNR")
            m."MATNR", m."MAKTX", m."MAKTG"
        FROM "{schema}"."MAKT" m
        JOIN mara_pool mp ON m."MATNR" = mp."MATNR"
        WHERE m."SPRAS" = 'EN'
        ORDER BY m."MATNR"
    ),
    mbew_primary AS (
        SELECT DISTINCT ON (b."MATNR")
            b."MATNR",
            b."BKLAS", b."VPRSV",
            CASE WHEN CAST(b."VERPR" AS VARCHAR) ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN CAST(b."VERPR" AS VARCHAR)::double precision ELSE 0.0 END AS moving_avg_price,
            CASE WHEN CAST(b."STPRS" AS VARCHAR) ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN CAST(b."STPRS" AS VARCHAR)::double precision ELSE 0.0 END AS standard_price,
            CASE WHEN CAST(b."PEINH" AS VARCHAR) ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN CAST(b."PEINH" AS VARCHAR)::double precision ELSE 1.0 END AS price_unit,
            CASE WHEN CAST(b."LBKUM" AS VARCHAR) ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN CAST(b."LBKUM" AS VARCHAR)::double precision ELSE 0.0 END AS total_valuated_stock,
            CASE WHEN CAST(b."SALK3" AS VARCHAR) ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN CAST(b."SALK3" AS VARCHAR)::double precision ELSE 0.0 END AS total_stock_value,
            b."LVORM" AS mbew_deletion_flag
        FROM "{schema}"."MBEW" b
        JOIN mara_pool mp ON b."MATNR" = mp."MATNR"
        ORDER BY b."MATNR",
                 CASE WHEN (b."BWTAR" IS NULL OR b."BWTAR" = '') THEN 0 ELSE 1 END,
                 b."BWKEY" ASC
    )
    SELECT
        '{schema}' AS schema_name,
        '1000' AS company_code,
        COALESCE(comp."BUTXT", 'J.K. CEMENT LIMITED') AS company_name,
        COALESCE(comp."ORT01", 'Kanpur') AS company_city,
        COALESCE(comp."LAND1", 'IN') AS company_country,
        COALESCE(comp."WAERS", 'INR') AS company_currency,
        a."MATNR" AS material_id,
        COALESCE(m."MAKTX", '') AS material_desc,
        COALESCE(m."MAKTG", UPPER(COALESCE(m."MAKTX", ''))) AS material_desc_upper,
        COALESCE(po.sample_po_short_text, '') AS po_short_text,
        COALESCE(a."MTART", '') AS material_type,
        COALESCE(t."MTBEZ", a."MTART") AS material_type_desc,
        COALESCE(a."MBRSH", 'C') AS industry_sector,
        COALESCE(a."MATKL", '') AS material_group,
        COALESCE(po.po_material_group, '') AS po_material_group,
        COALESCE(po.distinct_po_matkl_cnt, 0) AS distinct_po_matkl_cnt,
        COALESCE(a."BISMT", '') AS old_material_no,
        COALESCE(a."MEINS", '') AS base_uom,
        COALESCE(a."BSTME", '') AS order_uom,
        COALESCE(a."SPART", '') AS division,
        COALESCE(a."PRDHA", '') AS prod_hierarchy,
        COALESCE(a."GROES", '') AS size_dimensions,
        COALESCE(a."WRKST", '') AS basic_material_text,
        COALESCE(a."NORMT", '') AS industry_std_desc,
        COALESCE(a."TRAGR", '') AS trans_group,
        COALESCE(a."XCHPF", '') AS batch_mgmt_flag,
        COALESCE(a."MSTAE", '') AS cross_plant_status,
        COALESCE(a."VPSTA", '') AS complete_maint_status,
        COALESCE(a."PSTAT", '') AS mara_maint_status,
        COALESCE(a."LVORM", '') AS mara_deletion_flag,
        COALESCE(a.brgew_num, 0.0) AS gross_weight,
        COALESCE(a.ntgew_num, 0.0) AS net_weight,
        COALESCE(a."GEWEI", '') AS weight_unit,
        COALESCE(a.volum_num, 0.0) AS volume,
        COALESCE(a."VOLEH", '') AS volume_unit,
        a."ERSDA" AS created_date,
        COALESCE(a."ERNAM", 'SAP_MDM_USER') AS created_by,
        a."LAEDA" AS last_changed_date,
        COALESCE(a."AENAM", '') AS changed_by,
        COALESCE(c."WERKS", '') AS plant,
        COALESCE(w."NAME1", 'JK Cement Plant') AS plant_name,
        COALESCE(w."ORT01", 'Kanpur') AS plant_city,
        COALESCE(w."REGIO", '08') AS plant_region,
        COALESCE(w."LAND1", 'IN') AS plant_country,
        COALESCE(c.marc_pstat, '') AS marc_maint_status,
        COALESCE(c.marc_lvorm, '') AS marc_deletion_flag,
        COALESCE(c.plant_mat_status, '') AS plant_mat_status,
        COALESCE(c.abc_indicator, '') AS abc_indicator,
        COALESCE(c.purchasing_group, '') AS purchasing_group,
        COALESCE(c.mrp_type, '') AS mrp_type,
        COALESCE(c.mrp_controller, '') AS mrp_controller,
        COALESCE(c.procurement_type, '') AS procurement_type,
        COALESCE(c.planned_deliv_days, 0) AS planned_deliv_days,
        COALESCE(c.gr_processing_days, 0) AS gr_processing_days,
        COALESCE(c.profit_center, '') AS profit_center,
        COALESCE(c.loading_group, '') AS loading_group,
        COALESCE(b."BKLAS", '') AS valuation_class,
        COALESCE(b."VPRSV", '') AS price_control,
        COALESCE(b.moving_avg_price, 0.0) AS moving_avg_price,
        COALESCE(b.standard_price, 0.0) AS standard_price,
        COALESCE(b.price_unit, 1.0) AS price_unit,
        COALESCE(b.total_valuated_stock, 0.0) AS total_valuated_stock,
        COALESCE(b.total_stock_value, 0.0) AS total_stock_value,
        COALESCE(b.mbew_deletion_flag, '') AS mbew_deletion_flag,
        COALESCE(po.po_line_count, 0) AS po_line_count,
        COALESCE(po.total_po_spend, 0.0) AS total_po_spend,
        COALESCE(po.avg_po_net_price, 0.0) AS avg_po_net_price,
        COALESCE(u.alt_uom_count, 1) AS alt_uom_count,
        COALESCE(ei.info_record_count, 0) AS info_record_count
    FROM mara_pool a
    LEFT JOIN marc_primary c ON a."MATNR" = c."MATNR"
    LEFT JOIN mbew_primary b ON a."MATNR" = b."MATNR"
    LEFT JOIN makt_primary m ON a."MATNR" = m."MATNR"
    LEFT JOIN "{schema}"."T134T" t ON a."MTART" = t."MTART" AND t."SPRAS" = 'EN'
    LEFT JOIN "{schema}"."T001W" w ON c."WERKS" = w."WERKS"
    LEFT JOIN "{schema}"."T001" comp ON comp."BUKRS" = '1000'
    LEFT JOIN ekpo_agg po ON a."MATNR" = po."MATNR"
    LEFT JOIN marm_cnt u ON a."MATNR" = u."MATNR"
    LEFT JOIN eina_cnt ei ON a."MATNR" = ei."MATNR"
    ORDER BY a."LAEDA" DESC NULLS LAST, a."ERSDA" DESC NULLS LAST, a."MATNR";
    """


def extract_material_classification_data(limit=None, schemas=None, config=None) -> pd.DataFrame:
    """
    Extracts SAP Material Master & NLP Descriptions from PostgreSQL for all target schemas:
    - If schemas is None, reads schemas list from db_config.yaml (default: ["HANA_P2", "HANA_P3"]).
    - If limit is None or <= 0, extracts 100% of all material records with zero row limit.
    - Joins MARA, MAKT, MARC, MBEW, EKPO, MARM, EINA, T134T, T001W, and T001.
    """
    if config is None:
        config = load_config()

    if schemas is None:
        schemas = config["database"].get("schemas")
        if not schemas:
            single_schema = config["database"].get("schema", "HANA_P2")
            schemas = [single_schema]

    if isinstance(schemas, str):
        schemas = [schemas]

    engine = get_db_engine(config)
    dfs = []

    limit_desc = f"limit={limit:,}" if limit and limit > 0 else "NO LIMIT (100% complete server dataset)"
    print(f"[INFO] Initializing SAP Material extraction across schemas {schemas} with {limit_desc}...")

    for schema in schemas:
        print(f"[INFO] Querying schema '{schema}'...")
        sql = build_schema_query(schema, limit=limit)
        df_schema = pd.read_sql(sql, engine)
        print(f"[SUCCESS] Extracted {len(df_schema):,} records from '{schema}'.")
        dfs.append(df_schema)

    if len(dfs) == 1:
        combined_df = dfs[0]
    else:
        combined_df = pd.concat(dfs, ignore_index=True)

    print(f"[COMPLETE] Total extracted records across all schemas ({', '.join(schemas)}): {len(combined_df):,} rows, {combined_df.shape[1]} attributes.")
    return combined_df


# Aliases for seamless compatibility
extract_material_master_data = extract_material_classification_data
extract_procurement_data = extract_material_classification_data
