"""
Database Connector Module - Use Case 21: Material Master Data Quality Scoring
Handles connection and data extraction from PostgreSQL for SAP HANA Material Master & Plant/Valuation tables
across both HANA_P2 and HANA_P3 schemas without row limits.
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


def extract_material_master_data(limit=None, schemas=None, config=None):
    """
    Extracts SAP Material Master governance, plant/MRP, accounting/valuation, UoM, and purchasing info
    records from SAP tables in HANA_P2 and HANA_P3 without limit:
    - MARA: General Material Data (MATNR, MTART, MBRSH, MATKL, BISMT, MEINS, BSTME, SPART, PRDHA, TRAGR,
            XCHPF, MSTAE, VPSTA, PSTAT, LVORM, BRGEW, NTGEW, GEWEI, VOLUM, VOLEH, ERSDA, ERNAM, LAEDA, AENAM)
    - MAKT: Material Descriptions (MATNR, SPRAS, MAKTX)
    - MARC: Plant / MRP / Purchasing Data for Material (MATNR, WERKS, PSTAT, LVORM, MMSTA, MAABC, EKGRP,
            DISMM, DISPO, PLIFZ, WEBAZ, BESKZ, MINBE, EISBE, BSTMI, BSTMA, LOSGR, PRCTR, LADGR, MTVFP)
    - MBEW: Material Valuation & Accounting Data (MATNR, BWKEY, BWTAR, BKLAS, VPRSV, VERPR, STPRS, PEINH,
            LBKUM, SALK3, VMVER, VMSTP, LVORM)
    - MARM: Units of Measure for Material (MATNR, MEINH, UMREZ, UMREN)
    - EINA: Purchasing Info Record General Data (MATNR, INFNR, LIFNR, LOEKZ)
    - T134T: Material Type Descriptions (MTART, MTBEZ)
    - T001W: Plant / Valuation Area Master (WERKS, NAME1, ORT01, REGIO, LAND1)
    - T001: Company Code Master (BUKRS, BUTXT, ORT01, LAND1, WAERS)
    """
    if config is None:
        config = load_config()

    if schemas is None:
        schemas = config["database"].get("schemas", [config["database"].get("schema", "HANA_P2")])
    if isinstance(schemas, str):
        schemas = [schemas]

    engine = get_db_engine(config)
    limit_clause = f"LIMIT {limit}" if limit and limit > 0 else ""

    dfs = []
    for schema_name in schemas:
        print(f"[INFO] Extracting all SAP Material Master records from schema: {schema_name} (limit={limit or 'ALL'})...")
        query = f"""
        WITH marc_primary AS (
            SELECT DISTINCT ON (c."MATNR")
                c."MATNR", c."WERKS",
                c."PSTAT" AS marc_pstat,
                c."LVORM" AS marc_lvorm,
                c."MMSTA" AS plant_mat_status,
                c."MAABC" AS abc_indicator,
                c."EKGRP" AS purchasing_group,
                c."DISMM" AS mrp_type,
                c."DISPO" AS mrp_controller,
                c."PLIFZ" AS planned_deliv_days,
                c."WEBAZ" AS gr_processing_days,
                c."BESKZ" AS procurement_type,
                c."MINBE" AS reorder_point,
                c."EISBE" AS safety_stock,
                c."BSTMI" AS min_lot_size,
                c."BSTMA" AS max_lot_size,
                c."LOSGR" AS costing_lot_size,
                c."PRCTR" AS profit_center,
                c."LADGR" AS loading_group,
                c."MTVFP" AS avail_check_group
            FROM "{schema_name}"."MARC" c
            ORDER BY c."MATNR", c."WERKS" ASC
        ),
        makt_dedup AS (
            SELECT DISTINCT ON (m."MATNR") m."MATNR", m."MAKTX"
            FROM "{schema_name}"."MAKT" m
            WHERE m."SPRAS" = 'EN'
            ORDER BY m."MATNR"
        ),
        t001w_dedup AS (
            SELECT DISTINCT ON (w."WERKS") w."WERKS", w."NAME1", w."ORT01", w."REGIO", w."LAND1"
            FROM "{schema_name}"."T001W" w
            ORDER BY w."WERKS"
        ),
        mbew_dedup AS (
            SELECT DISTINCT ON (b."MATNR", b."BWKEY")
                b."MATNR", b."BWKEY", b."BKLAS", b."VPRSV", b."VERPR", b."STPRS",
                b."PEINH", b."LBKUM", b."SALK3", b."VMVER", b."VMSTP", b."LVORM"
            FROM "{schema_name}"."MBEW" b
            WHERE b."BWTAR" IS NULL OR b."BWTAR"::text = ''
            ORDER BY b."MATNR", b."BWKEY"
        ),
        marm_cnt AS (
            SELECT u."MATNR", COUNT(*) AS alt_uom_count
            FROM "{schema_name}"."MARM" u
            GROUP BY u."MATNR"
        ),
        eina_cnt AS (
            SELECT e."MATNR", COUNT(*) AS info_record_count
            FROM "{schema_name}"."EINA" e
            WHERE e."LOEKZ" IS NULL OR e."LOEKZ"::text = ''
            GROUP BY e."MATNR"
        )
        SELECT
            '{schema_name}' AS source_schema,
            '1000' AS company_code,
            COALESCE(comp."BUTXT", 'J.K. CEMENT LIMITED') AS company_name,
            COALESCE(comp."ORT01", 'Kanpur') AS company_city,
            COALESCE(comp."LAND1", 'IN') AS company_country,
            COALESCE(comp."WAERS", 'INR') AS company_currency,
            a."MATNR" AS material_id,
            m."MAKTX" AS material_desc,
            a."MTART" AS material_type,
            t."MTBEZ" AS material_type_desc,
            a."MBRSH" AS industry_sector,
            a."MATKL" AS material_group,
            a."BISMT" AS old_material_no,
            a."MEINS" AS base_uom,
            a."BSTME" AS order_uom,
            a."SPART" AS division,
            a."PRDHA" AS prod_hierarchy,
            a."TRAGR" AS trans_group,
            a."XCHPF" AS batch_mgmt_flag,
            a."MSTAE" AS cross_plant_status,
            a."VPSTA" AS complete_maint_status,
            a."PSTAT" AS mara_maint_status,
            a."LVORM" AS mara_deletion_flag,
            a."BRGEW" AS gross_weight,
            a."NTGEW" AS net_weight,
            a."GEWEI" AS weight_unit,
            a."VOLUM" AS volume,
            a."VOLEH" AS volume_unit,
            a."ERSDA" AS created_date,
            a."ERNAM" AS created_by,
            a."LAEDA" AS last_changed_date,
            a."AENAM" AS changed_by,
            c."WERKS" AS plant,
            w."NAME1" AS plant_name,
            w."ORT01" AS plant_city,
            w."REGIO" AS plant_region,
            w."LAND1" AS plant_country,
            c.marc_pstat,
            c.marc_lvorm AS marc_deletion_flag,
            c.plant_mat_status,
            c.abc_indicator,
            c.purchasing_group,
            c.mrp_type,
            c.mrp_controller,
            c.procurement_type,
            c.planned_deliv_days,
            c.gr_processing_days,
            c.reorder_point,
            c.safety_stock,
            c.min_lot_size,
            c.max_lot_size,
            c.costing_lot_size,
            c.profit_center,
            c.loading_group,
            c.avail_check_group,
            b."BKLAS" AS valuation_class,
            b."VPRSV" AS price_control,
            b."VERPR" AS moving_avg_price,
            b."STPRS" AS standard_price,
            b."PEINH" AS price_unit,
            b."LBKUM" AS total_valuated_stock,
            b."SALK3" AS total_stock_value,
            b."VMVER" AS prev_moving_price,
            b."VMSTP" AS prev_standard_price,
            b."LVORM" AS mbew_deletion_flag,
            u.alt_uom_count,
            ei.info_record_count
        FROM "{schema_name}"."MARA" a
        LEFT JOIN marc_primary c ON a."MATNR" = c."MATNR"
        LEFT JOIN mbew_dedup b ON c."MATNR" = b."MATNR" AND c."WERKS" = b."BWKEY"
        LEFT JOIN makt_dedup m ON a."MATNR" = m."MATNR"
        LEFT JOIN "{schema_name}"."T134T" t ON a."MTART" = t."MTART" AND t."SPRAS" = 'EN'
        LEFT JOIN t001w_dedup w ON c."WERKS" = w."WERKS"
        LEFT JOIN "{schema_name}"."T001" comp ON comp."BUKRS" = '1000'
        LEFT JOIN marm_cnt u ON a."MATNR" = u."MATNR"
        LEFT JOIN eina_cnt ei ON a."MATNR" = ei."MATNR"
        ORDER BY a."MATNR" ASC
        {limit_clause};
        """
        df_sch = pd.read_sql(query, engine)
        print(f"[SUCCESS] Extracted {len(df_sch):,} records from {schema_name}.")
        dfs.append(df_sch)

    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"[SUCCESS] Combined extraction complete: Total {len(df_combined):,} Material Master records across schemas {schemas}.")
    return df_combined


# Alias for seamless compatibility
extract_procurement_data = extract_material_master_data
