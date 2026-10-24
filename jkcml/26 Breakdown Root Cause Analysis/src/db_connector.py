"""
Database Connector Module - Use Case 26: Breakdown Root Cause Analysis (NLP)
Handles connection and data extraction from PostgreSQL for SAP HANA Plant Maintenance (PM)
Order Master Data (AUFK), Order Header Scheduling & Priority (AFKO), Order Confirmations &
Technician Execution Logs (AFRU), Goods Movements / Spare Parts Consumed (AUFM), Material
Descriptions (MAKT), Work Center Master (CRHD), Plant Master (T001W), and Company Code Master (T001)
across both HANA_P2 and HANA_P3 schemas without any arbitrary record limit.
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


def extract_breakdown_rca_data(limit=None, config=None):
    """
    Extracts ALL SAP Plant Maintenance (PM) Order Master Data, Maintenance Texts, Confirmation Texts,
    Spare Parts Consumed, and Work Center Metadata across HANA_P2 and HANA_P3 schemas:
    - AUFK: Order Master Data (AUFNR, AUART, AUTYP='30', KTEXT, LTEXT, BUKRS, WERKS, WAWRK,
            VAPLZ, KOSTV, PRCTR, GSBER, ERDAT, ERNAM, AEDAT, IDAT1, IDAT2, IDAT3,
            PHAS0, PHAS1, PHAS2, PHAS3, STORT, LOEKZ, SCOPE)
    - AFKO: Order Header Scheduling & Priority (AUFNR, GSTRP, GLTRP, GSTRI, GETRI, GLTRI, FTRMI,
            APRIO, DISPO, FEVOR, RSNUM, AUFPL) across HANA_P3 and HANA_P2
    - AFRU: Order Confirmations (AUFNR, RUECK, RMZHL, LTXA1, ARBID, WERKS, ISDD, IEDD,
            IDAUR, IDAUE, ISM01..ISM03, GRUND, AUERU, STOKZ) across HANA_P2 and HANA_P3
    - AUFM: Goods Movements for Order / Spare Parts Issued (AUFNR, MATNR, BWART, MENGE,
            MEINS, DMBTR, WAERS, LGORT)
    - MAKT: Material Descriptions for Spare Parts (MATNR, SPRAS='EN', MAKTX)
    - CRHD: Work Center Header (ARBPL, WERKS, OBJID, VERWE, STAND, VERAN, KTSCH)
    - T001W: Plant Master (WERKS, NAME1, ORT01)
    - T001: Company Code Master (BUKRS, BUTXT, ORT01, WAERS)

    If limit is None or 0, ALL data from the server is extracted with NO limit.
    """
    if config is None:
        config = load_config()
    engine = get_db_engine(config)

    limit_clause = f"LIMIT {int(limit)}" if limit and int(limit) > 0 else ""

    query = f"""
    WITH pm_orders AS (
        SELECT
            k."AUFNR", k."AUART", k."AUTYP", k."KTEXT", k."LTEXT",
            k."BUKRS", k."WERKS", k."WAWRK", k."VAPLZ", k."KOSTV", k."PRCTR", k."GSBER",
            k."ERDAT", k."ERNAM", k."AEDAT", k."IDAT1", k."IDAT2", k."IDAT3",
            k."PDAT1", k."PDAT2", k."PHAS0", k."PHAS1", k."PHAS2", k."PHAS3",
            k."STORT", k."LOEKZ", k."SCOPE"
        FROM "HANA_P2"."AUFK" k
        WHERE k."AUTYP" = '30' AND k."KTEXT" IS NOT NULL AND k."KTEXT" != ''
        ORDER BY k."AUFNR" ASC
        {limit_clause}
    ),
    afko_p3 AS (
        SELECT DISTINCT ON (af."AUFNR")
            af."AUFNR",
            af."GSTRP", af."GLTRP", af."GSTRI", af."GETRI", af."GLTRI", af."FTRMI",
            af."APRIO", af."DISPO", af."FEVOR", af."RSNUM", af."AUFPL"
        FROM "HANA_P3"."AFKO" af
        JOIN pm_orders p ON af."AUFNR" = p."AUFNR"
        ORDER BY af."AUFNR"
    ),
    afko_p2 AS (
        SELECT DISTINCT ON (af."AUFNR")
            af."AUFNR",
            af."GSTRP"::text, af."GLTRP"::text, af."GSTRI"::text, af."GETRI"::text, af."GLTRI"::text, af."FTRMI"::text,
            af."APRIO", af."DISPO", af."FEVOR", af."RSNUM", af."AUFPL"
        FROM "HANA_P2"."AFKO" af
        JOIN pm_orders p ON af."AUFNR" = p."AUFNR"
        ORDER BY af."AUFNR"
    ),
    afru_p2 AS (
        SELECT
            r."AUFNR",
            COUNT(*) AS confirmation_count,
            MAX(NULLIF(r."LTXA1", '')) AS confirmation_text,
            SUM(COALESCE(r."IDAUR", 0)) AS total_actual_duration,
            MAX(COALESCE(r."IDAUE", '')) AS duration_unit,
            SUM(COALESCE(r."ISM01", 0) + COALESCE(r."ISM02", 0)) AS total_labor_hours,
            MIN(r."ISDD"::text) AS exec_start_date,
            MAX(r."IEDD"::text) AS exec_end_date,
            MAX(COALESCE(r."GRUND", '')) AS variance_reason_code,
            MAX(COALESCE(r."AUERU", '')) AS final_conf_flag,
            MAX(COALESCE(r."ARBID", '')) AS conf_work_center_id
        FROM "HANA_P2"."AFRU" r
        JOIN pm_orders p ON r."AUFNR" = p."AUFNR"
        WHERE (r."STOKZ" IS NULL OR r."STOKZ" = '')
        GROUP BY r."AUFNR"
    ),
    afru_p3 AS (
        SELECT
            r."AUFNR",
            COUNT(*) AS confirmation_count,
            MAX(NULLIF(r."LTXA1", '')) AS confirmation_text,
            SUM(COALESCE(r."IDAUR", 0)) AS total_actual_duration,
            MAX(COALESCE(r."IDAUE", '')) AS duration_unit,
            SUM(COALESCE(r."ISM01", 0) + COALESCE(r."ISM02", 0)) AS total_labor_hours,
            MIN(r."ISDD"::text) AS exec_start_date,
            MAX(r."IEDD"::text) AS exec_end_date,
            MAX(COALESCE(r."GRUND", '')) AS variance_reason_code,
            MAX(COALESCE(r."AUERU", '')) AS final_conf_flag,
            MAX(COALESCE(r."ARBID", '')) AS conf_work_center_id
        FROM "HANA_P3"."AFRU" r
        JOIN pm_orders p ON r."AUFNR" = p."AUFNR"
        WHERE (r."STOKZ" IS NULL OR r."STOKZ" = '')
        GROUP BY r."AUFNR"
    ),
    makt_dedup AS (
        SELECT DISTINCT ON ("MATNR")
            "MATNR", "MAKTX"
        FROM "HANA_P2"."MAKT"
        WHERE "SPRAS" = 'EN'
        ORDER BY "MATNR"
    ),
    aufm_agg AS (
        SELECT
            m."AUFNR",
            COUNT(*) AS goods_movement_count,
            COUNT(DISTINCT m."MATNR") AS distinct_spares_count,
            MAX(m."MATNR") AS primary_spare_matnr,
            STRING_AGG(DISTINCT COALESCE(mk."MAKTX", m."MATNR"), ' | ') AS spare_parts_texts,
            SUM(CASE WHEN m."BWART" = '262' THEN -1 ELSE 1 END * COALESCE(NULLIF(REPLACE(m."DMBTR"::text, ',', ''), '')::numeric, 0)) AS total_spare_cost_inr,
            SUM(COALESCE(NULLIF(REPLACE(m."MENGE"::text, ',', ''), '')::numeric, 0)) AS total_spare_qty,
            MAX(COALESCE(m."MEINS", '')) AS spare_uom,
            MAX(COALESCE(m."LGORT", '')) AS storage_location
        FROM "HANA_P2"."AUFM" m
        JOIN pm_orders p ON m."AUFNR" = p."AUFNR"
        LEFT JOIN makt_dedup mk ON m."MATNR" = mk."MATNR"
        GROUP BY m."AUFNR"
    ),
    crhd_dedup AS (
        SELECT DISTINCT ON ("ARBPL", "WERKS")
            "ARBPL", "WERKS", "OBJID", "VERWE" AS work_center_cat, "STAND" AS wc_location, "VERAN" AS wc_person_resp, "KTSCH" AS wc_std_text_key
        FROM "HANA_P2"."CRHD"
        ORDER BY "ARBPL", "WERKS"
    ),
    t001w_dedup AS (
        SELECT DISTINCT ON ("WERKS")
            "WERKS", "NAME1", "ORT01"
        FROM "HANA_P2"."T001W"
        ORDER BY "WERKS"
    ),
    t001_dedup AS (
        SELECT DISTINCT ON ("BUKRS")
            "BUKRS", "BUTXT", "ORT01", "WAERS"
        FROM "HANA_P2"."T001"
        ORDER BY "BUKRS"
    )
    SELECT
        COALESCE(k."BUKRS", '1000') AS company_code,
        COALESCE(comp."BUTXT", 'J.K. CEMENT LIMITED') AS company_name,
        COALESCE(comp."ORT01", 'Kanpur') AS company_city,
        COALESCE(comp."WAERS", 'INR') AS company_currency,
        k."AUFNR" AS order_number,
        COALESCE(k."AUART", '') AS order_type,
        COALESCE(k."AUTYP", '30') AS order_category,
        COALESCE(k."KTEXT", '') AS maintenance_text,
        COALESCE(NULLIF(r2.confirmation_text, ''), NULLIF(r3.confirmation_text, ''), '') AS confirmation_text,
        COALESCE(m.spare_parts_texts, '') AS spare_parts_texts,
        COALESCE(k."WERKS", k."WAWRK", '') AS plant,
        COALESCE(w."NAME1", '') AS plant_name,
        COALESCE(w."ORT01", '') AS plant_city,
        COALESCE(k."VAPLZ", '') AS work_center,
        COALESCE(c.work_center_cat, '') AS work_center_cat,
        COALESCE(c.wc_person_resp, '') AS wc_person_resp,
        COALESCE(k."KOSTV", '') AS cost_center,
        COALESCE(k."PRCTR", '') AS profit_center,
        COALESCE(k."GSBER", '') AS business_area,
        COALESCE(k."STORT", c.wc_location, '') AS location_code,
        COALESCE(NULLIF(af3."APRIO", ''), COALESCE(af2."APRIO", '')) AS order_priority,
        COALESCE(NULLIF(af3."DISPO", ''), COALESCE(af2."DISPO", '')) AS mrp_controller,
        COALESCE(NULLIF(af3."FEVOR", ''), COALESCE(af2."FEVOR", '')) AS maintenance_supervisor,
        k."ERDAT"::text AS created_date,
        COALESCE(k."ERNAM", '') AS created_by,
        k."IDAT1"::text AS release_date,
        k."IDAT2"::text AS teco_date,
        k."IDAT3"::text AS close_date,
        COALESCE(NULLIF(af3."GSTRP", ''), COALESCE(af2."GSTRP", '')) AS basic_start_date,
        COALESCE(NULLIF(af3."GLTRP", ''), COALESCE(af2."GLTRP", '')) AS basic_finish_date,
        COALESCE(NULLIF(af3."GSTRI", ''), COALESCE(af2."GSTRI", '')) AS actual_start_date,
        COALESCE(NULLIF(af3."GETRI", ''), COALESCE(af2."GETRI", '')) AS actual_finish_date,
        COALESCE(NULLIF(r2.exec_start_date, ''), COALESCE(r3.exec_start_date, '')) AS exec_start_date,
        COALESCE(NULLIF(r2.exec_end_date, ''), COALESCE(r3.exec_end_date, '')) AS exec_end_date,
        COALESCE(k."PHAS0", '') AS phase_created,
        COALESCE(k."PHAS1", '') AS phase_released,
        COALESCE(k."PHAS2", '') AS phase_teco,
        COALESCE(k."PHAS3", '') AS phase_closed,
        GREATEST(COALESCE(r2.confirmation_count, 0), COALESCE(r3.confirmation_count, 0)) AS confirmation_count,
        GREATEST(COALESCE(r2.total_actual_duration, 0.0), COALESCE(r3.total_actual_duration, 0.0)) AS total_actual_duration,
        COALESCE(NULLIF(r2.duration_unit, ''), COALESCE(r3.duration_unit, '')) AS duration_unit,
        GREATEST(COALESCE(r2.total_labor_hours, 0.0), COALESCE(r3.total_labor_hours, 0.0)) AS total_labor_hours,
        COALESCE(NULLIF(r2.variance_reason_code, ''), COALESCE(r3.variance_reason_code, '')) AS variance_reason_code,
        COALESCE(NULLIF(r2.final_conf_flag, ''), COALESCE(r3.final_conf_flag, '')) AS final_conf_flag,
        COALESCE(m.goods_movement_count, 0) AS goods_movement_count,
        COALESCE(m.distinct_spares_count, 0) AS distinct_spares_count,
        COALESCE(m.primary_spare_matnr, '') AS primary_spare_matnr,
        COALESCE(m.total_spare_cost_inr, 0.0) AS total_spare_cost_inr,
        COALESCE(m.total_spare_qty, 0.0) AS total_spare_qty,
        COALESCE(m.spare_uom, '') AS spare_uom,
        COALESCE(m.storage_location, '') AS storage_location
    FROM pm_orders k
    LEFT JOIN afko_p3 af3 ON k."AUFNR" = af3."AUFNR"
    LEFT JOIN afko_p2 af2 ON k."AUFNR" = af2."AUFNR"
    LEFT JOIN afru_p2 r2 ON k."AUFNR" = r2."AUFNR"
    LEFT JOIN afru_p3 r3 ON k."AUFNR" = r3."AUFNR"
    LEFT JOIN aufm_agg m ON k."AUFNR" = m."AUFNR"
    LEFT JOIN crhd_dedup c ON k."VAPLZ" = c."ARBPL" AND COALESCE(k."WERKS", k."WAWRK") = c."WERKS"
    LEFT JOIN t001w_dedup w ON COALESCE(k."WERKS", k."WAWRK") = w."WERKS"
    LEFT JOIN t001_dedup comp ON COALESCE(k."BUKRS", '1000') = comp."BUKRS";
    """

    limit_desc = f"limit={limit:,}" if limit and int(limit) > 0 else "ALL DATA (NO LIMIT)"
    print(f"[INFO] Extracting SAP Plant Maintenance Orders across HANA_P2 & HANA_P3 ({limit_desc})...")
    df = pd.read_sql(query, engine)
    print(f"[SUCCESS] Extracted {len(df):,} Breakdown & Maintenance Order records with {len(df.columns)} attributes directly from server.")
    return df


# Aliases for seamless compatibility
extract_maintenance_rca_data = extract_breakdown_rca_data
extract_plant_maintenance_data = extract_breakdown_rca_data
