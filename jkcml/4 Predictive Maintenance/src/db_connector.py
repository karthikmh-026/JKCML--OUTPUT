"""
Database Connector Module - Use Case 4: Predictive Maintenance Classification
(Predict Equipment Failures Before Breakdown)
Handles connection and data extraction from PostgreSQL for SAP HANA Plant Maintenance (PM)
across both HANA_P2 and HANA_P3 schemas without limits:
- AUFK: Order Master Data (AUFNR, AUART, AUTYP='30', KTEXT, LTEXT, BUKRS, WERKS, VAPLZ,
        KOSTV, PRCTR, GSBER, ERDAT, AEDAT, IDAT1, IDAT2, IDAT3, PHAS0..PHAS3, LOEKZ)
- AFKO: Order Header Scheduling (AUFNR, GSTRP, GLTRP, GSTRI, GETRI, GLTRI, FTRMI,
        APRIO, DISPO, FEVOR, RSNUM, AUFPL)
- AFRU: Order Confirmations (AUFNR, RUECK, RMZHL, LTXA1, IDAUR, IDAUE, ISM01, ISM02,
        GRUND, AUERU, ISDD, IEDD)
- AUFM: Spare Parts Issued (AUFNR, MATNR, BWART, MENGE, MEINS, DMBTR, WAERS)
- MAKT: Material Descriptions for Spare Parts (MATNR, MAKTX)
- CRHD: Work Center Header (ARBPL, WERKS, OBJID, VERWE, STAND)
- T001W: Plant Master (WERKS, NAME1)
- T001: Company Code Master (BUKRS, BUTXT)
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


def extract_predictive_maintenance_data(limit=None, config=None):
    """
    Extracts SAP Plant Maintenance (PM) Order Master Data, Scheduling, Confirmation Texts,
    Spare Parts Consumed, and Work Center Metadata combining HANA_P2 and HANA_P3 schemas.

    When limit is None or <= 0, extracts ALL real data from the database with no limits.
    """
    if config is None:
        config = load_config()
    engine = get_db_engine(config)

    limit_clause = f"LIMIT {int(limit)}" if limit and int(limit) > 0 else ""

    query = f"""
    WITH pm_orders AS (
        SELECT DISTINCT ON ("AUFNR")
            "AUFNR", "AUART", "AUTYP", "KTEXT", "LTEXT", "BUKRS", "WERKS", "VAPLZ",
            "KOSTV", "PRCTR", "GSBER", "ERDAT"::text AS "ERDAT", "AEDAT"::text AS "AEDAT",
            "IDAT1"::text AS "IDAT1", "IDAT2"::text AS "IDAT2", "IDAT3"::text AS "IDAT3",
            "PHAS0", "PHAS1", "PHAS2", "PHAS3", "LOEKZ"
        FROM (
            SELECT "AUFNR", "AUART", "AUTYP", "KTEXT", "LTEXT", "BUKRS", "WERKS", "VAPLZ",
                   "KOSTV", "PRCTR", "GSBER", "ERDAT"::text, "AEDAT"::text, "IDAT1"::text, "IDAT2"::text, "IDAT3"::text,
                   "PHAS0", "PHAS1", "PHAS2", "PHAS3", "LOEKZ"
            FROM "HANA_P2"."AUFK"
            WHERE "AUTYP" = '30' AND "KTEXT" IS NOT NULL AND "KTEXT" != ''
            UNION ALL
            SELECT "AUFNR", "AUART", "AUTYP", "KTEXT", "LTEXT", "BUKRS", "WERKS", "VAPLZ",
                   "KOSTV", "PRCTR", "GSBER", "ERDAT"::text, "AEDAT"::text, "IDAT1"::text, "IDAT2"::text, "IDAT3"::text,
                   "PHAS0", "PHAS1", "PHAS2", "PHAS3", "LOEKZ"
            FROM "HANA_P3"."AUFK"
            WHERE "AUTYP" = '30' AND "KTEXT" IS NOT NULL AND "KTEXT" != ''
        ) u
        ORDER BY "AUFNR" ASC
        {limit_clause}
    ),
    afko_all AS (
        SELECT "AUFNR", "GSTRP"::text, "GLTRP"::text, "GSTRI"::text, "GETRI"::text, "GLTRI"::text, "FTRMI"::text,
               "APRIO", "DISPO", "FEVOR", "RSNUM", "AUFPL"
        FROM "HANA_P2"."AFKO"
        UNION ALL
        SELECT "AUFNR", "GSTRP"::text, "GLTRP"::text, "GSTRI"::text, "GETRI"::text, "GLTRI"::text, "FTRMI"::text,
               "APRIO", "DISPO", "FEVOR", "RSNUM", "AUFPL"
        FROM "HANA_P3"."AFKO"
    ),
    afko_dedup AS (
        SELECT DISTINCT ON (af."AUFNR")
            af."AUFNR", af."GSTRP", af."GLTRP", af."GSTRI", af."GETRI", af."GLTRI", af."FTRMI",
            af."APRIO", af."DISPO", af."FEVOR", af."RSNUM", af."AUFPL"
        FROM afko_all af
        JOIN pm_orders p ON af."AUFNR" = p."AUFNR"
        ORDER BY af."AUFNR"
    ),
    afru_all AS (
        SELECT "AUFNR", "LTXA1", "IDAUR"::numeric, "IDAUE", "ISM01"::numeric, "ISM02"::numeric,
               "ISDD"::text, "IEDD"::text, "GRUND", "AUERU", "ARBID", "STOKZ"
        FROM "HANA_P2"."AFRU"
        UNION ALL
        SELECT "AUFNR", "LTXA1", "IDAUR"::numeric, "IDAUE", "ISM01"::numeric, "ISM02"::numeric,
               "ISDD"::text, "IEDD"::text, "GRUND", "AUERU", "ARBID", "STOKZ"
        FROM "HANA_P3"."AFRU"
    ),
    afru_agg AS (
        SELECT
            r."AUFNR",
            COUNT(*) AS confirmation_count,
            MAX(NULLIF(r."LTXA1", '')) AS confirmation_text,
            STRING_AGG(DISTINCT NULLIF(r."LTXA1", ''), ' | ') AS all_confirmation_texts,
            SUM(COALESCE(r."IDAUR", 0)) AS total_actual_duration,
            MAX(COALESCE(r."IDAUE", '')) AS duration_unit,
            SUM(COALESCE(r."ISM01", 0) + COALESCE(r."ISM02", 0)) AS total_labor_hours,
            MIN(r."ISDD") AS exec_start_date,
            MAX(r."IEDD") AS exec_end_date,
            MAX(COALESCE(r."GRUND", '')) AS variance_reason_code,
            MAX(COALESCE(r."AUERU", '')) AS final_conf_flag,
            MAX(COALESCE(r."ARBID", '')) AS conf_work_center_id
        FROM afru_all r
        JOIN pm_orders p ON r."AUFNR" = p."AUFNR"
        WHERE r."STOKZ" IS NULL OR r."STOKZ" = ''
        GROUP BY r."AUFNR"
    ),
    makt_all AS (
        SELECT "MATNR", "MAKTX", "SPRAS" FROM "HANA_P2"."MAKT"
        UNION ALL
        SELECT "MATNR", "MAKTX", "SPRAS" FROM "HANA_P3"."MAKT"
    ),
    makt_dedup AS (
        SELECT DISTINCT ON ("MATNR")
            "MATNR", "MAKTX"
        FROM makt_all
        WHERE "SPRAS" = 'E' OR "SPRAS" = 'EN' OR "SPRAS" IS NOT NULL
        ORDER BY "MATNR", "SPRAS"
    ),
    aufm_all AS (
        SELECT "AUFNR", "MATNR", "BWART", "MENGE"::numeric, "MEINS", "DMBTR"::numeric, "WAERS"
        FROM "HANA_P2"."AUFM"
        UNION ALL
        SELECT "AUFNR", "MATNR", "BWART", NULLIF("MENGE", '')::numeric, "MEINS", NULLIF("DMBTR", '')::numeric, "WAERS"
        FROM "HANA_P3"."AUFM"
    ),
    aufm_agg AS (
        SELECT
            m."AUFNR",
            COUNT(DISTINCT m."MATNR") AS spare_parts_item_count,
            SUM(COALESCE(m."DMBTR", 0)) AS total_spare_parts_cost,
            SUM(COALESCE(m."MENGE", 0)) AS total_parts_quantity,
            STRING_AGG(DISTINCT NULLIF(kt."MAKTX", ''), ' | ') AS spare_parts_descriptions
        FROM aufm_all m
        JOIN pm_orders p ON m."AUFNR" = p."AUFNR"
        LEFT JOIN makt_dedup kt ON m."MATNR" = kt."MATNR"
        WHERE m."BWART" IN ('261', '262')
        GROUP BY m."AUFNR"
    ),
    crhd_all AS (
        SELECT "ARBPL", "WERKS", "VERWE" AS work_center_use, "STAND" AS work_center_location FROM "HANA_P2"."CRHD"
        UNION ALL
        SELECT "ARBPL", "WERKS", "VERWE" AS work_center_use, "STAND" AS work_center_location FROM "HANA_P3"."CRHD"
    ),
    crhd_dedup AS (
        SELECT DISTINCT ON ("ARBPL", "WERKS")
            "ARBPL", "WERKS", work_center_use, work_center_location
        FROM crhd_all
        ORDER BY "ARBPL", "WERKS"
    ),
    t001w_all AS (
        SELECT "WERKS", "NAME1" AS plant_name FROM "HANA_P2"."T001W"
        UNION ALL
        SELECT "WERKS", "NAME1" AS plant_name FROM "HANA_P3"."T001W"
    ),
    t001w_dedup AS (
        SELECT DISTINCT ON ("WERKS") "WERKS", plant_name
        FROM t001w_all
        ORDER BY "WERKS"
    ),
    t001_all AS (
        SELECT "BUKRS", "BUTXT" AS company_name FROM "HANA_P2"."T001"
        UNION ALL
        SELECT "BUKRS", "BUTXT" AS company_name FROM "HANA_P3"."T001"
    ),
    t001_dedup AS (
        SELECT DISTINCT ON ("BUKRS") "BUKRS", company_name
        FROM t001_all
        ORDER BY "BUKRS"
    )
    SELECT
        p."AUFNR" AS order_number,
        p."AUART" AS order_type,
        p."AUTYP" AS order_category,
        p."KTEXT" AS order_description,
        p."LTEXT" AS long_text_flag,
        p."BUKRS" AS company_code,
        p."WERKS" AS plant_code,
        p."VAPLZ" AS work_center,
        p."KOSTV" AS cost_center,
        p."PRCTR" AS profit_center,
        p."GSBER" AS business_area,
        p."ERDAT" AS creation_date,
        p."AEDAT" AS change_date,
        p."IDAT1" AS release_date,
        p."IDAT2" AS teco_date,
        p."IDAT3" AS close_date,
        p."PHAS0" AS phase_created,
        p."PHAS1" AS phase_released,
        p."PHAS2" AS phase_completed,
        p."PHAS3" AS phase_closed,
        p."LOEKZ" AS deletion_indicator,
        k."GSTRP" AS planned_start_date,
        k."GLTRP" AS planned_finish_date,
        k."GSTRI" AS actual_start_date,
        k."GETRI" AS actual_finish_date,
        k."GLTRI" AS confirmed_finish_date,
        k."FTRMI" AS sched_release_date,
        k."APRIO" AS priority_code,
        k."DISPO" AS mrp_controller,
        k."FEVOR" AS maint_scheduler,
        k."RSNUM" AS reservation_number,
        COALESCE(r.confirmation_count, 0) AS confirmation_count,
        COALESCE(r.confirmation_text, '') AS confirmation_text,
        COALESCE(r.all_confirmation_texts, '') AS all_confirmation_texts,
        COALESCE(r.total_actual_duration, 0.0) AS total_actual_duration,
        COALESCE(r.duration_unit, 'H') AS duration_unit,
        COALESCE(r.total_labor_hours, 0.0) AS total_labor_hours,
        r.exec_start_date,
        r.exec_end_date,
        COALESCE(r.variance_reason_code, '') AS variance_reason_code,
        COALESCE(r.final_conf_flag, '') AS final_conf_flag,
        COALESCE(m.spare_parts_item_count, 0) AS spare_parts_item_count,
        COALESCE(m.total_spare_parts_cost, 0.0) AS total_spare_parts_cost,
        COALESCE(m.total_parts_quantity, 0.0) AS total_parts_quantity,
        COALESCE(m.spare_parts_descriptions, '') AS spare_parts_descriptions,
        COALESCE(c.work_center_use, '') AS work_center_use,
        COALESCE(c.work_center_location, '') AS work_center_location,
        COALESCE(w.plant_name, '') AS plant_name,
        COALESCE(t.company_name, '') AS company_name
    FROM pm_orders p
    LEFT JOIN afko_dedup k ON p."AUFNR" = k."AUFNR"
    LEFT JOIN afru_agg r ON p."AUFNR" = r."AUFNR"
    LEFT JOIN aufm_agg m ON p."AUFNR" = m."AUFNR"
    LEFT JOIN crhd_dedup c ON p."VAPLZ" = c."ARBPL" AND p."WERKS" = c."WERKS"
    LEFT JOIN t001w_dedup w ON p."WERKS" = w."WERKS"
    LEFT JOIN t001_dedup t ON p."BUKRS" = t."BUKRS"
    ORDER BY p."AUFNR" ASC;
    """

    df = pd.read_sql_query(query, engine)
    return df
