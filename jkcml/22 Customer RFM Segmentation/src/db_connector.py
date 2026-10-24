"""
Database Connector Module - Use Case 20: Customer RFM Segmentation (Clustering)
Handles connection and data extraction from PostgreSQL for SAP HANA Customer Master,
Billing Documents, Sales Orders, and Accounts Receivable tables across HANA_P2 and HANA_P3.
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


def extract_customer_rfm_data(limit=None, config=None):
    """
    Extracts SAP Customer Master, Billing (RFM), Sales Order, and Open Accounts Receivable
    records from SAP HANA tables across HANA_P2 and HANA_P3 without limits:
    - KNA1: General Data in Customer Master
    - KNVV: Customer Master Sales Area Data
    - KNB1: Customer Master Company Code Data
    - VBRK: Billing Document Header Data
    - VBAK: Sales Document Header Data
    - BSID: Accounting Secondary Index for Customers - Open Items
    - T077X: Customer Account Group Names
    - T001: Company Code Master

    HANA_P3 contains 100% of the HANA_P2 dataset plus all additional billing and sales records
    (6.86M VBRK invoices, 4.67M VBAK orders, 1.23M BSID items, 543k KNA1 customers).
    """
    if config is None:
        config = load_config()
    
    # Primary schema containing full cumulative data from HANA_P2 and HANA_P3
    schema = config["database"].get("primary_schema", "HANA_P3")
    engine = get_db_engine(config)

    limit_clause = f"LIMIT {limit}" if limit and int(limit) > 0 else ""

    query = f"""
    WITH vbrk_norm AS (
        SELECT
            "KUNAG" AS kunnr,
            "VKORG" AS vkorg,
            "VTWEG" AS vtweg,
            "SPART" AS spart,
            "BUKRS" AS bukrs,
            "KDGRP" AS kdgrp,
            "BZIRK" AS bzirk,
            "PLTYP" AS pltyp,
            "ZTERM" AS zterm,
            "WAERK" AS waerk,
            "VBELN" AS vbeln,
            "FKSTO" AS fksto,
            "FKART" AS fkart,
            CASE
                WHEN "FKDAT" ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN CAST("FKDAT" AS DATE)
                WHEN "FKDAT" ~ '^[0-9]{2}\\.[0-9]{2}\\.[0-9]{4}' THEN TO_DATE("FKDAT", 'DD.MM.YYYY')
                ELSE NULL
            END AS fkdat_clean,
            COALESCE(CAST(NULLIF(REGEXP_REPLACE("NETWR", '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0) AS netwr_clean,
            COALESCE(CAST(NULLIF(REGEXP_REPLACE("MWSBK", '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0) AS mwsbk_clean
        FROM "{schema}"."VBRK"
        WHERE "KUNAG" IS NOT NULL AND "KUNAG" != ''
    ),
    vbrk_agg AS (
        SELECT
            kunnr,
            MAX(vkorg) AS vkorg,
            MAX(vtweg) AS vtweg,
            MAX(spart) AS spart,
            MAX(bukrs) AS bukrs,
            MAX(kdgrp) AS kdgrp,
            MAX(bzirk) AS bzirk,
            MAX(pltyp) AS pltyp,
            MAX(zterm) AS zterm,
            MAX(waerk) AS waerk,
            COUNT(DISTINCT vbeln) AS billing_doc_count,
            COUNT(DISTINCT CASE WHEN fksto = 'X' OR fkart IN ('RE', 'G2', 'S1', 'S2', 'ZRE') THEN vbeln END) AS cancelled_or_return_bills,
            SUM(CASE WHEN fksto IS NULL OR fksto = '' THEN netwr_clean ELSE 0.0 END) AS total_billed_net_value,
            SUM(CASE WHEN fksto IS NULL OR fksto = '' THEN mwsbk_clean ELSE 0.0 END) AS total_billed_tax_value,
            AVG(CASE WHEN fksto IS NULL OR fksto = '' THEN netwr_clean ELSE NULL END) AS avg_bill_value,
            MAX(CASE WHEN fksto IS NULL OR fksto = '' THEN netwr_clean ELSE 0.0 END) AS max_bill_value,
            STDDEV_POP(CASE WHEN fksto IS NULL OR fksto = '' THEN netwr_clean ELSE NULL END) AS std_bill_value,
            MIN(fkdat_clean) AS first_billing_date,
            MAX(fkdat_clean) AS last_billing_date,
            COUNT(DISTINCT fkdat_clean) AS distinct_billing_days,
            COUNT(DISTINCT SUBSTRING(CAST(fkdat_clean AS VARCHAR), 1, 7)) AS active_billing_months
        FROM vbrk_norm
        GROUP BY kunnr
    ),
    vbak_norm AS (
        SELECT
            "KUNNR" AS kunnr,
            "VBELN" AS vbeln,
            "LIFSK" AS lifsk,
            "FAKSK" AS faksk,
            "CMGST" AS cmgst,
            CASE
                WHEN "ERDAT" ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN CAST("ERDAT" AS DATE)
                WHEN "ERDAT" ~ '^[0-9]{2}\\.[0-9]{2}\\.[0-9]{4}' THEN TO_DATE("ERDAT", 'DD.MM.YYYY')
                ELSE NULL
            END AS erdat_clean,
            COALESCE(CAST(NULLIF(REGEXP_REPLACE("NETWR", '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0) AS netwr_clean
        FROM "{schema}"."VBAK"
        WHERE "KUNNR" IS NOT NULL AND "KUNNR" != ''
    ),
    vbak_agg AS (
        SELECT
            kunnr,
            COUNT(DISTINCT vbeln) AS sales_order_count,
            SUM(netwr_clean) AS total_order_net_value,
            AVG(netwr_clean) AS avg_order_value,
            MIN(erdat_clean) AS first_order_date,
            MAX(erdat_clean) AS last_order_date,
            COUNT(DISTINCT CASE WHEN (lifsk IS NOT NULL AND lifsk != '') OR (faksk IS NOT NULL AND faksk != '') OR cmgst IN ('B', 'C') THEN vbeln END) AS blocked_order_count
        FROM vbak_norm
        GROUP BY kunnr
    ),
    bsid_norm AS (
        SELECT
            "KUNNR" AS kunnr,
            "BELNR" AS belnr,
            "SHKZG" AS shkzg,
            CASE
                WHEN "BUDAT" ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}' THEN CAST("BUDAT" AS DATE)
                WHEN "BUDAT" ~ '^[0-9]{2}\\.[0-9]{2}\\.[0-9]{4}' THEN TO_DATE("BUDAT", 'DD.MM.YYYY')
                ELSE NULL
            END AS budat_clean,
            COALESCE(CAST(NULLIF(REGEXP_REPLACE("PENDAYS", '[^0-9-]', '', 'g'), '') AS INTEGER), 0) AS pendays_clean,
            COALESCE(CAST(NULLIF(REGEXP_REPLACE("DMBTR", '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0) AS dmbtr_clean
        FROM "{schema}"."BSID"
        WHERE "KUNNR" IS NOT NULL AND "KUNNR" != ''
    ),
    bsid_agg AS (
        SELECT
            kunnr,
            COUNT(DISTINCT belnr) AS open_ar_item_count,
            SUM(CASE WHEN shkzg = 'S' THEN dmbtr_clean ELSE -dmbtr_clean END) AS open_ar_balance,
            MAX(budat_clean) AS last_ar_posting_date,
            AVG(pendays_clean) AS avg_ar_pending_days
        FROM bsid_norm
        GROUP BY kunnr
    ),
    active_customers AS (
        SELECT kunnr FROM vbrk_agg
        UNION
        SELECT kunnr FROM vbak_agg
        UNION
        SELECT kunnr FROM bsid_agg
    ),
    knvv_primary AS (
        SELECT DISTINCT ON (kv."KUNNR")
            kv."KUNNR", kv."VKORG", kv."VTWEG", kv."SPART", kv."KDGRP", kv."BZIRK",
            kv."PLTYP", kv."KONDA", kv."KLABC", kv."ZTERM" AS knvv_zterm,
            kv."VKBUR", kv."VKGRP", kv."WAERS" AS knvv_waers,
            kv."ZCUSTOMER_GROUP", kv."ZCUSTOMER_GRP_DES", kv."ZRURAL_URBAN",
            kv."ZMETRO_NONMETRO", kv."ZDISTRIBUTOR_TYP", kv."ZTIER"
        FROM "{schema}"."KNVV" kv
        JOIN active_customers ac ON kv."KUNNR" = ac.kunnr
        ORDER BY kv."KUNNR", kv."VKORG" ASC
    ),
    knb1_primary AS (
        SELECT DISTINCT ON (kb."KUNNR")
            kb."KUNNR", kb."BUKRS", kb."AKONT", kb."ZTERM" AS knb1_zterm,
            kb."ZWELS", kb."FDGRV", kb."SPERR" AS knb1_sperr, kb."LOEVM" AS knb1_loevm
        FROM "{schema}"."KNB1" kb
        JOIN active_customers ac ON kb."KUNNR" = ac.kunnr
        ORDER BY kb."KUNNR", kb."BUKRS" ASC
    ),
    t077x_clean AS (
        SELECT DISTINCT ON ("KTOKD") "KTOKD", "TXT30"
        FROM "{schema}"."T077X"
        WHERE "SPRAS" IN ('E', 'EN')
        ORDER BY "KTOKD", (CASE WHEN "SPRAS" = 'EN' THEN 1 ELSE 2 END)
    ),
    t001_clean AS (
        SELECT DISTINCT ON ("BUKRS") "BUKRS", "BUTXT", "ORT01", "LAND1", "WAERS"
        FROM "{schema}"."T001"
        ORDER BY "BUKRS"
    )
    SELECT
        COALESCE(vb.bukrs, kb."BUKRS", '1000') AS company_code,
        COALESCE(comp."BUTXT", 'J.K. CEMENT LIMITED') AS company_name,
        COALESCE(comp."ORT01", 'Kanpur') AS company_city,
        COALESCE(comp."LAND1", 'IN') AS company_country,
        COALESCE(vb.waerk, kv.knvv_waers, comp."WAERS", 'INR') AS currency,
        ac.kunnr AS customer_id,
        COALESCE(k."NAME1", '') AS customer_name,
        COALESCE(k."ORT01", '') AS customer_city,
        COALESCE(k."PSTLZ", '') AS postal_code,
        COALESCE(k."REGIO", '') AS customer_region,
        COALESCE(k."LAND1", 'IN') AS customer_country,
        COALESCE(k."KTOKD", '') AS account_group,
        COALESCE(ag."TXT30", k."KTOKD", '') AS account_group_desc,
        COALESCE(k."KUKLA", '') AS customer_classification,
        COALESCE(k."BRSCH", '') AS industry_code,
        k."ERDAT" AS master_created_date,
        COALESCE(k."ERNAM", '') AS created_by,
        COALESCE(k."AUFSD", '') AS order_block,
        COALESCE(k."FAKSD", '') AS billing_block,
        COALESCE(k."LIFSD", '') AS delivery_block,
        COALESCE(k."SPERR", kb.knb1_sperr, '') AS posting_block,
        COALESCE(k."LOEVM", kb.knb1_loevm, '') AS deletion_flag,
        COALESCE(k."STCD3", '') AS gstin_tax_no,
        COALESCE(vb.vkorg, kv."VKORG", '1000') AS sales_org,
        COALESCE(vb.vtweg, kv."VTWEG", '10') AS dist_channel,
        COALESCE(vb.spart, kv."SPART", '10') AS division,
        COALESCE(vb.kdgrp, kv."KDGRP", '') AS customer_group,
        COALESCE(vb.bzirk, kv."BZIRK", '') AS sales_district,
        COALESCE(vb.pltyp, kv."PLTYP", '') AS price_list_type,
        COALESCE(kv."KONDA", '') AS price_group,
        COALESCE(kv."KLABC", '') AS abc_class,
        COALESCE(kv."VKBUR", '') AS sales_office,
        COALESCE(kv."VKGRP", '') AS sales_group,
        COALESCE(vb.zterm, kv.knvv_zterm, kb.knb1_zterm, '') AS payment_terms,
        COALESCE(kb."AKONT", '') AS recon_account,
        COALESCE(kb."ZWELS", '') AS payment_methods,
        COALESCE(kv."ZCUSTOMER_GRP_DES", kv."ZCUSTOMER_GROUP", '') AS channel_segment_desc,
        COALESCE(kv."ZRURAL_URBAN", '') AS rural_urban_flag,
        COALESCE(kv."ZMETRO_NONMETRO", '') AS metro_tier_flag,
        COALESCE(kv."ZDISTRIBUTOR_TYP", '') AS distributor_type,
        COALESCE(kv."ZTIER", '') AS dealer_tier,
        COALESCE(vb.billing_doc_count, 0) AS billing_doc_count,
        COALESCE(vb.cancelled_or_return_bills, 0) AS cancelled_or_return_bills,
        COALESCE(vb.total_billed_net_value, 0.0) AS total_billed_net_value,
        COALESCE(vb.total_billed_tax_value, 0.0) AS total_billed_tax_value,
        COALESCE(vb.avg_bill_value, 0.0) AS avg_bill_value,
        COALESCE(vb.max_bill_value, 0.0) AS max_bill_value,
        COALESCE(vb.std_bill_value, 0.0) AS std_bill_value,
        vb.first_billing_date,
        vb.last_billing_date,
        COALESCE(vb.distinct_billing_days, 0) AS distinct_billing_days,
        COALESCE(vb.active_billing_months, 0) AS active_billing_months,
        COALESCE(vo.sales_order_count, 0) AS sales_order_count,
        COALESCE(vo.total_order_net_value, 0.0) AS total_order_net_value,
        COALESCE(vo.avg_order_value, 0.0) AS avg_order_value,
        vo.first_order_date,
        vo.last_order_date,
        COALESCE(vo.blocked_order_count, 0) AS blocked_order_count,
        COALESCE(ar.open_ar_item_count, 0) AS open_ar_item_count,
        COALESCE(ar.open_ar_balance, 0.0) AS open_ar_balance,
        ar.last_ar_posting_date,
        COALESCE(ar.avg_ar_pending_days, 0.0) AS avg_ar_pending_days
    FROM active_customers ac
    LEFT JOIN "{schema}"."KNA1" k ON ac.kunnr = k."KUNNR"
    LEFT JOIN vbrk_agg vb ON ac.kunnr = vb.kunnr
    LEFT JOIN vbak_agg vo ON ac.kunnr = vo.kunnr
    LEFT JOIN bsid_agg ar ON ac.kunnr = ar.kunnr
    LEFT JOIN knvv_primary kv ON ac.kunnr = kv."KUNNR"
    LEFT JOIN knb1_primary kb ON ac.kunnr = kb."KUNNR"
    LEFT JOIN t077x_clean ag ON k."KTOKD" = ag."KTOKD"
    LEFT JOIN t001_clean comp ON COALESCE(vb.bukrs, kb."BUKRS", '1000') = comp."BUKRS"
    ORDER BY COALESCE(vb.total_billed_net_value, 0.0) DESC, ac.kunnr ASC
    {limit_clause};
    """

    limit_desc = f"limit={limit}" if limit and int(limit) > 0 else "NO LIMIT (all data from server)"
    print(f"[INFO] Extracting SAP Customer Master, Billing (VBRK), Sales Order (VBAK), and AR (BSID) from {schema} ({limit_desc})...")
    df = pd.read_sql(query, engine)
    print(f"[SUCCESS] Extracted {len(df):,} Customer RFM records with {len(df.columns)} attributes directly from server.")
    return df


# Aliases for seamless compatibility
extract_material_master_data = extract_customer_rfm_data
extract_procurement_data = extract_customer_rfm_data
