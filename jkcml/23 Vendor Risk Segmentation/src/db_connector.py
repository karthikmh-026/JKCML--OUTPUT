"""
Database Connector Module - Use Case 23: Vendor Risk Segmentation (Clustering)
Handles connection and data extraction from PostgreSQL for SAP HANA Vendor Master,
Purchase Orders, Invoice Receipts, Blocked Invoices, Accounts Payable, and Quality Inspection tables.
Supports multi-schema extraction across HANA_P2 and HANA_P3 without row limits (limit=None).
"""
import os
import yaml
import gc
import psycopg2
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config", "db_config.yaml")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_db_engine(config=None):
    """Create SQLAlchemy engine using loaded configuration with robust 30s timeout."""
    if config is None:
        config = load_config()
    db = config["database"]
    uri = f"postgresql+psycopg2://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['dbname']}"
    return create_engine(uri, connect_args={'connect_timeout': 30})


def get_db_connection(config=None):
    """Create raw psycopg2 connection with robust 30s timeout."""
    if config is None:
        config = load_config()
    db = config["database"]
    return psycopg2.connect(
        host=db["host"],
        port=db["port"],
        dbname=db["dbname"],
        user=db["user"],
        password=db["password"],
        connect_timeout=30
    )


def build_schema_extraction_query(schema: str, limit: int = None) -> str:
    """
    Builds an optimized, robust SQL query for extracting Vendor Master, PO, Invoice, Blocked Invoice,
    Open AP, and Quality records from a specified schema (e.g. HANA_P2 or HANA_P3).
    Deduplicates T001 and T077Y to prevent cartesian duplication and casts dates to VARCHAR to ensure
    seamless cross-schema compatibility.
    """
    limit_clause = f"LIMIT {limit}" if limit and limit > 0 else ""

    query = f"""
    WITH t001_dedup AS (
        SELECT DISTINCT ON ("BUKRS")
            "BUKRS", "BUTXT", "ORT01", "LAND1", "WAERS"
        FROM "{schema}"."T001"
        WHERE "BUKRS" IS NOT NULL AND "BUKRS" != ''
        ORDER BY "BUKRS"
    ),
    t077y_dedup AS (
        SELECT DISTINCT ON ("KTOKK")
            "KTOKK", "TXT30"
        FROM "{schema}"."T077Y"
        WHERE "KTOKK" IS NOT NULL AND "KTOKK" != ''
        ORDER BY "KTOKK"
    ),
    ekko_agg AS (
        SELECT
            h."LIFNR" AS lifnr,
            COUNT(DISTINCT h."EBELN") AS po_count,
            COUNT(DISTINCT CASE WHEN h."LOEKZ" IS NOT NULL AND h."LOEKZ" != '' THEN h."EBELN" END) AS deleted_po_count,
            SUM(COALESCE(CAST(NULLIF(REGEXP_REPLACE(CAST(i."NETWR" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0)) AS total_po_net_value,
            AVG(CASE WHEN COALESCE(CAST(NULLIF(REGEXP_REPLACE(CAST(i."NETWR" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0) > 0 THEN CAST(NULLIF(REGEXP_REPLACE(CAST(i."NETWR" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION) END) AS avg_po_item_value,
            MAX(COALESCE(CAST(NULLIF(REGEXP_REPLACE(CAST(i."NETWR" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0)) AS max_po_item_value,
            STDDEV_POP(CASE WHEN COALESCE(CAST(NULLIF(REGEXP_REPLACE(CAST(i."NETWR" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0) > 0 THEN CAST(NULLIF(REGEXP_REPLACE(CAST(i."NETWR" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION) END) AS std_po_item_value,
            COUNT(i."EBELP") AS po_item_count,
            SUM(CASE WHEN i."LOEKZ" IS NOT NULL AND i."LOEKZ" != '' THEN 1 ELSE 0 END) AS deleted_po_item_count,
            SUM(CASE WHEN i."RETPO" = 'X' THEN 1 ELSE 0 END) AS return_po_item_count,
            SUM(CASE WHEN i."ELIKZ" = 'X' THEN 1 ELSE 0 END) AS delivery_completed_items,
            SUM(CASE WHEN i."EREKZ" = 'X' THEN 1 ELSE 0 END) AS final_invoiced_items,
            AVG(COALESCE(CAST(NULLIF(REGEXP_REPLACE(CAST(i."WEBAZ" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0)) AS avg_gr_processing_days,
            AVG(COALESCE(CAST(NULLIF(REGEXP_REPLACE(CAST(i."UEBTO" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0)) AS avg_overdelivery_tol,
            SUM(COALESCE(CAST(NULLIF(REGEXP_REPLACE(CAST(i."MAHNZ" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS INTEGER), 0)) AS total_po_dunning_reminders,
            MIN(CAST(COALESCE(h."BEDAT", h."AEDAT") AS VARCHAR)) AS first_po_date,
            MAX(CAST(COALESCE(h."BEDAT", h."AEDAT") AS VARCHAR)) AS last_po_date,
            COUNT(DISTINCT SUBSTRING(CAST(COALESCE(h."BEDAT", h."AEDAT") AS VARCHAR), 1, 7)) AS active_po_months,
            MAX(h."BUKRS") AS bukrs,
            MAX(h."EKORG") AS ekorg,
            MAX(h."EKGRP") AS ekgrp,
            MAX(h."WAERS") AS waers,
            MAX(h."ZTERM") AS zterm,
            MAX(h."BSART") AS primary_po_type
        FROM "{schema}"."EKKO" h
        LEFT JOIN "{schema}"."EKPO" i ON h."EBELN" = i."EBELN"
        WHERE h."LIFNR" IS NOT NULL AND h."LIFNR" != ''
        GROUP BY h."LIFNR"
    ),
    rbkp_agg AS (
        SELECT
            r."LIFNR" AS lifnr,
            COUNT(DISTINCT r."BELNR") AS invoice_count,
            SUM(COALESCE(CAST(NULLIF(REGEXP_REPLACE(CAST(r."RMWWR" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0)) AS total_invoice_gross_value,
            AVG(CASE WHEN COALESCE(CAST(NULLIF(REGEXP_REPLACE(CAST(r."RMWWR" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0) > 0 THEN CAST(NULLIF(REGEXP_REPLACE(CAST(r."RMWWR" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION) END) AS avg_invoice_value,
            MAX(COALESCE(CAST(NULLIF(REGEXP_REPLACE(CAST(r."RMWWR" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0)) AS max_invoice_value,
            STDDEV_POP(CASE WHEN COALESCE(CAST(NULLIF(REGEXP_REPLACE(CAST(r."RMWWR" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0) > 0 THEN CAST(NULLIF(REGEXP_REPLACE(CAST(r."RMWWR" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION) END) AS std_invoice_value,
            SUM(COALESCE(CAST(NULLIF(REGEXP_REPLACE(CAST(r."BEZNK" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0)) AS total_unplanned_deliv_costs,
            SUM(COALESCE(CAST(NULLIF(REGEXP_REPLACE(CAST(r."WMWST1" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0)) AS total_invoice_tax_value,
            COUNT(DISTINCT CASE WHEN r."ZLSPR" IS NOT NULL AND r."ZLSPR" != '' THEN r."BELNR" END) AS payment_blocked_invoices,
            COUNT(DISTINCT CASE WHEN r."STBLG" IS NOT NULL AND r."STBLG" != '' THEN r."BELNR" END) AS reversed_invoices,
            COUNT(DISTINCT CASE WHEN r."XRECH" IS NULL OR r."XRECH" != 'X' THEN r."BELNR" END) AS credit_memo_or_adj_count,
            MIN(CAST(r."BUDAT" AS VARCHAR)) AS first_invoice_date,
            MAX(CAST(r."BUDAT" AS VARCHAR)) AS last_invoice_date
        FROM "{schema}"."RBKP" r
        WHERE r."LIFNR" IS NOT NULL AND r."LIFNR" != ''
        GROUP BY r."LIFNR"
    ),
    rbkp_blk_agg AS (
        SELECT
            b."LIFNR" AS lifnr,
            COUNT(DISTINCT b."BELNR") AS mrm_blocked_invoice_count
        FROM "{schema}"."RBKP_BLOCKED" b
        WHERE b."LIFNR" IS NOT NULL AND b."LIFNR" != ''
        GROUP BY b."LIFNR"
    ),
    bsik_agg AS (
        SELECT
            ap."LIFNR" AS lifnr,
            MAX(ap."BUKRS") AS ap_bukrs,
            COUNT(DISTINCT ap."BELNR") AS open_ap_item_count,
            SUM(CASE WHEN ap."SHKZG" = 'H' THEN COALESCE(CAST(NULLIF(REGEXP_REPLACE(CAST(ap."WRBTR" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0) ELSE -COALESCE(CAST(NULLIF(REGEXP_REPLACE(CAST(ap."WRBTR" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0) END) AS open_ap_balance,
            COUNT(DISTINCT CASE WHEN ap."ZLSPR" IS NOT NULL AND ap."ZLSPR" != '' THEN ap."BELNR" END) AS blocked_ap_item_count,
            MAX(CAST(ap."BUDAT" AS VARCHAR)) AS last_ap_posting_date,
            AVG(COALESCE(CAST(NULLIF(REGEXP_REPLACE(CAST(ap."PENDAYS" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0)) AS avg_ap_pending_days
        FROM "{schema}"."BSIK" ap
        WHERE ap."LIFNR" IS NOT NULL AND ap."LIFNR" != ''
        GROUP BY ap."LIFNR"
    ),
    qals_agg AS (
        SELECT
            q."LIFNR" AS lifnr,
            COUNT(DISTINCT q."PRUEFLOS") AS inspection_lot_count,
            SUM(COALESCE(CAST(NULLIF(REGEXP_REPLACE(CAST(q."LOSMENGE" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0)) AS total_inspected_qty,
            SUM(COALESCE(CAST(NULLIF(REGEXP_REPLACE(CAST(q."LMENGE01" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0)) AS total_accepted_qty,
            SUM(COALESCE(CAST(NULLIF(REGEXP_REPLACE(CAST(q."LMENGE04" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0) + COALESCE(CAST(NULLIF(REGEXP_REPLACE(CAST(q."LMENGE07" AS VARCHAR), '[^0-9.-]', '', 'g'), '') AS DOUBLE PRECISION), 0.0)) AS total_rejected_blocked_qty
        FROM "{schema}"."QALS" q
        WHERE q."LIFNR" IS NOT NULL AND q."LIFNR" != ''
        GROUP BY q."LIFNR"
    ),
    lfbk_agg AS (
        SELECT
            bk."LIFNR" AS lifnr,
            COUNT(*) AS bank_account_count
        FROM "{schema}"."LFBK" bk
        WHERE bk."LIFNR" IS NOT NULL AND bk."LIFNR" != ''
        GROUP BY bk."LIFNR"
    ),
    lfb1_agg AS (
        SELECT
            lb."LIFNR" AS lifnr,
            MAX(lb."BUKRS") AS bukrs,
            MAX(lb."AKONT") AS akont,
            MAX(lb."ZTERM") AS lfb1_zterm,
            MAX(lb."ZWELS") AS zwels,
            MAX(lb."REPRF") AS reprf,
            MAX(lb."ZAHLS") AS zahls,
            MAX(lb."SPERR") AS lfb1_sperr,
            MAX(lb."LOEVM") AS lfb1_loevm,
            MAX(lb."QSSKZ") AS qsskz,
            MAX(lb."FDGRV") AS fdgrv
        FROM "{schema}"."LFB1" lb
        WHERE lb."LIFNR" IS NOT NULL AND lb."LIFNR" != ''
        GROUP BY lb."LIFNR"
    )
    SELECT
        '{schema}' AS source_schema,
        COALESCE(po.bukrs, lb.bukrs, ap.ap_bukrs, '1000') AS company_code,
        COALESCE(comp."BUTXT", 'J.K. CEMENT LIMITED') AS company_name,
        COALESCE(comp."ORT01", 'Kanpur') AS company_city,
        COALESCE(comp."LAND1", 'IN') AS company_country,
        COALESCE(po.waers, comp."WAERS", 'INR') AS currency,
        v."LIFNR" AS vendor_id,
        COALESCE(v."NAME1", '') AS vendor_name,
        COALESCE(v."ORT01", '') AS vendor_city,
        COALESCE(v."PSTLZ", '') AS postal_code,
        COALESCE(v."REGIO", '') AS vendor_region,
        COALESCE(v."LAND1", 'IN') AS vendor_country,
        COALESCE(v."KTOKK", '') AS account_group,
        COALESCE(ag."TXT30", v."KTOKK") AS account_group_desc,
        COALESCE(v."BRSCH", '') AS industry_code,
        CAST(v."ERDAT" AS VARCHAR) AS master_created_date,
        COALESCE(v."ERNAM", '') AS created_by,
        COALESCE(v."SPERR", lb.lfb1_sperr, '') AS posting_block,
        COALESCE(v."SPERM", '') AS purchasing_block,
        COALESCE(v."SPERQ", '') AS quality_block,
        COALESCE(v."SPERZ", lb.zahls, '') AS payment_block,
        COALESCE(v."LOEVM", lb.lfb1_loevm, '') AS deletion_flag,
        COALESCE(v."STCD3", '') AS gstin_tax_no,
        COALESCE(v."J_1IPANNO", '') AS pan_number,
        COALESCE(v."XZEMP", '') AS alt_payee_allowed,
        COALESCE(v."XCPDK", '') AS one_time_vendor_flag,
        COALESCE(po.ekorg, '1000') AS purchasing_org,
        COALESCE(po.ekgrp, '') AS purchasing_group,
        COALESCE(po.primary_po_type, '') AS primary_po_type,
        COALESCE(po.zterm, lb.lfb1_zterm, '') AS payment_terms,
        COALESCE(lb.akont, '') AS recon_account,
        COALESCE(lb.zwels, '') AS payment_methods,
        COALESCE(lb.reprf, '') AS dup_invoice_check_flag,
        COALESCE(lb.qsskz, '') AS withholding_tax_code,
        COALESCE(lb.fdgrv, '') AS cash_mgmt_group,
        COALESCE(bk.bank_account_count, 0) AS bank_account_count,
        COALESCE(po.po_count, 0) AS po_count,
        COALESCE(po.po_item_count, 0) AS po_item_count,
        COALESCE(po.deleted_po_count, 0) AS deleted_po_count,
        COALESCE(po.deleted_po_item_count, 0) AS deleted_po_item_count,
        COALESCE(po.return_po_item_count, 0) AS return_po_item_count,
        COALESCE(po.delivery_completed_items, 0) AS delivery_completed_items,
        COALESCE(po.final_invoiced_items, 0) AS final_invoiced_items,
        COALESCE(po.total_po_net_value, 0.0) AS total_po_net_value,
        COALESCE(po.avg_po_item_value, 0.0) AS avg_po_item_value,
        COALESCE(po.max_po_item_value, 0.0) AS max_po_item_value,
        COALESCE(po.std_po_item_value, 0.0) AS std_po_item_value,
        COALESCE(po.avg_gr_processing_days, 0.0) AS avg_gr_processing_days,
        COALESCE(po.avg_overdelivery_tol, 0.0) AS avg_overdelivery_tol,
        COALESCE(po.total_po_dunning_reminders, 0) AS total_po_dunning_reminders,
        po.first_po_date,
        po.last_po_date,
        COALESCE(po.active_po_months, 0) AS active_po_months,
        COALESCE(rb.invoice_count, 0) AS invoice_count,
        COALESCE(rb.total_invoice_gross_value, 0.0) AS total_invoice_gross_value,
        COALESCE(rb.avg_invoice_value, 0.0) AS avg_invoice_value,
        COALESCE(rb.max_invoice_value, 0.0) AS max_invoice_value,
        COALESCE(rb.std_invoice_value, 0.0) AS std_invoice_value,
        COALESCE(rb.total_unplanned_deliv_costs, 0.0) AS total_unplanned_deliv_costs,
        COALESCE(rb.total_invoice_tax_value, 0.0) AS total_invoice_tax_value,
        COALESCE(rb.payment_blocked_invoices, 0) AS payment_blocked_invoices,
        COALESCE(rb.reversed_invoices, 0) AS reversed_invoices,
        COALESCE(rb.credit_memo_or_adj_count, 0) AS credit_memo_or_adj_count,
        rb.first_invoice_date,
        rb.last_invoice_date,
        COALESCE(rblk.mrm_blocked_invoice_count, 0) AS mrm_blocked_invoice_count,
        COALESCE(ap.open_ap_item_count, 0) AS open_ap_item_count,
        COALESCE(ap.open_ap_balance, 0.0) AS open_ap_balance,
        COALESCE(ap.blocked_ap_item_count, 0) AS blocked_ap_item_count,
        ap.last_ap_posting_date,
        COALESCE(ap.avg_ap_pending_days, 0.0) AS avg_ap_pending_days,
        COALESCE(qa.inspection_lot_count, 0) AS inspection_lot_count,
        COALESCE(qa.total_inspected_qty, 0.0) AS total_inspected_qty,
        COALESCE(qa.total_accepted_qty, 0.0) AS total_accepted_qty,
        COALESCE(qa.total_rejected_blocked_qty, 0.0) AS total_rejected_blocked_qty
    FROM "{schema}"."LFA1" v
    LEFT JOIN ekko_agg po ON v."LIFNR" = po.lifnr
    LEFT JOIN rbkp_agg rb ON v."LIFNR" = rb.lifnr
    LEFT JOIN rbkp_blk_agg rblk ON v."LIFNR" = rblk.lifnr
    LEFT JOIN bsik_agg ap ON v."LIFNR" = ap.lifnr
    LEFT JOIN qals_agg qa ON v."LIFNR" = qa.lifnr
    LEFT JOIN lfbk_agg bk ON v."LIFNR" = bk.lifnr
    LEFT JOIN lfb1_agg lb ON v."LIFNR" = lb.lifnr
    LEFT JOIN t077y_dedup ag ON v."KTOKK" = ag."KTOKK"
    LEFT JOIN t001_dedup comp ON COALESCE(po.bukrs, lb.bukrs, ap.ap_bukrs, '1000') = comp."BUKRS"
    WHERE v."LIFNR" IS NOT NULL AND v."LIFNR" != ''
    {limit_clause};
    """
    return query


def consolidate_multischema_vendors(dfs: list) -> pd.DataFrame:
    """
    Consolidates vendor records extracted from multiple schemas (e.g. HANA_P2 and HANA_P3):
    - Sums transaction metrics (POs, Spend, Invoices, AP balance, Quality lots) across schemas.
    - Aggregates min of first dates and max of last dates with robust datetime type coercion.
    - Takes latest non-empty master data attributes (Vendor Name, Company Code, Region, Tax IDs, etc.).
    - Tracks source schemas as 'HANA_P2, HANA_P3' for overlap, or single schema name via fast vector operations.
    """
    if len(dfs) == 1:
        return dfs[0]

    # Pre-extract sets for fast schema tracking
    p2_vendors = set()
    p3_vendors = set()
    for d in dfs:
        s_name = str(d['source_schema'].iloc[0]) if len(d) > 0 and 'source_schema' in d.columns else ''
        if 'HANA_P2' in s_name:
            p2_vendors.update(d['vendor_id'].dropna())
        if 'HANA_P3' in s_name:
            p3_vendors.update(d['vendor_id'].dropna())
    both_vendors = p2_vendors & p3_vendors

    combined = pd.concat(dfs, ignore_index=True)
    print(f"[INFO] Merging {len(combined):,} raw vendor records across {len(dfs)} schemas by unique LIFNR...")

    # Robust date type coercion before aggregation
    date_cols = ['master_created_date', 'first_po_date', 'last_po_date', 'first_invoice_date', 'last_invoice_date', 'last_ap_posting_date']
    for c in date_cols:
        if c in combined.columns:
            combined[c] = pd.to_datetime(combined[c], errors='coerce', dayfirst=True)

    sum_cols = [
        'po_count', 'po_item_count', 'deleted_po_count', 'deleted_po_item_count',
        'return_po_item_count', 'delivery_completed_items', 'final_invoiced_items',
        'total_po_net_value', 'total_po_dunning_reminders', 'active_po_months',
        'invoice_count', 'total_invoice_gross_value', 'total_unplanned_deliv_costs',
        'total_invoice_tax_value', 'payment_blocked_invoices', 'reversed_invoices',
        'credit_memo_or_adj_count', 'mrm_blocked_invoice_count', 'open_ap_item_count',
        'open_ap_balance', 'blocked_ap_item_count', 'inspection_lot_count',
        'total_inspected_qty', 'total_accepted_qty', 'total_rejected_blocked_qty',
        'bank_account_count'
    ]
    max_cols = [
        'max_po_item_value', 'max_invoice_value', 'avg_po_item_value',
        'avg_invoice_value', 'avg_gr_processing_days', 'avg_overdelivery_tol',
        'avg_ap_pending_days', 'std_po_item_value', 'std_invoice_value',
        'last_po_date', 'last_invoice_date', 'last_ap_posting_date'
    ]
    min_date_cols = ['master_created_date', 'first_po_date', 'first_invoice_date']

    text_cols = [
        'company_code', 'company_name', 'company_city', 'company_country', 'currency',
        'vendor_name', 'vendor_city', 'postal_code', 'vendor_region', 'vendor_country',
        'account_group', 'account_group_desc', 'industry_code', 'created_by',
        'posting_block', 'purchasing_block', 'quality_block', 'payment_block',
        'deletion_flag', 'gstin_tax_no', 'pan_number', 'alt_payee_allowed',
        'one_time_vendor_flag', 'purchasing_org', 'purchasing_group', 'primary_po_type',
        'payment_terms', 'recon_account', 'payment_methods', 'dup_invoice_check_flag',
        'withholding_tax_code', 'cash_mgmt_group'
    ]

    # Ensure numeric columns are strictly numeric
    for col in sum_cols + ['max_po_item_value', 'max_invoice_value', 'avg_po_item_value', 'avg_invoice_value', 'avg_gr_processing_days', 'avg_overdelivery_tol', 'avg_ap_pending_days', 'std_po_item_value', 'std_invoice_value']:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors='coerce').fillna(0.0)

    agg_dict = {}
    for col in sum_cols:
        if col in combined.columns:
            agg_dict[col] = 'sum'
    for col in max_cols:
        if col in combined.columns:
            agg_dict[col] = 'max'
    for col in min_date_cols:
        if col in combined.columns:
            agg_dict[col] = 'min'
    for col in text_cols:
        if col in combined.columns:
            agg_dict[col] = 'last'

    consolidated = combined.groupby('vendor_id', as_index=False).agg(agg_dict)
    del combined
    gc.collect()

    # Fast vectorized source_schema assignment
    conds = [
        consolidated['vendor_id'].isin(both_vendors),
        consolidated['vendor_id'].isin(p2_vendors)
    ]
    consolidated['source_schema'] = np.select(conds, ['HANA_P2, HANA_P3', 'HANA_P2'], default='HANA_P3')

    print(f"[SUCCESS] Consolidated to {len(consolidated):,} unique vendor records across all schemas.")
    return consolidated


def extract_vendor_risk_data(limit=None, config=None) -> pd.DataFrame:
    """
    Extracts SAP Vendor Master, Purchase Orders, Invoice Receipts, Blocked Invoices,
    Open Accounts Payable, and Quality Inspection records from PostgreSQL.
    Extracts from all configured schemas (HANA_P2 and HANA_P3).
    If limit is None or <= 0, extracts ALL vendor records without limit.
    """
    if config is None:
        config = load_config()

    db_cfg = config.get("database", {})
    schemas = db_cfg.get("schemas", None)
    if not schemas:
        single_schema = db_cfg.get("schema", "HANA_P2")
        schemas = [single_schema]

    engine = get_db_engine(config)
    dfs = []

    for schema in schemas:
        limit_desc = f"limit={limit:,}" if limit and limit > 0 else "NO LIMIT - FULL EXTRACTION"
        print(f"[INFO] Extracting SAP Vendor records from schema '{schema}' ({limit_desc})...")
        query = build_schema_extraction_query(schema=schema, limit=limit)
        try:
            df_schema = pd.read_sql(query, engine)
            print(f"[SUCCESS] Extracted {len(df_schema):,} records from schema '{schema}'.")
            dfs.append(df_schema)
        except Exception as e:
            print(f"[WARNING] Extraction from schema '{schema}' failed: {e}")

    if not dfs:
        raise RuntimeError("Failed to extract data from PostgreSQL database server (192.168.1.44:5432).")

    if len(dfs) > 1:
        consolidated_df = consolidate_multischema_vendors(dfs)
    else:
        consolidated_df = dfs[0]

    # Clean text columns: replace None/nan with empty string ""
    str_cols = consolidated_df.select_dtypes(include=['object', 'string']).columns
    for c in str_cols:
        consolidated_df[c] = consolidated_df[c].fillna('').astype(str).str.strip()
        consolidated_df[c] = consolidated_df[c].replace({'nan': '', 'None': '', 'null': '', 'NaT': '', '<NA>': '', 'undefined': ''})

    print(f"[COMPLETE] Final dataset ready: {len(consolidated_df):,} vendors with {len(consolidated_df.columns)} attributes.")
    return consolidated_df


# Aliases for seamless compatibility
extract_customer_rfm_data = extract_vendor_risk_data
extract_procurement_data = extract_vendor_risk_data
