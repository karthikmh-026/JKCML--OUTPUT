"""
Database Connector Module - Use Case 5: Credit Risk Scoring
Handles fast and robust data extraction directly from PostgreSQL (192.168.1.44:5432 / JKC_Input).
Extracts customer master data, open receivables (BSID), cleared settlement history (BSAD),
sales billing (VBRK), sales orders (VBAK), customer account groups (T077X), and company codes (T001)
across BOTH enterprise schemas: HANA_P2 and HANA_P3 without row limits.
Implements push-down SQL aggregations and local persistent caching in data/cache/.
"""
import os
import yaml
import time
import psycopg2
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config", "db_config.yaml")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_db_engine(config: dict = None):
    """Create SQLAlchemy engine using loaded configuration."""
    if config is None:
        config = load_config()
    db = config["database"]
    uri = f"postgresql+psycopg2://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['dbname']}"
    return create_engine(uri)


def get_db_connection(config: dict = None):
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


def extract_credit_risk_data(limit: int = None, config: dict = None, use_cache: bool = True, cache_path: str = None) -> pd.DataFrame:
    """
    Extracts complete SAP Customer Credit and Financial records across BOTH HANA_P2 and HANA_P3 schemas.
    Uses high-performance push-down SQL querying and in-memory aggregation.
    No row limits applied by default (extracts all active commercial accounts).
    Caches extracted data to data/cache/credit_risk_raw_data.csv.
    """
    if cache_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_path = os.path.join(base_dir, "data", "cache", "credit_risk_raw_data.csv")

    if use_cache and os.path.exists(cache_path):
        print(f"[CACHE] Loading cached customer credit dataset from: {cache_path}")
        df = pd.read_csv(cache_path, low_memory=False)
        # Handle string columns: replace NaN with blank string ""
        text_cols = ['customer_name', 'country', 'region', 'city', 'account_group', 'account_group_name', 'company_code', 'company_name']
        for c in text_cols:
            if c in df.columns:
                df[c] = df[c].fillna('').astype(str).str.strip()
        if limit and limit > 0:
            df = df.head(limit)
        print(f"[CACHE] Successfully loaded {len(df):,} customer records.")
        return df

    if config is None:
        config = load_config()

    schemas = config["database"].get("schemas", ["HANA_P2", "HANA_P3"])
    engine = get_db_engine(config)
    t0 = time.time()

    print(f"[INFO] Connecting to PostgreSQL Server ({config['database']['host']}:{config['database']['port']} / {config['database']['dbname']})...")
    print(f"[INFO] Unified full-scale extraction across schemas: {', '.join(schemas)} (No row limits)...")

    # 1. Fetch Open Items (BSID across both schemas)
    print("       1/5: Extracting Open AR Items from BSID (HANA_P2 + HANA_P3)...")
    bsid_parts = []
    for s in schemas:
        bsid_parts.append(f"""
            SELECT 
                "KUNNR"::text as kunnr, "BUKRS"::text as bukrs, "BELNR"::text as belnr, "SHKZG"::text as shkzg, 
                "BUDAT"::text as budat, "ZFBDT"::text as zfbdt, "ZLSPR"::text as zlspr, "DMBTR"::text as dmbtr
            FROM "{s}"."BSID"
            WHERE "KUNNR" IS NOT NULL AND "KUNNR" != ''
        """)
    bsid_sql = " UNION ALL ".join(bsid_parts)
    df_bsid = pd.read_sql(bsid_sql, engine)
    df_bsid['amount'] = pd.to_numeric(df_bsid['dmbtr'].astype(str).str.replace(r'[^0-9.-]', '', regex=True), errors='coerce').fillna(0.0)

    # Date-based overdue calculation
    budat_dt = pd.to_datetime(df_bsid['budat'], errors='coerce')
    zfbdt_dt = pd.to_datetime(df_bsid['zfbdt'], errors='coerce')
    effective_date = zfbdt_dt.fillna(budat_dt)
    snapshot_date = pd.to_datetime('2025-06-25')
    credit_terms_days = 30.0
    days_past_due = (snapshot_date - effective_date).dt.days - credit_terms_days
    df_bsid['days_past_due'] = days_past_due.fillna(0.0).clip(lower=0.0)

    df_bsid['is_blocked'] = df_bsid['zlspr'].fillna('').astype(str).str.strip().ne('')
    df_bsid['signed_amount'] = np.where(df_bsid['shkzg'] == 'S', df_bsid['amount'], -df_bsid['amount'])
    df_bsid['overdue_amount'] = np.where((df_bsid['shkzg'] == 'S') & (df_bsid['days_past_due'] > 0), df_bsid['amount'], 0.0)
    df_bsid['severe_overdue'] = np.where((df_bsid['shkzg'] == 'S') & (df_bsid['days_past_due'] > 90), df_bsid['amount'], 0.0)
    df_bsid['blocked_amount'] = np.where(df_bsid['is_blocked'], df_bsid['amount'], 0.0)

    bsid_agg = df_bsid.groupby('kunnr').agg(
        open_ar_item_count=('belnr', 'nunique'),
        open_ar_balance=('signed_amount', 'sum'),
        overdue_ar_balance=('overdue_amount', 'sum'),
        severe_overdue_balance=('severe_overdue', 'sum'),
        max_overdue_days=('days_past_due', 'max'),
        avg_overdue_days=('days_past_due', 'mean'),
        blocked_ar_item_count=('is_blocked', 'sum'),
        blocked_ar_balance=('blocked_amount', 'sum'),
        primary_bukrs=('bukrs', 'first')
    ).reset_index()
    print(f"            BSID Aggregated: {len(bsid_agg):,} customer accounts.")

    # 2. Fetch Cleared Items (BSAD across both schemas via push-down SQL)
    print("       2/5: Extracting Cleared AR History from BSAD (Push-down SQL across schemas)...")
    bsad_parts = []
    for s in schemas:
        bsad_parts.append(f"""
            SELECT 
                "KUNNR"::text as kunnr,
                COUNT(DISTINCT "BELNR"::text) as cleared_doc_count,
                SUM(CASE WHEN "SHKZG"::text = 'S' THEN CAST(REGEXP_REPLACE(CAST("DMBTR" AS TEXT), '[^0-9.-]', '', 'g') AS NUMERIC) 
                         ELSE -CAST(REGEXP_REPLACE(CAST("DMBTR" AS TEXT), '[^0-9.-]', '', 'g') AS NUMERIC) END) as total_cleared_volume,
                AVG(GREATEST(0, (NULLIF("AUGDT"::text, '')::date - NULLIF("ZFBDT"::text, '')::date))) as avg_clearing_delay_days,
                MAX(GREATEST(0, (NULLIF("AUGDT"::text, '')::date - NULLIF("ZFBDT"::text, '')::date))) as max_clearing_delay_days,
                COUNT(CASE WHEN (NULLIF("AUGDT"::text, '')::date - NULLIF("ZFBDT"::text, '')::date) > 0 THEN 1 END) as late_settlement_count
            FROM "{s}"."BSAD"
            WHERE "KUNNR" IS NOT NULL AND "KUNNR" != ''
              AND NULLIF("AUGDT"::text, '') IS NOT NULL AND NULLIF("ZFBDT"::text, '') IS NOT NULL
            GROUP BY "KUNNR"::text
        """)
    bsad_sql = f"""
        WITH bsad_all AS (
            {" UNION ALL ".join(bsad_parts)}
        )
        SELECT 
            kunnr,
            SUM(cleared_doc_count) as cleared_doc_count,
            SUM(total_cleared_volume) as total_cleared_volume,
            AVG(avg_clearing_delay_days) as avg_clearing_delay_days,
            MAX(max_clearing_delay_days) as max_clearing_delay_days,
            SUM(late_settlement_count) as late_settlement_count
        FROM bsad_all
        GROUP BY kunnr;
    """
    bsad_agg = pd.read_sql(bsad_sql, engine)
    print(f"            BSAD Aggregated: {len(bsad_agg):,} customer accounts.")

    # 3. Fetch Billing Invoices (VBRK across both schemas via push-down SQL)
    print("       3/5: Extracting Billing Documents from VBRK (Push-down SQL across schemas)...")
    vbrk_parts = []
    for s in schemas:
        vbrk_parts.append(f"""
            SELECT 
                "KUNAG"::text as kunnr,
                COUNT(DISTINCT "VBELN"::text) as billing_doc_count,
                SUM(CASE WHEN NULLIF("FKSTO"::text, '') IS NULL OR "FKSTO"::text != 'X' THEN CAST(REGEXP_REPLACE(CAST("NETWR" AS TEXT), '[^0-9.-]', '', 'g') AS NUMERIC) ELSE 0 END) as total_billed_amount,
                AVG(CASE WHEN NULLIF("FKSTO"::text, '') IS NULL OR "FKSTO"::text != 'X' THEN CAST(REGEXP_REPLACE(CAST("NETWR" AS TEXT), '[^0-9.-]', '', 'g') AS NUMERIC) ELSE NULL END) as avg_bill_value,
                STDDEV(CASE WHEN NULLIF("FKSTO"::text, '') IS NULL OR "FKSTO"::text != 'X' THEN CAST(REGEXP_REPLACE(CAST("NETWR" AS TEXT), '[^0-9.-]', '', 'g') AS NUMERIC) ELSE NULL END) as std_bill_value,
                COUNT(CASE WHEN "FKSTO"::text = 'X' OR "FKART"::text IN ('RE', 'G2', 'S1', 'S2', 'ZRE') THEN 1 END) as return_memo_count
            FROM "{s}"."VBRK"
            WHERE "KUNAG" IS NOT NULL AND "KUNAG" != ''
            GROUP BY "KUNAG"::text
        """)
    vbrk_sql = f"""
        WITH vbrk_all AS (
            {" UNION ALL ".join(vbrk_parts)}
        )
        SELECT 
            kunnr,
            SUM(billing_doc_count) as billing_doc_count,
            SUM(total_billed_amount) as total_billed_amount,
            AVG(avg_bill_value) as avg_bill_value,
            AVG(std_bill_value) as std_bill_value,
            SUM(return_memo_count) as return_memo_count
        FROM vbrk_all
        GROUP BY kunnr;
    """
    vbrk_agg = pd.read_sql(vbrk_sql, engine)
    print(f"            VBRK Aggregated: {len(vbrk_agg):,} customer accounts.")

    # 4. Fetch Sales Orders (VBAK across both schemas via push-down SQL)
    print("       4/5: Extracting Sales Orders from VBAK (Push-down SQL across schemas)...")
    vbak_parts = []
    for s in schemas:
        vbak_parts.append(f"""
            SELECT 
                "KUNNR"::text as kunnr,
                COUNT(DISTINCT "VBELN"::text) as sales_order_count,
                SUM(CAST(REGEXP_REPLACE(CAST("NETWR" AS TEXT), '[^0-9.-]', '', 'g') AS NUMERIC)) as total_order_amount,
                COUNT(CASE WHEN (NULLIF(TRIM("LIFSK"::text), '') IS NOT NULL) OR (NULLIF(TRIM("FAKSK"::text), '') IS NOT NULL) OR "CMGST"::text IN ('B', 'C') THEN 1 END) as blocked_order_count
            FROM "{s}"."VBAK"
            WHERE "KUNNR" IS NOT NULL AND "KUNNR" != ''
            GROUP BY "KUNNR"::text
        """)
    vbak_sql = f"""
        WITH vbak_all AS (
            {" UNION ALL ".join(vbak_parts)}
        )
        SELECT 
            kunnr,
            SUM(sales_order_count) as sales_order_count,
            SUM(total_order_amount) as total_order_amount,
            SUM(blocked_order_count) as blocked_order_count
        FROM vbak_all
        GROUP BY kunnr;
    """
    vbak_agg = pd.read_sql(vbak_sql, engine)
    print(f"            VBAK Aggregated: {len(vbak_agg):,} customer accounts.")

    # Active customer universe across all transactions
    active_customers = list(
        set(bsid_agg['kunnr']) | set(bsad_agg['kunnr']) | set(vbrk_agg['kunnr']) | set(vbak_agg['kunnr'])
    )
    if limit and limit > 0:
        active_customers = active_customers[:limit]
    print(f"       Identified {len(active_customers):,} active customer accounts across all financial & commercial tables.")

    # 5. Fetch Customer Master & Company Code details (KNA1, KNB1, T077X, T001)
    print("       5/5: Joining Customer Master & Company Code details (KNA1, KNB1, T077X, T001)...")
    kna1_parts = [f"""
        SELECT 
            "KUNNR"::text as kunnr, "NAME1"::text as customer_name, "LAND1"::text as country, "REGIO"::text as region, 
            "ORT01"::text as city, "KTOKD"::text as account_group, "BRSCH"::text as industry_code, "ERDAT"::text as customer_created_date,
            "SPERR"::text as central_posting_block, "LOEVM"::text as central_deletion_flag
        FROM "{s}"."KNA1" WHERE "KUNNR" IS NOT NULL AND "KUNNR" != ''
    """ for s in schemas]
    kna1_sql = " UNION ".join(kna1_parts)
    df_kna1 = pd.read_sql(kna1_sql, engine).drop_duplicates(subset=['kunnr'], keep='first')

    knb1_parts = [f"""
        SELECT 
            "KUNNR"::text as kunnr, "BUKRS"::text as company_code, "AKONT"::text as gl_reconciliation_acc, 
            "ZTERM"::text as payment_terms, "ZWELS"::text as payment_methods, "SPERR"::text as co_code_posting_block, 
            "LOEVM"::text as co_code_deletion_flag, "VLIBB"::text as credit_limit, "WEBTR"::text as bill_of_exchange_limit, 
            "KULTG"::text as allowed_excess_days
        FROM "{s}"."KNB1" WHERE "KUNNR" IS NOT NULL AND "KUNNR" != ''
    """ for s in schemas]
    knb1_sql = " UNION ".join(knb1_parts)
    df_knb1 = pd.read_sql(knb1_sql, engine).drop_duplicates(subset=['kunnr'], keep='first')

    t077x_parts = [f"""
        SELECT DISTINCT "KTOKD"::text as account_group, "TXT30"::text as account_group_name
        FROM "{s}"."T077X" WHERE "SPRAS"::text = 'E'
    """ for s in schemas]
    t077x_sql = " UNION ".join(t077x_parts)
    try:
        df_t077x = pd.read_sql(t077x_sql, engine).drop_duplicates(subset=['account_group'], keep='first')
    except Exception:
        df_t077x = pd.DataFrame(columns=['account_group', 'account_group_name'])

    t001_parts = [f"""
        SELECT DISTINCT "BUKRS"::text as company_code, "BUTXT"::text as company_name
        FROM "{s}"."T001"
    """ for s in schemas]
    t001_sql = " UNION ".join(t001_parts)
    try:
        df_t001 = pd.read_sql(t001_sql, engine).drop_duplicates(subset=['company_code'], keep='first')
    except Exception:
        df_t001 = pd.DataFrame(columns=['company_code', 'company_name'])

    # Assemble complete customer dataset
    df_base = pd.DataFrame({'kunnr': active_customers})
    df_merged = (
        df_base
        .merge(df_kna1, on='kunnr', how='left')
        .merge(df_knb1, on='kunnr', how='left')
        .merge(df_t077x, on='account_group', how='left')
        .merge(bsid_agg, on='kunnr', how='left')
        .merge(bsad_agg, on='kunnr', how='left')
        .merge(vbrk_agg, on='kunnr', how='left')
        .merge(vbak_agg, on='kunnr', how='left')
    )

    df_merged.rename(columns={'kunnr': 'customer_id'}, inplace=True)

    # Missing value handling:
    # Text columns: keep if present, else blank ""
    text_cols = [
        'customer_name', 'country', 'region', 'city', 'account_group', 'account_group_name',
        'industry_code', 'gl_reconciliation_acc', 'payment_terms', 'payment_methods',
        'central_posting_block', 'central_deletion_flag', 'co_code_posting_block', 'co_code_deletion_flag'
    ]
    for c in text_cols:
        if c in df_merged.columns:
            df_merged[c] = df_merged[c].fillna('').astype(str).str.strip()

    # Company code & Company Name: map from T001, leave blank "" if not present
    df_merged['company_code'] = df_merged['company_code'].replace('', np.nan).fillna(df_merged['primary_bukrs']).fillna('').astype(str).str.strip()
    df_merged = df_merged.merge(df_t001, on='company_code', how='left')
    df_merged['company_name'] = df_merged['company_name'].fillna('').astype(str).str.strip()

    # Numerical columns: fill missing with 0.0
    fill_zero_cols = [
        'credit_limit', 'bill_of_exchange_limit', 'allowed_excess_days',
        'open_ar_item_count', 'open_ar_balance', 'overdue_ar_balance', 'severe_overdue_balance',
        'max_overdue_days', 'avg_overdue_days', 'blocked_ar_item_count', 'blocked_ar_balance',
        'cleared_doc_count', 'total_cleared_volume', 'avg_clearing_delay_days', 'max_clearing_delay_days',
        'late_settlement_count', 'billing_doc_count', 'total_billed_amount', 'avg_bill_value',
        'std_bill_value', 'return_memo_count', 'sales_order_count', 'total_order_amount', 'blocked_order_count'
    ]
    for c in fill_zero_cols:
        if c in df_merged.columns:
            df_merged[c] = pd.to_numeric(df_merged[c], errors='coerce').fillna(0.0)

    elapsed = time.time() - t0
    print(f"[SUCCESS] Extracted {len(df_merged):,} Customer Credit records with {df_merged.shape[1]} attributes in {elapsed:.2f}s.")

    # Save to local cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df_merged.to_csv(cache_path, index=False)
    print(f"[CACHE] Saved raw extraction data to: {cache_path}")

    return df_merged
