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


def build_schema_query(schema_name: str, limit: int = None) -> str:
    """
    Builds single-schema SQL extraction query for SAP incoming invoices.
    Applies clean deduplication on lookup tables (MAKT, T001W, T001, LFA1, LFB1, EKPO)
    and strictly preserves missing values as NULL/blank rather than fake placeholder strings.
    """
    limit_clause = f"LIMIT {limit}" if limit and limit > 0 else ""

    query = f"""
    WITH rbkp_source AS (
        SELECT
            r."BELNR", r."GJAHR", r."BLART", r."BLDAT", r."BUDAT", r."CPUDT",
            r."USNAM", r."TCODE", r."XBLNR", r."BUKRS", r."LIFNR", r."WAERS",
            NULLIF(r."RMWWR"::text, '')::double precision AS "RMWWR",
            NULLIF(r."WMWST1"::text, '')::double precision AS "WMWST1",
            NULLIF(r."BEZNK"::text, '')::double precision AS "BEZNK",
            r."ZTERM", r."ZFBDT", r."ZLSPR", r."ZLSCH", r."RBSTAT"
        FROM "{schema_name}"."RBKP" r
        WHERE NULLIF(r."RMWWR"::text, '') IS NOT NULL 
          AND NULLIF(r."RMWWR"::text, '')::double precision > 0
          AND r."LIFNR" IS NOT NULL AND r."LIFNR" != ''
          AND r."XBLNR" IS NOT NULL AND r."XBLNR" != ''
          AND (r."STBLG" IS NULL OR r."STBLG" = '')
    ),
    rseg_mat AS (
        SELECT DISTINCT ON (s."BELNR", s."GJAHR")
            s."BELNR",
            s."GJAHR",
            s."BUZEI" AS invoice_item,
            s."EBELN" AS po_number,
            s."EBELP" AS po_item,
            s."MATNR" AS material_id,
            s."WERKS" AS plant,
            NULLIF(s."WRBTR"::text, '')::double precision AS item_amount,
            NULLIF(s."MENGE"::text, '')::double precision AS invoiced_qty,
            s."BSTME" AS order_unit,
            s."SHKZG" AS debit_credit_ind,
            s."MWSKZ" AS tax_code
        FROM "{schema_name}"."RSEG" s
        JOIN rbkp_source rr ON s."BELNR" = rr."BELNR" AND s."GJAHR" = rr."GJAHR"
        WHERE NULLIF(s."WRBTR"::text, '') IS NOT NULL 
          AND NULLIF(s."WRBTR"::text, '')::double precision > 0
          AND NULLIF(s."MENGE"::text, '') IS NOT NULL 
          AND NULLIF(s."MENGE"::text, '')::double precision > 0
          AND s."MATNR" IS NOT NULL AND s."MATNR" != ''
        ORDER BY s."BELNR", s."GJAHR", s."BUZEI" ASC
    ),
    lfb1_dedup AS (
        SELECT DISTINCT ON ("LIFNR", "BUKRS")
            "LIFNR", "BUKRS", "REPRF" AS double_inv_check_flag, "ZTERM" AS vendor_pay_terms
        FROM "{schema_name}"."LFB1"
        ORDER BY "LIFNR", "BUKRS"
    ),
    t001_dedup AS (
        SELECT DISTINCT ON ("BUKRS") "BUKRS", "BUTXT", "ORT01", "LAND1", "WAERS"
        FROM "{schema_name}"."T001"
        ORDER BY "BUKRS"
    ),
    t001w_dedup AS (
        SELECT DISTINCT ON ("WERKS") "WERKS", "NAME1", "ORT01"
        FROM "{schema_name}"."T001W"
        ORDER BY "WERKS"
    ),
    lfa1_dedup AS (
        SELECT DISTINCT ON ("LIFNR") "LIFNR", "NAME1", "ORT01", "REGIO", "LAND1"
        FROM "{schema_name}"."LFA1"
        ORDER BY "LIFNR"
    ),
    makt_dedup AS (
        SELECT DISTINCT ON ("MATNR") "MATNR", "MAKTX"
        FROM "{schema_name}"."MAKT"
        WHERE "SPRAS" = 'EN' OR "SPRAS" IS NULL
        ORDER BY "MATNR", "SPRAS" ASC
    ),
    mara_dedup AS (
        SELECT DISTINCT ON ("MATNR") "MATNR", "MTART"
        FROM "{schema_name}"."MARA"
        ORDER BY "MATNR"
    ),
    ekpo_dedup AS (
        SELECT DISTINCT ON ("EBELN", "EBELP")
            "EBELN", "EBELP",
            NULLIF("MENGE"::text, '')::double precision AS po_menge,
            NULLIF("NETPR"::text, '')::double precision AS po_netpr,
            NULLIF("NETWR"::text, '')::double precision AS po_netwr,
            "MATKL", "TXZ01"
        FROM "{schema_name}"."EKPO"
        ORDER BY "EBELN", "EBELP"
    )
    SELECT 
        r."BUKRS" AS company_code,
        c."BUTXT" AS company_name,
        c."ORT01" AS company_city,
        c."LAND1" AS company_country,
        c."WAERS" AS company_currency,
        r."LIFNR" AS vendor_id,
        v."NAME1" AS vendor_name,
        v."ORT01" AS vendor_city,
        v."REGIO" AS vendor_region,
        v."LAND1" AS vendor_country,
        NULLIF(lf."double_inv_check_flag", '') AS double_inv_check_flag,
        rp."plant",
        w."NAME1" AS plant_name,
        COALESCE(w."ORT01", c."ORT01") AS plant_city,
        rp."material_id",
        COALESCE(m."MAKTX", p."TXZ01") AS material_desc,
        ra."MTART" AS material_type,
        p."MATKL" AS material_group,
        r."BELNR" AS invoice_doc_number,
        r."GJAHR" AS fiscal_year,
        rp."invoice_item",
        r."BLART" AS doc_type,
        r."XBLNR" AS reference_invoice_no,
        r."BLDAT" AS invoice_date,
        r."BUDAT" AS posting_date,
        r."CPUDT" AS entry_date,
        r."USNAM" AS posted_by_user,
        r."TCODE" AS transaction_code,
        rp."po_number",
        rp."po_item",
        p."po_menge" AS po_quantity,
        p."po_netpr" AS po_unit_price,
        p."po_netwr" AS po_net_value,
        rp."invoiced_qty",
        rp."order_unit",
        rp."item_amount" AS invoice_item_amount,
        r."RMWWR" AS gross_invoice_amount,
        r."WMWST1" AS tax_amount,
        r."BEZNK" AS unplanned_delivery_cost,
        r."WAERS" AS currency,
        COALESCE(NULLIF(r."ZTERM", ''), NULLIF(lf."vendor_pay_terms", '')) AS payment_terms,
        r."ZFBDT" AS baseline_date,
        r."ZLSPR" AS payment_block,
        r."ZLSCH" AS payment_method,
        r."RBSTAT" AS invoice_status,
        rp."debit_credit_ind"
    FROM rbkp_source r
    JOIN rseg_mat rp ON r."BELNR" = rp."BELNR" AND r."GJAHR" = rp."GJAHR"
    LEFT JOIN t001_dedup c ON r."BUKRS" = c."BUKRS"
    LEFT JOIN lfa1_dedup v ON r."LIFNR" = v."LIFNR"
    LEFT JOIN lfb1_dedup lf ON r."LIFNR" = lf."LIFNR" AND r."BUKRS" = lf."BUKRS"
    LEFT JOIN t001w_dedup w ON rp."plant" = w."WERKS"
    LEFT JOIN ekpo_dedup p ON rp."po_number" = p."EBELN" AND rp."po_item" = p."EBELP"
    LEFT JOIN makt_dedup m ON rp."material_id" = m."MATNR"
    LEFT JOIN mara_dedup ra ON rp."material_id" = ra."MATNR"
    ORDER BY r."BUDAT" DESC NULLS LAST, r."LIFNR", r."RMWWR" DESC
    {limit_clause}
    """
    return query


def extract_invoice_data(limit=None, schema=None, config=None):
    """
    Extracts incoming vendor invoice and procurement master data from SAP tables in PostgreSQL.
    Supports extraction from HANA_P2, HANA_P3, or ALL (both HANA_P2 and HANA_P3 combined).
    Zero limits by default; uses 100% of available data directly from server.
    Preserves exact missing values without synthetic defaults.
    """
    if config is None:
        config = load_config()

    if schema is None:
        schema = config["database"].get("schema", "ALL")

    if limit is None and "ml_parameters" in config:
        limit = config["ml_parameters"].get("limit", None)

    conn = get_db_connection(config)

    # Determine schemas to extract
    target_schemas = []
    if isinstance(schema, list):
        target_schemas = schema
    elif str(schema).upper() in ["ALL", "BOTH", "HANA_P2,HANA_P3", "HANA_P2_P3"]:
        target_schemas = ["HANA_P3", "HANA_P2"]
    else:
        target_schemas = [schema]

    print(f"[INFO] Extracting SAP Incoming Invoices from schema(s): {target_schemas} (limit={'NO LIMIT (All Data)' if not limit else limit})...")

    dfs = []
    seen_keys = set()

    for s in target_schemas:
        print(f"  -> Querying schema '{s}'...")
        q = build_schema_query(s, limit=limit)
        df_s = pd.read_sql(q, conn)
        print(f"     Retrieved {len(df_s):,} records from '{s}'.")

        if len(target_schemas) > 1:
            # Deduplicate by document key (BELNR, GJAHR, BUKRS) across schemas
            df_s['__key'] = df_s['company_code'].astype(str) + '_' + df_s['fiscal_year'].astype(str) + '_' + df_s['invoice_doc_number'].astype(str)
            mask_new = ~df_s['__key'].isin(seen_keys)
            df_unique = df_s[mask_new].drop(columns=['__key'])
            seen_keys.update(df_s['__key'])
            dfs.append(df_unique)
            print(f"     Added {len(df_unique):,} distinct records from '{s}' (total so far: {len(seen_keys):,}).")
        else:
            dfs.append(df_s)

    conn.close()

    if len(dfs) > 1:
        df_final = pd.concat(dfs, ignore_index=True)
    elif len(dfs) == 1:
        df_final = dfs[0]
    else:
        df_final = pd.DataFrame()

    # Apply limit if explicitly specified and > 0
    if limit and limit > 0 and len(df_final) > limit:
        df_final = df_final.iloc[:limit].copy()

    print(f"[SUCCESS] Total Extracted SAP Invoice Records: {len(df_final):,} with {len(df_final.columns)} columns directly from server.")
    return df_final


# Alias for seamless backward compatibility
extract_procurement_data = extract_invoice_data
