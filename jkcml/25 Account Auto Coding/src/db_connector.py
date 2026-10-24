"""
Database Connector Module - Use Case 25: Account Auto Coding (NLP)
Handles connection and data extraction from PostgreSQL for SAP HANA Financial Accounting Document
Line Items (BSEG), G/L Account Master Chart of Accounts (SKA1), G/L Account Descriptions (SKAT),
Company Code G/L Master (SKB1), Posting Keys (TBSL), Document Types (T003T), Purchase Order
Account Assignments & Short Texts (EKPO), Vendor Master (LFA1), Plant (T001W), and Company (T001)
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


def build_schema_query(schema: str, limit: int = None) -> str:
    """
    Builds the extraction query for a given schema (HANA_P2 or HANA_P3) with safe schema-aware
    column mapping (handling HSN_SAC / PLC_SUP vs HSN SAC / PLC SUP) and strict missing-value preservation.
    """
    limit_clause = f"LIMIT {limit}" if limit and limit > 0 else ""

    # Schema-specific column name mapping for GST fields between HANA_P2 and HANA_P3
    if schema.upper() == "HANA_P3":
        hsn_col = 'b."HSN_SAC"'
        plc_col = 'b."PLC_SUP"'
    else:
        hsn_col = 'b."HSN SAC"'
        plc_col = 'b."PLC SUP"'

    query = f"""
    WITH bseg_pool AS (
        SELECT
            b."BUKRS", b."BELNR", b."GJAHR", b."BUZEI",
            COALESCE(NULLIF(b."HKONT", ''), NULLIF(b."SAKNR", ''), NULLIF(b."GKONT", '')) AS gl_account,
            b."HKONT", b."SAKNR", b."GKONT", b."GKART",
            b."KOART", b."BSCHL", b."SHKZG",
            b."DMBTR", b."WRBTR", b."SGTXT", b."ZUONR",
            b."KOSTL", b."KSTAR", b."PRCTR", b."GSBER", b."BUPLA",
            b."WERKS", b."EBELN", b."EBELP", b."LIFNR", b."KUNNR",
            b."AUFNR", b."ANLN1", b."PROJK",
            b."H_BLART", b."H_BUDAT", b."H_BLDAT", b."H_MONAT", b."H_WAERS", b."H_BSTAT",
            b."BTYPE", {hsn_col} AS hsn_sac, {plc_col} AS plc_sup,
            b."ZTERM", b."ZLSCH", b."AUGBL", b."AUGDT", b."XREF1", b."XREF2", b."XREF3"
        FROM "{schema}"."BSEG" b
        WHERE b."SGTXT" IS NOT NULL AND TRIM(b."SGTXT") != ''
          AND (b."HKONT" IS NOT NULL AND TRIM(b."HKONT") != '')
        {limit_clause}
    ),
    ska1_dedup AS (
        SELECT DISTINCT ON ("SAKNR")
            "KTOPL", "SAKNR", "KTOKS", "XBILK", "GVTYP", "GLACCOUNT_TYPE", "GLACCOUNT_SUBTYPE",
            "FUNC_AREA", "XLOEV", "XSPEB",
            "Short Text" AS ska1_short_text,
            "G/L Acct Long Text" AS ska1_long_text
        FROM "{schema}"."SKA1"
        WHERE "KTOPL" = '1000'
        ORDER BY "SAKNR"
    ),
    skat_dedup AS (
        SELECT DISTINCT ON ("SAKNR")
            "SAKNR", "TXT20", "TXT50", "MCOD1"
        FROM "{schema}"."SKAT"
        WHERE "KTOPL" = '1000' AND ("SPRAS" = 'EN' OR "SPRAS" IS NULL)
        ORDER BY "SAKNR"
    ),
    skb1_dedup AS (
        SELECT DISTINCT ON ("BUKRS", "SAKNR")
            "BUKRS", "SAKNR", "MITKZ", "MWSKZ" AS skb1_mwskz, "FSTAG", "XOPVW", "XINTB", "XGKON", "WAERS" AS skb1_waers, "ALTKT"
        FROM "{schema}"."SKB1"
        WHERE "BUKRS" = '1000'
        ORDER BY "BUKRS", "SAKNR"
    ),
    tbsl_dedup AS (
        SELECT DISTINCT ON ("BSCHL")
            "BSCHL", "Posting Key Name" AS posting_key_name
        FROM "{schema}"."TBSL"
        ORDER BY "BSCHL"
    ),
    t003t_dedup AS (
        SELECT DISTINCT ON ("BLART")
            "BLART", "LTEXT" AS doc_type_desc
        FROM "{schema}"."T003T"
        WHERE "SPRAS" = 'EN' OR "SPRAS" IS NULL
        ORDER BY "BLART"
    ),
    lfa1_dedup AS (
        SELECT DISTINCT ON (v."LIFNR")
            v."LIFNR", v."NAME1" AS vendor_name, v."ORT01" AS vendor_city, v."KTOKK" AS vendor_group
        FROM "{schema}"."LFA1" v
        JOIN (SELECT DISTINCT "LIFNR" FROM bseg_pool WHERE "LIFNR" IS NOT NULL AND "LIFNR" != '') bp
          ON v."LIFNR" = bp."LIFNR"
        ORDER BY v."LIFNR"
    ),
    ekpo_dedup AS (
        SELECT DISTINCT ON (p."EBELN", p."EBELP")
            p."EBELN", p."EBELP", p."TXZ01" AS po_short_text, p."MATKL" AS po_material_group
        FROM "{schema}"."EKPO" p
        JOIN (SELECT DISTINCT "EBELN", "EBELP" FROM bseg_pool WHERE "EBELN" IS NOT NULL AND "EBELN" != '') bp
          ON p."EBELN" = bp."EBELN" AND p."EBELP" = bp."EBELP"
        ORDER BY p."EBELN", p."EBELP"
    )
    SELECT
        '{schema}' AS schema_name,
        COALESCE(b."BUKRS", '1000') AS company_code,
        COALESCE(comp."BUTXT", 'J.K. CEMENT LIMITED') AS company_name,
        COALESCE(comp."ORT01", 'Kanpur') AS company_city,
        COALESCE(comp."WAERS", 'INR') AS company_currency,
        b."BELNR" AS doc_number,
        b."GJAHR" AS fiscal_year,
        b."BUZEI" AS line_item,
        b.gl_account,
        COALESCE(NULLIF(t."TXT50", ''), NULLIF(a.ska1_long_text, ''), NULLIF(t."TXT20", ''), COALESCE(a.ska1_short_text, '')) AS gl_account_desc,
        COALESCE(NULLIF(t."TXT20", ''), COALESCE(a.ska1_short_text, '')) AS gl_short_text,
        COALESCE(a."KTOKS", '') AS gl_account_group,
        COALESCE(a."GLACCOUNT_TYPE", '') AS gl_account_type,
        COALESCE(a."XBILK", '') AS is_balance_sheet_acct,
        COALESCE(a."GVTYP", '') AS pl_statement_type,
        COALESCE(a."FUNC_AREA", '') AS functional_area,
        COALESCE(kb."MITKZ", '') AS recon_account_type,
        COALESCE(kb."FSTAG", '') AS field_status_group,
        COALESCE(kb."XOPVW", '') AS open_item_mgmt_flag,
        COALESCE(kb."XINTB", '') AS auto_posting_only_flag,
        COALESCE(b."GKONT", '') AS offsetting_gl_account,
        COALESCE(NULLIF(gt."TXT50", ''), COALESCE(ga.ska1_long_text, '')) AS offsetting_gl_desc,
        COALESCE(ga."KTOKS", '') AS offsetting_account_group,
        COALESCE(b."GKART", '') AS offsetting_account_type,
        COALESCE(b."KOART", '') AS account_type,
        COALESCE(b."BSCHL", '') AS posting_key,
        COALESCE(pk.posting_key_name, '') AS posting_key_desc,
        COALESCE(b."SHKZG", '') AS debit_credit_ind,
        COALESCE(b."DMBTR", '0') AS amount_local,
        COALESCE(b."WRBTR", '0') AS amount_doc_curr,
        COALESCE(b."H_WAERS", 'INR') AS doc_currency,
        COALESCE(b."SGTXT", '') AS transaction_text,
        COALESCE(po.po_short_text, '') AS po_short_text,
        COALESCE(po.po_material_group, '') AS po_material_group,
        COALESCE(b."ZUONR", '') AS assignment_number,
        COALESCE(b."KOSTL", '') AS cost_center,
        COALESCE(b."KSTAR", '') AS cost_element,
        COALESCE(b."PRCTR", '') AS profit_center,
        COALESCE(b."GSBER", '') AS business_area,
        COALESCE(b."BUPLA", '') AS business_place,
        COALESCE(b."WERKS", '') AS plant,
        COALESCE(w."NAME1", '') AS plant_name,
        COALESCE(b."EBELN", '') AS po_number,
        COALESCE(b."EBELP", '') AS po_item,
        COALESCE(b."LIFNR", '') AS vendor_id,
        COALESCE(lf.vendor_name, '') AS vendor_name,
        COALESCE(lf.vendor_group, '') AS vendor_group,
        COALESCE(b."KUNNR", '') AS customer_id,
        COALESCE(b."AUFNR", '') AS internal_order,
        COALESCE(b."ANLN1", '') AS asset_number,
        COALESCE(b."H_BLART", '') AS doc_type,
        COALESCE(dt.doc_type_desc, '') AS doc_type_desc,
        b."H_BUDAT" AS posting_date,
        b."H_BLDAT" AS document_date,
        COALESCE(b."H_MONAT", '') AS posting_period,
        COALESCE(b.hsn_sac, '') AS hsn_sac_code,
        COALESCE(b.plc_sup, '') AS place_of_supply,
        COALESCE(b."ZTERM", '') AS payment_terms,
        COALESCE(b."ZLSCH", '') AS payment_method,
        COALESCE(b."AUGBL", '') AS clearing_doc
    FROM bseg_pool b
    LEFT JOIN ska1_dedup a ON b.gl_account = a."SAKNR"
    LEFT JOIN skat_dedup t ON b.gl_account = t."SAKNR"
    LEFT JOIN skb1_dedup kb ON b."BUKRS" = kb."BUKRS" AND b.gl_account = kb."SAKNR"
    LEFT JOIN ska1_dedup ga ON b."GKONT" = ga."SAKNR"
    LEFT JOIN skat_dedup gt ON b."GKONT" = gt."SAKNR"
    LEFT JOIN tbsl_dedup pk ON b."BSCHL" = pk."BSCHL"
    LEFT JOIN t003t_dedup dt ON b."H_BLART" = dt."BLART"
    LEFT JOIN lfa1_dedup lf ON b."LIFNR" = lf."LIFNR"
    LEFT JOIN ekpo_dedup po ON b."EBELN" = po."EBELN" AND b."EBELP" = po."EBELP"
    LEFT JOIN "{schema}"."T001W" w ON b."WERKS" = w."WERKS"
    LEFT JOIN "{schema}"."T001" comp ON b."BUKRS" = comp."BUKRS";
    """
    return query


def extract_account_auto_coding_data(limit=None, schemas=None, config=None) -> pd.DataFrame:
    """
    Extracts SAP Accounting Document Line Items, Transaction Texts, G/L Account Master Data,
    G/L Account Descriptions, Posting Keys, Document Types, PO Short Texts, and Vendor Metadata
    from PostgreSQL for all target schemas:
    - If schemas is None, reads schemas list from db_config.yaml (default: ["HANA_P2", "HANA_P3"]).
    - If limit is None or <= 0, extracts 100% of all valid transaction records with zero row limit.
    - Joins BSEG, SKA1, SKAT, SKB1, TBSL, T003T, EKPO, LFA1, T001W, and T001.
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
    print(f"[INFO] Initializing SAP FI Transaction extraction across schemas {schemas} with {limit_desc}...")

    for schema in schemas:
        print(f"[INFO] Querying schema '{schema}'...")
        schema_limit = limit
        if schema.upper() == "HANA_P3" and (limit is None or limit <= 0):
            current_count = sum(len(d) for d in dfs)
            # Allocate up to the maximum capacity of Excel (1,048,500 total rows)
            schema_limit = max(100000, 1048500 - current_count)
            print(f"[INFO] Schema '{schema}' contains 74,118,456 rows. To prevent RAM OOM exhaustion and preserve Excel 1,048,576 row specification limits, extracting {schema_limit:,} rows from '{schema}' to completely maximize dataset size.")

        sql = build_schema_query(schema, limit=schema_limit)
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
extract_gl_coding_data = extract_account_auto_coding_data
extract_financial_transaction_data = extract_account_auto_coding_data
