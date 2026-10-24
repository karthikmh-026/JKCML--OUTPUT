"""
Preprocessing & NLP Feature Engineering Module - Use Case 25: Account Auto Coding (NLP)
Transforms raw SAP Accounting Document Segments (BSEG), G/L Account Master (SKA1, SKAT, SKB1),
Posting Keys (TBSL), Document Types (T003T), and Purchase Order Short Texts (EKPO) into
32 domain-engineered numerical NLP & Chart-of-Accounts features to predict G/L accounts from
transaction text and detect miscoded or ambiguous accounting line items.
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# NLP Lexical Taxonomies for SAP Financial Accounting & G/L Auto-Coding (J.K. Cement Chart of Accounts)
REGEX_FREIGHT_LOGISTICS = re.compile(
    r'\b(FREIGHT|FRT|TRANSPORT|TRANSPORTER|TRANSPORTATION|LOADING|UNLOADING|HANDLING|FORWARDING|'
    r'WHARFAGE|WHRFG|YARD|RAKE|RAIL|RAILWAY|DUMP|DISPATCH|CARTAGE|TANKER|FLEET|TRANSIT|DELIVERY|'
    r'DEMURRAGE|TOLL|LOGISTIC|TRUCK|TRAILER|LORRY|CARRIER|CONSIGNMENT|UNLOAD|LOAD\s*CHG)\b',
    re.IGNORECASE
)

REGEX_PLANT_MAINT_RENT = re.compile(
    r'\b(CABLE|CRANE|HIRING|REPAIR|REPAIRING|MAINT|MAINTENANCE|MAINTAINCE|GODOWN|RENT|LEASE|'
    r'SUBLETTING|WATER|CANTEEN|WELDER|WELDING|TESTING|ULTRASONIC|SPARE|SPARES|MACHINERY|PLANT|'
    r'PACKING|POLYTHENE|ELECTRIC|POWER|FUEL|DIESEL|COAL|SECURITY|HOUSEKEEPING|UPKEEPING|KEEPER|'
    r'FAN|KETTLE|WASHER|COMPRESSOR|KILN|MILL|CRUSHER|CIVIL|FABRICATION|ERECTION|MANPOWER|CONTRACTOR)\b',
    re.IGNORECASE
)

REGEX_SALES_PROMO_COMM = re.compile(
    r'\b(ADV|ADVERTISEMENT|PUBLICITY|BRANDING|PROMOTION|PROMOTIONAL|INCENTIVE|COMMISSION|COMM|'
    r'DEALER|DISCOUNT|REBATE|NEWSPAPER|HOARDING|BANNER|BOARD|EVENT|CAMPAIGN|GIFT|SPEAKER|'
    r'MARKETING|SCHEME|LIASIONING|LIAISONING|TPC|AGENCY|BROCHURE|DISPLAY|EXHIBITION|MEET)\b',
    re.IGNORECASE
)

REGEX_TRAVEL_IT_PROF = re.compile(
    r'\b(TAXI|CAB|TRAVEL|TRAVELLING|TOUR|HOTEL|STAY|LODGING|BOARDING|INTERVIEW|FLIGHT|AIR\s*TICKET|'
    r'TRAIN|VISA|PASSPORT|PHOTO|MEAL|FOOD|DINNER|LUNCH|GUEST\s*HOUSE|MEETING|VIP|CONSULTANCY|'
    r'CONSULTANT|LEGAL|AUDIT|AUDITOR|PROFESSIONAL|SURVEY|FEE|FEES|NETWORK|SOFTWARE|LICENSE|'
    r'BROADBAND|INTERNET|VODAFONE|AIRTEL|JIO|BSNL|MOBILE|PHONE|TELEPHONE|ZOOM|IT\s*SERVICE|'
    r'PF|SALARY|STIPEND|EMPLOYEE|STAFF|YOGA|MEDICAL|WELFARE|TRAINING|RECRUITMENT)\b',
    re.IGNORECASE
)

REGEX_TAX_BANK_PROV = re.compile(
    r'\b(TDS|GST|CGST|SGST|IGST|UGST|ITC|TAX|WITHHOLDING|CESS|DUTY|CUSTOM|ROYALTY|BANK|LC|'
    r'BG|GUARANTEE|SAFTA|CONFIRM|CERTIFICATE|CHARGES|PROVISION|LIABILITY|OUTSTANDING|ADJUSTMENT|'
    r'ADJ|REIMBURSEMENT|REIMB|ADVANCE|CLEARING|SETTLEMENT|CAPEX|ASSET|LAND|MINES|PURCHASED)\b',
    re.IGNORECASE
)

REGEX_ACCOUNTING_REF_TOKENS = re.compile(
    r'(?:\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)[\'\-\s/]*\d{2,4}\b|'
    r'\b(?:PO|OL|BILL|INV|CHALLAN|DOC|REF|VOUCHER|M/O|FTM|MT|TON|KG|KM|HRS|NOS|QTY)[\.\-\s:/]*[A-Z0-9]+\b|'
    r'\b202[0-9]\b|\b\d{2}[\./\-]\d{2}[\./\-]\d{2,4}\b)',
    re.IGNORECASE
)


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw SAP Accounting Document Line Items (BSEG) and G/L Account Master data (SKA1, SKAT, SKB1),
    predicts recommended G/L Accounts and Accounting Domains from transaction text using NLP, and
    engineers 32 domain-specific features for Unsupervised Account Auto Coding.
    """
    import gc

    # 1. Clean string columns across BSEG, SKA1, SKAT, SKB1, TBSL, T003T, EKPO, LFA1
    str_cols = [
        'company_code', 'company_name', 'company_city', 'company_currency',
        'doc_number', 'fiscal_year', 'line_item', 'gl_account', 'gl_account_desc',
        'gl_short_text', 'gl_account_group', 'gl_account_type', 'is_balance_sheet_acct',
        'pl_statement_type', 'functional_area', 'recon_account_type', 'field_status_group',
        'open_item_mgmt_flag', 'auto_posting_only_flag', 'offsetting_gl_account',
        'offsetting_gl_desc', 'offsetting_account_group', 'offsetting_account_type',
        'account_type', 'posting_key', 'posting_key_desc', 'debit_credit_ind',
        'doc_currency', 'transaction_text', 'po_short_text', 'po_material_group',
        'assignment_number', 'cost_center', 'cost_element', 'profit_center',
        'business_area', 'business_place', 'plant', 'plant_name', 'po_number',
        'po_item', 'vendor_id', 'vendor_name', 'vendor_group', 'customer_id',
        'internal_order', 'asset_number', 'doc_type', 'doc_type_desc',
        'posting_period', 'hsn_sac_code', 'place_of_supply', 'payment_terms',
        'payment_method', 'clearing_doc'
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna('')
                .astype(str)
                .str.replace('\ufffd', ' ', regex=False)
                .str.replace(r'\s+', ' ', regex=True)
                .str.strip()
            )
        else:
            df[col] = ''

    # 2. Clean numeric amount columns (handling comma-formatted numbers in SAP BSEG DMBTR/WRBTR)
    for amt_col in ['amount_local', 'amount_doc_curr']:
        if amt_col in df.columns:
            cleaned_amt = (
                df[amt_col]
                .astype(str)
                .str.replace(',', '', regex=False)
                .str.strip()
            )
            df[amt_col] = pd.to_numeric(cleaned_amt, errors='coerce').fillna(0.0).abs()
        else:
            df[amt_col] = 0.0

    # 3. Construct Consolidated NLP Transaction Text Corpus from BSEG-SGTXT, EKPO-TXZ01, and Vendor Context
    primary_text = np.where(df['transaction_text'] != '', df['transaction_text'], df['po_short_text'])
    df['transaction_text'] = pd.Series(primary_text, index=df.index).fillna('').astype(str).str.strip()

    combined_corpus = (
        df['transaction_text'] + " " +
        np.where(df['po_short_text'] != df['transaction_text'], df['po_short_text'], "") + " " +
        df['vendor_name']
    ).str.strip()

    # 4. Lexical & Structural NLP Features from Transaction Text (BSEG-SGTXT)
    df['sgtxt_char_length'] = df['transaction_text'].str.len().clip(0, 120).astype(float)
    tokens_series = df['transaction_text'].str.findall(r'[A-Za-z0-9]+')
    df['sgtxt_word_count'] = tokens_series.str.len().fillna(0).clip(0, 30).astype(float)

    alpha_chars = df['transaction_text'].str.count(r'[A-Za-z]').fillna(0).astype(float)
    digit_chars = df['transaction_text'].str.count(r'[0-9]').fillna(0).astype(float)
    safe_len = np.maximum(1.0, df['sgtxt_char_length'])

    df['sgtxt_avg_word_len'] = np.where(
        df['sgtxt_word_count'] > 0,
        (alpha_chars + digit_chars) / df['sgtxt_word_count'],
        0.0
    ).clip(0.0, 25.0)
    df['sgtxt_digit_ratio'] = (digit_chars / safe_len).clip(0.0, 1.0)
    df['sgtxt_alpha_ratio'] = (alpha_chars / safe_len).clip(0.0, 1.0)

    # Count structured accounting reference & period tokens in SGTXT
    ref_matches = df['transaction_text'].str.findall(REGEX_ACCOUNTING_REF_TOKENS)
    df['accounting_ref_token_count'] = ref_matches.str.len().fillna(0).clip(0, 10).astype(float)

    # Duplicate SGTXT mapped to multiple different G/L Accounts (HKONT)
    norm_sgtxt = df['transaction_text'].str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
    df['_norm_sgtxt'] = np.where(norm_sgtxt == '', 'EMPTY_SGTXT', norm_sgtxt)
    df['dup_sgtxt_count'] = df.groupby('_norm_sgtxt')['doc_number'].transform('count').astype(float)
    df['dup_sgtxt_diff_gl_count'] = df.groupby('_norm_sgtxt')['gl_account'].transform('nunique').astype(float)

    # 5. TF-IDF Vectorization & TruncatedSVD (Latent Semantic Analysis) on Transaction Texts
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=350,
        min_df=2,
        sublinear_tf=True,
        token_pattern=r'(?u)\b[A-Za-z0-9][A-Za-z0-9\-\./]{1,}\b'
    )
    corpus_list = combined_corpus.replace('', 'UNKNOWN_TRANSACTION').tolist()
    tfidf_matrix = tfidf.fit_transform(corpus_list)

    nnz_per_row = np.diff(tfidf_matrix.indptr).astype(float)
    l1_energy = np.asarray(tfidf_matrix.sum(axis=1)).ravel().astype(float)
    df['tfidf_active_ngrams'] = np.clip(nnz_per_row, 0.0, 35.0)
    df['tfidf_semantic_energy'] = np.round(np.clip(l1_energy, 0.0, 10.0), 4)

    n_svd = min(5, max(1, tfidf_matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_svd, random_state=42)
    svd_matrix = svd.fit_transform(tfidf_matrix)
    for i in range(5):
        col_name = f'svd_semantic_comp_{i+1}'
        if i < svd_matrix.shape[1]:
            df[col_name] = np.round(svd_matrix[:, i], 5)
        else:
            df[col_name] = 0.0

    # 6. Unsupervised NLP G/L Account Domain Affinity Scores [0-100]
    frt_hits = combined_corpus.str.findall(REGEX_FREIGHT_LOGISTICS).str.len().fillna(0).astype(float)
    plt_hits = combined_corpus.str.findall(REGEX_PLANT_MAINT_RENT).str.len().fillna(0).astype(float)
    sal_hits = combined_corpus.str.findall(REGEX_SALES_PROMO_COMM).str.len().fillna(0).astype(float)
    trv_hits = combined_corpus.str.findall(REGEX_TRAVEL_IT_PROF).str.len().fillna(0).astype(float)
    tax_hits = combined_corpus.str.findall(REGEX_TAX_BANK_PROV).str.len().fillna(0).astype(float)

    del corpus_list, tfidf_matrix, svd_matrix, combined_corpus
    gc.collect()

    # Contextual Chart-of-Accounts structural cues from G/L Account Group (KTOKS) & Account range
    gl_And_off_desc = (df['gl_account_desc'] + " " + df['offsetting_gl_desc']).str.upper()
    cue_frt = gl_And_off_desc.str.contains(r'FREIGHT|TRANSPORT|HANDLING|LOADING|WHARFAGE', regex=True, na=False).astype(float)
    cue_plt = gl_And_off_desc.str.contains(r'RENT|GODOWN|MAINTENANCE|REPAIR|SERVICE PROVIDER|MANPOWER|SPARE|FUEL|WATER', regex=True, na=False).astype(float)
    cue_sal = gl_And_off_desc.str.contains(r'ADVERTISEMENT|PROMOTION|COMMISSION|INCENTIVE|BRANDING|PUBLICITY', regex=True, na=False).astype(float)
    cue_trv = gl_And_off_desc.str.contains(r'TRAVELLING|VEHICLE|HOTEL|GUEST HOUSE|MEETING|PHONE|MOBILE|IT SERVICE|PROFESSIONAL|LEGAL', regex=True, na=False).astype(float)
    cue_tax = (
        df['gl_account_group'].isin({'LOAD', 'PROV', 'CABK', 'FXAS'}) |
        gl_And_off_desc.str.contains(r'TDS|ITC|CGST|SGST|IGST|BANK|LIABILITY|ASSET|LAND', regex=True, na=False)
    ).astype(float)

    df['nlp_freight_logistics_affinity'] = np.clip(frt_hits * 36.0 + cue_frt * 24.0 + df['accounting_ref_token_count'] * 4.0, 0.0, 100.0)
    df['nlp_plant_maint_rent_affinity'] = np.clip(plt_hits * 35.0 + cue_plt * 22.0 + df['accounting_ref_token_count'] * 4.0, 0.0, 100.0)
    df['nlp_sales_promo_comm_affinity'] = np.clip(sal_hits * 38.0 + cue_sal * 24.0 + df['accounting_ref_token_count'] * 4.0, 0.0, 100.0)
    df['nlp_travel_it_prof_affinity'] = np.clip(trv_hits * 38.0 + cue_trv * 24.0 + df['accounting_ref_token_count'] * 4.0, 0.0, 100.0)
    df['nlp_tax_bank_prov_affinity'] = np.clip(tax_hits * 34.0 + cue_tax * 22.0 + df['accounting_ref_token_count'] * 4.0, 0.0, 100.0)

    affinity_stack = np.vstack([
        df['nlp_freight_logistics_affinity'].values,
        df['nlp_plant_maint_rent_affinity'].values,
        df['nlp_sales_promo_comm_affinity'].values,
        df['nlp_travel_it_prof_affinity'].values,
        df['nlp_tax_bank_prov_affinity'].values
    ]).T  # shape (N, 5)

    sorted_aff = np.sort(affinity_stack, axis=1)
    top1_aff = sorted_aff[:, -1]
    top2_aff = sorted_aff[:, -2]
    dominant_domain_idx = np.argmax(affinity_stack, axis=1)

    domain_labels = np.array([
        'Freight, Logistics & Material Handling G/L',
        'Plant Operations, Maintenance, Rent & Facility G/L',
        'Sales Promotion, Advertising & Commission G/L',
        'Travel, IT, Telecom & Professional Services G/L',
        'Statutory Taxes (GST/TDS), Banking & Provisions G/L'
    ])
    df['predicted_gl_domain'] = domain_labels[dominant_domain_idx]

    # 7. Predict Specific G/L Account (HKONT/SAKNR) & Description from Transaction Text
    # Canonical SAP J.K. Cement G/L Account mapping based on granular NLP transaction text patterns
    sgtxt_upper = df['transaction_text'].str.upper()

    pred_gl_conds = [
        # Statutory Tax & Input Tax Credit (LOAD / TDS)
        df['gl_account'].isin({'169234', '169235', '169236', '171323', '171325', '171327'}),
        # Specific NLP Text -> Expense / Liability G/L Account Rules
        sgtxt_upper.str.contains(r'\b(?:UNLOADING|LOADING|HANDLING|WHARFAGE|WHRFG|FORWARDING)\b', regex=True, na=False),
        sgtxt_upper.str.contains(r'\b(?:RAIL|RAKE)\b', regex=True, na=False),
        sgtxt_upper.str.contains(r'\b(?:DUMP|FACTORY\s*TO\s*DUMP|STL)\b', regex=True, na=False),
        sgtxt_upper.str.contains(r'\b(?:FREIGHT|FRT|TRANSPORT|TRANSPORTER|CARTAGE|DISPATCH)\b', regex=True, na=False),
        sgtxt_upper.str.contains(r'\b(?:RENT|LEASE|SUBLETTING|G/R)\b', regex=True, na=False),
        sgtxt_upper.str.contains(r'\b(?:GODOWN|KEEPER|YELLOW\s*LINING|WAREHOUSE)\b', regex=True, na=False),
        sgtxt_upper.str.contains(r'\b(?:NEWSPAPER|ADVERTISEMENT|PUBLICITY|HOARDING|BRANDING|ADV\s+CHARGES)\b', regex=True, na=False),
        sgtxt_upper.str.contains(r'\b(?:COMMISSION|COMM|INCENTIVE|TPC)\b', regex=True, na=False),
        sgtxt_upper.str.contains(r'\b(?:PROMOTION|SPEAKER|EVENT|GIFT|VISIT)\b', regex=True, na=False),
        sgtxt_upper.str.contains(r'\b(?:TAXI|CAB|TRAVEL|TRAVELLING|FLIGHT|TOUR)\b', regex=True, na=False),
        sgtxt_upper.str.contains(r'\b(?:INTERVIEW|RECRUITMENT)\b', regex=True, na=False),
        sgtxt_upper.str.contains(r'\b(?:GUEST\s*HOUSE|FLAT|FLATE|MAINTAINCE\s+AT\s+EG)\b', regex=True, na=False),
        sgtxt_upper.str.contains(r'\b(?:MEETING|VIP|MEAL|DINNER|LUNCH)\b', regex=True, na=False),
        sgtxt_upper.str.contains(r'\b(?:VODAFONE|AIRTEL|JIO|MOBILE|PHONE|BROADBAND|INTERNET)\b', regex=True, na=False),
        sgtxt_upper.str.contains(r'\b(?:NETWORK|SOFTWARE|ZOOM|IT\s*SERVICE|LAPTOP|COMPUTER)\b', regex=True, na=False),
        sgtxt_upper.str.contains(r'\b(?:CONSULTANCY|LEGAL|AUDIT|PROFESSIONAL|SURVEY)\b', regex=True, na=False),
        sgtxt_upper.str.contains(r'\b(?:BANK|LC\s*CHARGE|SAFTA|GUARANTEE|CONFIRM-CERTIFICATE)\b', regex=True, na=False),
        sgtxt_upper.str.contains(r'\b(?:CABLE|CRANE|HIRING|WELDER|TESTING|CANTEEN|WATER|FAN|KETTLE)\b', regex=True, na=False),
    ]
    pred_gl_choices = [
        df['gl_account'],  # Keep verified statutory tax/TDS code when already assigned to tax GL
        '332308',          # Loading/Unloading & Forwarding Charges
        '332301',          # Freight & Handling Outward-Rail
        '332303',          # Freight Paid Factory To Dump
        '332302',          # Freight Paid F.O.R
        '330001',          # Rent for godown
        '332109',          # Godown Maintenance Expenses
        '332101',          # Advertisement & Publicity
        '332205',          # Sales Promotion Incentive / Commission
        '332103',          # Sales Promotion Expenses
        '331001',          # Travelling Expenses
        '331403',          # Recruitment / Interview Hotel Stay Expenses
        '331705',          # Guest House Expenses
        '331704',          # Corporate/Management Meeting Expenses
        '330956',          # Mobile/cellular Phone Expenses
        '171116',          # Vendors-IT services
        '171114',          # Vendors-legal & Professional fees
        '331601',          # Bank charges / LC Charges
        '171006',          # Service Providers (Taxable) / Plant Operations
    ]

    domain_default_gl = pd.Series(dominant_domain_idx, index=df.index).map({
        0: '332302',
        1: '171006',
        2: '332103',
        3: '331001',
        4: '173101'
    })
    raw_pred_gl = np.select(pred_gl_conds, pred_gl_choices, default=domain_default_gl)

    # If the current GL account is already a specific operational/expense account (not generic provision 173101/171001)
    # and has no domain conflict, preserve or align with predicted GL
    gl_desc_lookup = (
        df[df['gl_account_desc'] != '']
        .groupby('gl_account')['gl_account_desc']
        .first()
        .to_dict()
    )
    canonical_gl_desc = {
        '332308': 'Loading/Unloading & Forwarding Charges',
        '332301': 'Freight & Handling Outward-Rail',
        '332303': 'Freight Paid Factory To Dump',
        '332302': 'Freight Paid F.O.R',
        '330001': 'Rent for godown',
        '332109': 'Godown Maintenance Expenses',
        '332101': 'Advertisement & Publicity',
        '332205': 'Sales Promotion Incentive',
        '332103': 'Sales Promotion Expenses',
        '331001': 'Travelling Expenses',
        '331403': 'Recruitment & Interview Expenses',
        '331705': 'Guest House Expenses',
        '331704': 'Corporate/Management Meeting Expenses',
        '330956': 'Mobile/cellular Phone Expenses',
        '171116': 'Vendors-IT services',
        '171114': 'Vendors-legal & Professional fees',
        '331601': 'Bank charges',
        '171006': 'Service Providers (Taxable)',
        '173101': 'Outstanding liability for expenses',
        '169234': 'ITC-SGST',
        '169235': 'ITC-CGST',
        '169236': 'ITC-IGST',
        '171323': 'TDS - Contractors',
        '171325': 'TDS - Rent',
        '171327': 'TDS - Professionals',
    }
    gl_desc_lookup.update(canonical_gl_desc)

    df['predicted_gl_account'] = pd.Series(raw_pred_gl, index=df.index).astype(str)
    df['predicted_gl_account_desc'] = df['predicted_gl_account'].map(
        lambda x: gl_desc_lookup.get(x, 'Standard Chart of Accounts G/L')
    )

    # 8. Lexical & Semantic Similarity between Transaction Text (SGTXT) and Assigned GL Description (SKAT-TXT50)
    total_keyword_hits = frt_hits + plt_hits + sal_hits + trv_hits + tax_hits
    same_gl_or_offset = (
        (df['gl_account'] == df['predicted_gl_account']) |
        (df['offsetting_gl_account'] == df['predicted_gl_account']) |
        ((cue_frt > 0) & (frt_hits > 0)) |
        ((cue_plt > 0) & (plt_hits > 0)) |
        ((cue_sal > 0) & (sal_hits > 0)) |
        ((cue_trv > 0) & (trv_hits > 0)) |
        ((cue_tax > 0) & (tax_hits > 0))
    )
    df['text_gl_desc_similarity_score'] = np.where(
        same_gl_or_offset,
        np.clip(78.0 + 6.0 * np.minimum(total_keyword_hits, 3.0), 0.0, 100.0),
        np.where(total_keyword_hits > 0, 48.0, 32.0)
    )

    # 9. Financial Transaction Amount & Peer GL Z-Score Features
    df['log_amount_local'] = np.log1p(np.maximum(0.0, df['amount_local']))
    gl_mean = df.groupby('gl_account')['log_amount_local'].transform('mean')
    gl_std = df.groupby('gl_account')['log_amount_local'].transform('std').fillna(1.0).replace(0, 1.0)
    df['gl_account_amount_zscore'] = ((df['log_amount_local'] - gl_mean) / gl_std).clip(-6.0, 6.0).fillna(0.0)

    # 10. Chart of Accounts Governance & Miscoding Signal Flags
    # In SAP FI Extended Withholding Tax, 2-char tax codes (FE, EE, JE, HE, JA) on TDS GLs (171323, 171325, 171327) are valid auto-generated tax lines
    is_valid_auto_tds_code = (
        df['gl_account'].isin({'171323', '171325', '171327'}) &
        df['transaction_text'].str.fullmatch(r'[A-Z]{2}', na=False)
    )

    # Boost statutory tax affinity for valid auto-generated SAP withholding tax codes
    df['nlp_tax_bank_prov_affinity'] = np.where(
        is_valid_auto_tds_code,
        np.maximum(df['nlp_tax_bank_prov_affinity'], 82.0),
        df['nlp_tax_bank_prov_affinity']
    )

    # Flag 1: Cryptic or code-only SGTXT on non-auto-TDS accounts (e.g., '18', '04', '55', '02', '1')
    df['is_cryptic_or_code_only_sgtxt'] = (
        (~is_valid_auto_tds_code) & (
            (df['sgtxt_char_length'] <= 5) |
            (df['sgtxt_word_count'] <= 1) |
            (df['sgtxt_alpha_ratio'] < 0.25)
        )
    ).astype(int)

    # Flag 2: Specific operational expense text parked in generic Provision / Sundry Vendor account (173101 / 171001)
    has_specific_expense_text = ((frt_hits + sal_hits + trv_hits) >= 1)
    df['is_generic_provision_or_clearing_gl'] = (
        (df['gl_account'].isin({'173101', '173102', '171001', '149955'})) &
        (has_specific_expense_text | (df['cost_center'] == ''))
    ).astype(int)

    # Flag 3: Missing Cost Object (KOSTL / AUFNR) on P&L Expense Account (GLACCOUNT_TYPE == 'P' or SA&O / MEXP)
    is_pl_acct = (
        (df['gl_account_type'] == 'P') |
        (df['gl_account_group'].isin({'SA&O', 'MEXP', 'PTPE', 'SALE', 'SECO'}))
    )
    df['is_missing_cost_object_on_pl'] = (
        is_pl_acct &
        (df['cost_center'] == '') &
        (df['internal_order'] == '') &
        (df['profit_center'] == '')
    ).astype(int)

    # Flag 4: NLP Semantic Domain Conflict between Transaction Text (SGTXT) and Posted GL Account
    df['is_nlp_gl_domain_conflict'] = (
        ((frt_hits >= 1) & (df['gl_account'].isin({'173101', '332101', '332103', '331001'})) & (~sgtxt_upper.str.contains('TRAVEL|TAXI|PROMO', regex=True, na=False))) |
        ((trv_hits >= 1) & (df['gl_account'].isin({'173101', '332302', '332303', '171091'}))) |
        ((sal_hits >= 1) & (df['gl_account'].isin({'149955', '173101', '332302'}))) |
        ((df['dup_sgtxt_diff_gl_count'] >= 3) & (df['is_cryptic_or_code_only_sgtxt'] == 0) & (~df['gl_account'].isin({'169234', '169235', '169236', '171323', '171325', '171327'})))
    ).astype(int)

    # 11. Three Component Scores [0-100] & Composite NLP Account Auto-Coding Confidence / Risk Scores
    # (A) Transaction Text Clarity & Completeness Score [0-100]
    len_score = np.where(is_valid_auto_tds_code, 78.0, np.clip((df['sgtxt_char_length'] / 32.0) * 100.0, 0.0, 100.0))
    word_score = np.where(is_valid_auto_tds_code, 76.0, np.clip((df['sgtxt_word_count'] / 5.0) * 100.0, 0.0, 100.0))
    ref_bonus = np.clip(df['accounting_ref_token_count'] * 12.0, 0.0, 24.0)
    tfidf_bonus = np.clip(df['tfidf_active_ngrams'] * 4.5, 0.0, 25.0)

    raw_text_clarity = (
        0.38 * len_score +
        0.34 * word_score +
        ref_bonus +
        tfidf_bonus -
        42.0 * df['is_cryptic_or_code_only_sgtxt']
    )
    df['text_clarity_score'] = np.round(np.clip(raw_text_clarity, 0.0, 100.0), 2)

    # (B) NLP Semantic G/L Prediction Confidence Score [0-100]
    eff_top1 = np.where(is_valid_auto_tds_code, np.maximum(top1_aff, 82.0), top1_aff)
    margin_signal = np.clip(eff_top1 - 0.35 * top2_aff, 0.0, 100.0)
    raw_semantic_conf = (
        24.0 +
        0.50 * eff_top1 +
        0.26 * margin_signal +
        np.clip(df['tfidf_semantic_energy'] * 10.0, 0.0, 22.0) -
        24.0 * ((total_keyword_hits == 0) & (~is_valid_auto_tds_code)).astype(float) -
        22.0 * df['is_cryptic_or_code_only_sgtxt']
    )
    df['semantic_gl_confidence_score'] = np.round(np.clip(raw_semantic_conf, 0.0, 100.0), 2)

    # (C) Chart of Accounts (CoA) & Cost Object Alignment Score [0-100]
    raw_coa_align = (
        0.74 * df['text_gl_desc_similarity_score'] +
        26.0 * (1.0 - df['is_generic_provision_or_clearing_gl']) -
        34.0 * df['is_nlp_gl_domain_conflict'] -
        24.0 * df['is_missing_cost_object_on_pl'] -
        18.0 * ((df['dup_sgtxt_diff_gl_count'] >= 3) & (df['is_cryptic_or_code_only_sgtxt'] == 0) & (~is_valid_auto_tds_code)).astype(float)
    )
    df['coa_alignment_score'] = np.round(np.clip(raw_coa_align, 0.0, 100.0), 2)

    # Composite NLP Account Auto-Coding Confidence Score [0-100] & Miscoding Risk Index [0-100]
    raw_conf = (
        0.34 * df['text_clarity_score'] +
        0.36 * df['semantic_gl_confidence_score'] +
        0.30 * df['coa_alignment_score']
    )
    df['nlp_coding_confidence_score'] = np.round(np.clip(raw_conf, 0.0, 100.0), 2)
    df['nlp_miscoding_risk_index'] = np.round(100.0 - df['nlp_coding_confidence_score'], 2)

    # Discrete 1-5 Quintile Ratings (T-S-C: Text Clarity, Semantic GL Confidence, CoA Alignment)
    df['t_rating'] = np.clip(np.ceil(df['text_clarity_score'] / 20.0), 1, 5).astype(int)
    df['s_rating'] = np.clip(np.ceil(df['semantic_gl_confidence_score'] / 20.0), 1, 5).astype(int)
    df['c_rating'] = np.clip(np.ceil(df['coa_alignment_score'] / 20.0), 1, 5).astype(int)
    df['coding_tier_code'] = (
        "T" + df['t_rating'].astype(str) +
        "-S" + df['s_rating'].astype(str) +
        "-C" + df['c_rating'].astype(str)
    )

    # 12. Post-Hoc Benchmark Account Auto-Coding Segment & Exception Flag (Y)
    # Strictly excluded from unsupervised clustering training; used ONLY for post-hoc evaluation & ranking of the 5 models
    p22 = float(np.percentile(df['nlp_coding_confidence_score'], 22.0))
    p52 = float(np.percentile(df['nlp_coding_confidence_score'], 52.0))
    p78 = float(np.percentile(df['nlp_coding_confidence_score'], 78.0))

    seg_conditions = [
        (df['nlp_coding_confidence_score'] >= p78),
        (df['nlp_coding_confidence_score'] >= p52),
        (df['nlp_coding_confidence_score'] >= p22),
    ]
    seg_choices = [
        'High-Confidence Auto-Coded G/L Postings',
        'Verified Expense & Statutory G/L Postings',
        'Moderate-Confidence / Needs Narration Enrichment',
    ]
    df['benchmark_coding_segment'] = np.select(
        seg_conditions,
        seg_choices,
        default='Miscoded / Ambiguous G/L Coding Exceptions'
    )

    risk_cutoff = round(100.0 - p22, 2)
    df['is_coding_exception'] = (
        (df['nlp_miscoding_risk_index'] >= risk_cutoff) |
        ((df['nlp_miscoding_risk_index'] >= (risk_cutoff - 6.5)) & (
            (df['is_nlp_gl_domain_conflict'] == 1) |
            (df['is_generic_provision_or_clearing_gl'] == 1) |
            (df['is_cryptic_or_code_only_sgtxt'] == 1)
        ))
    ).astype(int)

    df.drop(columns=['_norm_sgtxt'], inplace=True, errors='ignore')
    return df


def get_feature_matrix(df: pd.DataFrame):
    """
    Extracts 32 numerical NLP & Financial Accounting features for Unsupervised Clustering.
    """
    feature_cols = [
        'nlp_coding_confidence_score',
        'nlp_miscoding_risk_index',
        'text_clarity_score',
        'semantic_gl_confidence_score',
        'coa_alignment_score',
        'nlp_freight_logistics_affinity',
        'nlp_plant_maint_rent_affinity',
        'nlp_sales_promo_comm_affinity',
        'nlp_travel_it_prof_affinity',
        'nlp_tax_bank_prov_affinity',
        'sgtxt_char_length',
        'sgtxt_word_count',
        'sgtxt_avg_word_len',
        'sgtxt_digit_ratio',
        'sgtxt_alpha_ratio',
        'accounting_ref_token_count',
        'tfidf_active_ngrams',
        'tfidf_semantic_energy',
        'svd_semantic_comp_1',
        'svd_semantic_comp_2',
        'svd_semantic_comp_3',
        'svd_semantic_comp_4',
        'svd_semantic_comp_5',
        'text_gl_desc_similarity_score',
        'dup_sgtxt_count',
        'dup_sgtxt_diff_gl_count',
        'is_cryptic_or_code_only_sgtxt',
        'is_generic_provision_or_clearing_gl',
        'is_missing_cost_object_on_pl',
        'is_nlp_gl_domain_conflict',
        'log_amount_local',
        'gl_account_amount_zscore'
    ]

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df['is_coding_exception'].values
    return X, y, feature_cols


def prepare_train_test_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42, max_eval_samples: int = 100000):
    """
    Extracts 32 NLP & Chart-of-Accounts features, performs stratified train/test split for post-hoc evaluation,
    and applies RobustScaler (fit strictly on X_train without labels).
    If dataset exceeds max_eval_samples, takes a representative stratified sample for model training & evaluation
    so all 5 unsupervised models benchmark rapidly without OOM.
    """
    if max_eval_samples is not None and len(df) > max_eval_samples:
        stratify_col = df['is_coding_exception'] if 'is_coding_exception' in df.columns else None
        df_eval, _ = train_test_split(
            df,
            train_size=max_eval_samples,
            random_state=random_state,
            stratify=stratify_col
        )
    else:
        df_eval = df

    X, y, feature_cols = get_feature_matrix(df_eval)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_eval.index, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'idx_train': idx_train,
        'idx_test': idx_test,
        'scaler': scaler,
        'feature_cols': feature_cols
    }
