"""
Preprocessing & NLP Feature Engineering Module - Use Case 24: Material Auto Classification (NLP)
Transforms raw SAP Material Master (MARA), Material Descriptions (MAKT), Plant/MRP (MARC),
Valuation (MBEW), and Purchase Order Short Texts (EKPO) into 32 domain-engineered numerical
NLP & Taxonomy features for Unsupervised Material Auto Classification.
Handles missing values comprehensively across all data types.
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# NLP Lexical Taxonomies for SAP Industrial & Building Materials (J.K. Cement / Paints / MRO)
REGEX_MECH_SPARES = re.compile(
    r'\b(BEARING|SEAL|RING|SHAFT|GEAR|GEARBOX|PUMP|VALVE|IMPELLER|LINER|ROLLER|KILN|CRUSHER|MILL|'
    r'COOLER|CONVEYOR|PULLEY|SPROCKET|BUSH|BUSHING|COUPLING|PISTON|CYLINDER|NOZZLE|FILTER|SPRING|'
    r'DIAPHRAGM|GASKET|CASING|ROTOR|STATOR|SPARE|ASSY|ASSEMBLY|HOUSING|PINION|IDLER|DAMPER|BLOWER|'
    r'FAN|COMPRESSOR|HYDRAULIC|PNEUMATIC|CHUTE|HOPPER|BUCKET|ELEVATOR|SEPARATOR|CYCLONE|BURNER|GRATE)\b',
    re.IGNORECASE
)

REGEX_ELEC_INST = re.compile(
    r'\b(CABLE|WIRE|SWITCH|SENSOR|MOTOR|CONTACTOR|RELAY|BREAKER|MCB|MCCB|ACB|FUSE|PLC|MODULE|'
    r'TRANSMITTER|GAUGE|THERMOCOUPLE|RTD|TRANSFORMER|COIL|CONNECTOR|TERMINAL|LED|LAMP|LIGHT|'
    r'INVERTER|VFD|DRIVE|CONTROLLER|PROXIMITY|SOLENOID|ACTUATOR|ANALYZER|METER|SWITCHGEAR|PANEL|'
    r'CAPACITOR|RESISTOR|THYRISTOR|RECTIFIER|BATTERY|UPS|ENCODER|TRANSDUCER|ISOLATOR)\b',
    re.IGNORECASE
)

REGEX_CONSUMABLE_FASTENER = re.compile(
    r'\b(BOLT|NUT|WASHER|SCREW|STUD|FASTENER|ELECTRODE|WELD|WELDING|PIPE|FITTING|ELBOW|FLANGE|'
    r'NIPPLE|SOCKET|TEE|REDUCER|BELT|VBELT|HOSE|TUBE|GREASE|OIL|LUBRICANT|LUBE|GLOVE|SHOE|HELMET|'
    r'GOGGLE|MASK|HARNESS|SAFETY|TOOL|SPANNER|WRENCH|DRILL|GRINDING|WHEEL|CUTTING|DISC|BRUSH|'
    r'PLATE|MS\s*PLATE|SS\s*PLATE|CHANNEL|ANGLE|FLAT|ROD|BAR|ROPE|CHAIN|CLAMP|PACKING\s*CORD|GLAND)\b',
    re.IGNORECASE
)

REGEX_FINISHED_PAINT_CEMENT = re.compile(
    r'\b(CEMENT|CONCRET|CONCRETE|RMC|M\d{2}|WALL\s*PUTTY|PUTTY|PRIMER|PAINT|EMULSION|ENAMEL|ACRYLIC|'
    r'WOODAMORE|PROFIX|TILEMAXX|GROUT|EGROUT|THINNER|TINT|STAINER|COAT|COATING|GLOSS|SHEEN|WEATHER|'
    r'EXTERIOR|INTERIOR|WHITE\s*CEMENT|OPC|PPC|PSC|DISTEMPER|WATERPROOF|ADHESIVE|RESIN|HARDENER|'
    r'BITU|TEXTURE|SEALANT|VARNISH|LACQUER|UNDERCOAT|ACROCOTE|MICACEO|ENAMELO|SUPER\s*CONCRET)\b',
    re.IGNORECASE
)

REGEX_PACKAGING_RAW = re.compile(
    r'\b(BAG|PAPER\s*BAG|PP\s*BAG|BOPP|HDPE|SACK|BOX|CORRUGATED|CARTON|CRTN|DRUM|BUCKET|PAIL|TIN|'
    r'CAN|POUCH|ROLL|WRAP|STRETCH|PALLET|LABEL|STICKER|TAPE|CONTAINER|BOTTLE|JAR|CAP|LID|STRAPPING|'
    r'GYPSUM|CLINKER|LIMESTONE|LIME|SLAG|FLYASH|COAL|PETCOKE|DIESEL|HSD|FUEL|AFR|WASTE|ACID|ALKALI|'
    r'SOLVENT|PIGMENT|EXTENDER|TITANIUM|CALCIUM|SILICA|DOLOMITE|BAUXITE|IRON\s*ORE|CHEMICAL|ADDITIVE|EXPLOSIVE)\b',
    re.IGNORECASE
)

REGEX_SPEC_TOKENS = re.compile(
    r'(\b\d+(\.\d+)?\s*(MM|CM|M|INCH|IN|FT|KG|GM|G|MG|TO|MT|LTR|L|ML|KW|MW|HP|V|KV|VOLT|A|AMP|RPM|HZ|BAR|PSI|DEG|M3|SQM|PCS|NO|SET)\b|'
    r'\b\d+\s*[Xx*]\s*\d+|\bM\d{1,2}(\.\d)?\b|\bDN\d+\b|\bPN\d+\b|\bNB\d+\b|\bØ\d+)',
    re.IGNORECASE
)


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw SAP Material Master & Description data across MARA, MAKT, MARC, MBEW, and EKPO,
    handles missing values properly (empty strings for missing text, zeros for missing metrics),
    and engineers 32 domain-specific NLP & Taxonomy features to automatically classify materials
    from their descriptions and flag ambiguous or misclassified material records.
    """
    df = df.copy()

    # 1. Clean string attributes: replace NaN, None, "nan", "None", "<NA>" with blank ""
    str_cols = [
        'schema_name', 'company_code', 'company_name', 'company_city', 'company_country', 'company_currency',
        'material_id', 'material_desc', 'material_desc_upper', 'po_short_text',
        'material_type', 'material_type_desc', 'industry_sector', 'material_group',
        'po_material_group', 'old_material_no', 'base_uom', 'order_uom', 'division',
        'prod_hierarchy', 'size_dimensions', 'basic_material_text', 'industry_std_desc',
        'trans_group', 'batch_mgmt_flag', 'cross_plant_status', 'complete_maint_status',
        'mara_maint_status', 'mara_deletion_flag', 'weight_unit', 'volume_unit',
        'created_by', 'changed_by', 'plant', 'plant_name', 'plant_city', 'plant_region',
        'plant_country', 'marc_maint_status', 'marc_deletion_flag', 'plant_mat_status',
        'abc_indicator', 'purchasing_group', 'mrp_type', 'mrp_controller', 'procurement_type',
        'profit_center', 'loading_group', 'valuation_class', 'price_control', 'mbew_deletion_flag'
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .fillna('')
                .astype(str)
                .str.strip()
                .replace({'nan': '', 'None': '', 'None': '', 'NULL': '', '<NA>': '', 'NaT': ''})
            )
        else:
            df[col] = ''

    # 2. Clean numeric attributes: coerce non-numeric to NaN, fillna with 0.0
    num_cols = [
        'distinct_po_matkl_cnt', 'gross_weight', 'net_weight', 'volume',
        'planned_deliv_days', 'gr_processing_days', 'moving_avg_price', 'standard_price',
        'price_unit', 'total_valuated_stock', 'total_stock_value', 'po_line_count',
        'total_po_spend', 'avg_po_net_price', 'alt_uom_count', 'info_record_count'
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        else:
            df[col] = 0.0

    df['price_unit'] = df['price_unit'].replace(0, 1.0).clip(lower=1.0)

    # 3. Construct Consolidated NLP Text Corpus from MAKT-MAKTX, MARA-GROES/WRKST, and EKPO-TXZ01
    primary_desc = np.where(df['material_desc'] != '', df['material_desc'], df['po_short_text'])
    df['material_desc'] = pd.Series(primary_desc, index=df.index).fillna('').astype(str).str.strip()

    combined_corpus = (
        df['material_desc'] + " " +
        df['size_dimensions'] + " " +
        df['basic_material_text'] + " " +
        df['industry_std_desc'] + " " +
        np.where(df['po_short_text'] != df['material_desc'], df['po_short_text'], "")
    ).str.strip()

    # 4. Lexical & Structural NLP Features from Material Description (MAKTX)
    df['desc_char_length'] = df['material_desc'].str.len().clip(0, 120).astype(float)
    tokens_series = df['material_desc'].str.findall(r'[A-Za-z0-9]+')
    df['desc_word_count'] = tokens_series.str.len().fillna(0).clip(0, 30).astype(float)

    alpha_chars = df['material_desc'].str.count(r'[A-Za-z]').fillna(0).astype(float)
    digit_chars = df['material_desc'].str.count(r'[0-9]').fillna(0).astype(float)
    safe_len = np.maximum(1.0, df['desc_char_length'])

    df['desc_avg_word_len'] = np.where(
        df['desc_word_count'] > 0,
        (alpha_chars + digit_chars) / df['desc_word_count'],
        0.0
    ).clip(0.0, 25.0)
    df['desc_digit_ratio'] = (digit_chars / safe_len).clip(0.0, 1.0)
    df['desc_alpha_ratio'] = (alpha_chars / safe_len).clip(0.0, 1.0)

    # Count technical & dimensional specification tokens in description
    spec_matches = combined_corpus.str.findall(REGEX_SPEC_TOKENS)
    df['desc_spec_token_count'] = spec_matches.str.len().fillna(0).clip(0, 12).astype(float)

    # PO Short Text (EKPO-TXZ01) vs Master Description (MAKT-MAKTX) lexical consistency
    norm_maktx = df['material_desc'].str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
    norm_txz01 = df['po_short_text'].str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
    has_po_text = (norm_txz01 != '')
    exact_or_prefix_match = (
        (~has_po_text) |
        (norm_maktx == norm_txz01) |
        (norm_maktx.str[:15] == norm_txz01.str[:15])
    )
    df['po_desc_consistency_score'] = np.where(
        exact_or_prefix_match,
        100.0,
        np.where(norm_maktx.str[:6] == norm_txz01.str[:6], 75.0, 35.0)
    )

    # Duplicate description mapped to multiple different Material Groups (MATKL)
    df['_norm_desc'] = np.where(norm_maktx == '', 'EMPTY_DESC', norm_maktx)
    df['dup_desc_count'] = df.groupby('_norm_desc')['material_id'].transform('nunique').astype(float)
    df['dup_desc_diff_matkl_count'] = df.groupby('_norm_desc')['material_group'].transform('nunique').astype(float)

    # 5. TF-IDF Vectorization & TruncatedSVD (Latent Semantic Analysis) on Material Descriptions
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=350,
        min_df=2,
        sublinear_tf=True,
        token_pattern=r'(?u)\b[A-Za-z0-9][A-Za-z0-9\-\./]{1,}\b'
    )
    corpus_list = combined_corpus.replace('', 'UNKNOWN_MATERIAL').tolist()
    tfidf_matrix = tfidf.fit_transform(corpus_list)

    # Lexical TF-IDF richness: number of active n-grams & L1 semantic activation energy
    nnz_per_row = np.diff(tfidf_matrix.indptr).astype(float)
    l1_energy = np.asarray(tfidf_matrix.sum(axis=1)).ravel().astype(float)
    df['tfidf_active_ngrams'] = np.clip(nnz_per_row, 0.0, 35.0)
    df['tfidf_semantic_energy'] = np.round(np.clip(l1_energy, 0.0, 10.0), 4)

    n_svd = min(5, max(1, tfidf_matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_svd, random_state=42)
    svd_Matrix = svd.fit_transform(tfidf_matrix)
    for i in range(5):
        col_name = f'svd_semantic_comp_{i+1}'
        if i < svd_Matrix.shape[1]:
            df[col_name] = np.round(svd_Matrix[:, i], 5)
        else:
            df[col_name] = 0.0

    # 6. Unsupervised NLP Domain Taxonomy Affinity Scores [0-100]
    mech_hits = combined_corpus.str.findall(REGEX_MECH_SPARES).str.len().fillna(0).astype(float)
    elec_hits = combined_corpus.str.findall(REGEX_ELEC_INST).str.len().fillna(0).astype(float)
    cons_hits = combined_corpus.str.findall(REGEX_CONSUMABLE_FASTENER).str.len().fillna(0).astype(float)
    fin_hits = combined_corpus.str.findall(REGEX_FINISHED_PAINT_CEMENT).str.len().fillna(0).astype(float)
    pack_raw_hits = combined_corpus.str.findall(REGEX_PACKAGING_RAW).str.len().fillna(0).astype(float)

    # Also incorporate contextual SAP Material Group (MATKL) & Material Type (MTART) structural cues
    is_mtart_spare = df['material_type'].isin({'ZSPA', 'ZPJS', 'ZIMS', 'ZPSP', 'ZCFM', 'ZFOC'}).astype(float)
    is_mtart_cons = df['material_type'].isin({'ZCON', 'ZPJC', 'ZIMC', 'ZMED', 'ZMES', 'ZCOL'}).astype(float)
    is_mtart_fin = df['material_type'].isin({'ZFIN', 'ZTPC', 'ZTAC', 'ZTWP', 'ZTRA', 'ZSEM', 'ZPSF', 'ZTRP', 'ZTRX'}).astype(float)
    is_mtart_pack_raw = df['material_type'].isin({'ZPAC', 'ZPPM', 'ZPAK', 'ZRAW', 'ZPRM', 'ZCHM', 'ZFUE', 'ZEXP', 'ZSCR'}).astype(float)

    df['nlp_spare_machinery_affinity'] = np.clip(mech_hits * 34.0 + is_mtart_spare * 24.0 + df['desc_spec_token_count'] * 5.0, 0.0, 100.0)
    df['nlp_elec_inst_affinity'] = np.clip(elec_hits * 38.0 + is_mtart_spare * 14.0 + df['desc_spec_token_count'] * 4.0, 0.0, 100.0)
    df['nlp_consumable_fastener_affinity'] = np.clip(cons_hits * 36.0 + is_mtart_cons * 24.0 + df['desc_spec_token_count'] * 5.0, 0.0, 100.0)
    df['nlp_finished_paint_cement_affinity'] = np.clip(fin_hits * 40.0 + is_mtart_fin * 28.0 + df['desc_spec_token_count'] * 6.0, 0.0, 100.0)
    df['nlp_packaging_raw_affinity'] = np.clip(pack_raw_hits * 38.0 + is_mtart_pack_raw * 26.0 + df['desc_spec_token_count'] * 5.0, 0.0, 100.0)

    affinity_stack = np.vstack([
        df['nlp_spare_machinery_affinity'].values,
        df['nlp_elec_inst_affinity'].values,
        df['nlp_consumable_fastener_affinity'].values,
        df['nlp_finished_paint_cement_affinity'].values,
        df['nlp_packaging_raw_affinity'].values
    ]).T  # shape (N, 5)

    sorted_aff = np.sort(affinity_stack, axis=1)
    top1_aff = sorted_aff[:, -1]
    top2_aff = sorted_aff[:, -2]
    dominant_domain_idx = np.argmax(affinity_stack, axis=1)

    domain_labels = np.array([
        'Mechanical & Plant Machinery Spares (MRO)',
        'Electrical, Instrumentation & Automation Spares',
        'Fasteners, Hardware & General Consumables',
        'Finished Cement, Concrete, Paint & Coatings',
        'Packaging, Raw Materials & Industrial Chemicals'
    ])
    df['auto_classified_domain'] = domain_labels[dominant_domain_idx]

    # Recommend standardized SAP Material Group (MATKL) based on NLP description classification
    recommended_matkl_map = {
        0: 'MECH-SPR',
        1: 'ELECT-INST',
        2: 'CONS-HDWR',
        3: 'FG-CEM-PNT',
        4: 'PACK-RAW'
    }
    default_rec_matkl = pd.Series(dominant_domain_idx, index=df.index).map(recommended_matkl_map)
    df['auto_classified_matkl'] = np.where(
        (df['material_group'] != '') & (~df['material_group'].str.fullmatch(r'\d+', na=False)),
        df['material_group'],
        default_rec_matkl
    )

    # 7. Valuation, Spend & Price Context Features
    eff_price = np.where(
        df['price_control'] == 'S',
        np.where(df['standard_price'] > 0, df['standard_price'], df['moving_avg_price']),
        np.where(df['moving_avg_price'] > 0, df['moving_avg_price'], df['standard_price'])
    ) / df['price_unit']
    eff_price = np.where(eff_price > 0, eff_price, df['avg_po_net_price'])
    df['effective_unit_price'] = np.clip(eff_price, 0.0, 1e8)
    df['log_effective_unit_price'] = np.log1p(df['effective_unit_price'])
    df['log_po_spend'] = np.log1p(np.maximum(0.0, df['total_po_spend']))
    df['log_valuated_stock'] = np.log1p(df['total_valuated_stock'].abs())

    mg_mean = df.groupby('material_group')['log_effective_unit_price'].transform('mean')
    mg_std = df.groupby('material_group')['log_effective_unit_price'].transform('std').fillna(1.0).replace(0, 1.0)
    df['matkl_price_zscore'] = ((df['log_effective_unit_price'] - mg_mean) / mg_std).clip(-6.0, 6.0).fillna(0.0)

    # 8. Taxonomy Governance & Misclassification Signal Flags
    df['is_cryptic_or_short_desc'] = (
        (df['desc_char_length'] < 9) |
        (df['desc_word_count'] <= 1) |
        (df['desc_alpha_ratio'] < 0.25)
    ).astype(int)

    df['is_unstandardized_matkl'] = (
        (df['material_group'] == '') |
        (df['material_group'].str.fullmatch(r'\d+', na=False)) |
        (df['material_group'].isin({'MISC', 'OTHER', 'GENERAL', 'NA'}))
    ).astype(int)

    df['is_po_matkl_mismatch'] = (
        ((df['po_material_group'] != '') & (df['material_group'] != '') & (df['po_material_group'] != df['material_group'])) |
        (df['distinct_po_matkl_cnt'] > 1)
    ).astype(int)

    # Detect semantic conflict between NLP-detected category in MAKTX and assigned MTART
    total_keyword_hits = mech_hits + elec_hits + cons_hits + fin_hits + pack_raw_hits
    df['is_nlp_mtart_conflict'] = (
        ((fin_hits >= 2) & (mech_hits == 0) & (df['material_type'].isin({'ZSPA', 'ZPJS', 'ZSCR'}))) |
        ((mech_hits >= 2) & (fin_hits == 0) & (df['material_type'].isin({'ZFIN', 'ZTPC', 'ZTAC', 'ZTWP'}))) |
        ((pack_raw_hits >= 2) & (mech_hits == 0) & (df['material_type'] == 'ZSPA') & (~combined_corpus.str.contains('PACK', case=False, na=False))) |
        ((df['dup_desc_diff_matkl_count'] >= 2) & (df['_norm_desc'] != 'EMPTY_DESC'))
    ).astype(int)

    df['is_missing_valuation_or_uom'] = (
        (df['base_uom'] == '') |
        ((~df['material_type'].isin({'ZNVM', 'ZFOC', 'ZMES', 'ZCFM'})) & (df['valuation_class'] == ''))
    ).astype(int)

    # 9. Three Component Scores [0-100] & Composite NLP Classification Confidence / Risk Scores
    # (A) Text Clarity & Specification Completeness Score [0-100]
    len_score = np.clip((df['desc_char_length'] / 32.0) * 100.0, 0.0, 100.0)
    word_score = np.clip((df['desc_word_count'] / 5.0) * 100.0, 0.0, 100.0)
    spec_bonus = np.clip(df['desc_spec_token_count'] * 14.0, 0.0, 28.0)
    tfidf_bonus = np.clip(df['tfidf_active_ngrams'] * 4.5, 0.0, 25.0)

    raw_text_clarity = (
        0.38 * len_score +
        0.34 * word_score +
        spec_bonus +
        tfidf_bonus -
        38.0 * df['is_cryptic_or_short_desc'] -
        12.0 * np.clip(df['dup_desc_count'] - 3.0, 0.0, 4.0)
    )
    df['text_clarity_score'] = np.round(np.clip(raw_text_clarity, 0.0, 100.0), 2)

    # (B) NLP Semantic Classification Confidence Score [0-100]
    margin_signal = np.clip(top1_aff - 0.35 * top2_aff, 0.0, 100.0)
    raw_semantic_conf = (
        22.0 +
        0.52 * top1_aff +
        0.26 * margin_signal +
        np.clip(df['tfidf_semantic_energy'] * 10.0, 0.0, 22.0) -
        24.0 * (total_keyword_hits == 0).astype(float) -
        20.0 * df['is_cryptic_or_short_desc']
    )
    df['semantic_confidence_score'] = np.round(np.clip(raw_semantic_conf, 0.0, 100.0), 2)

    # (C) Master Taxonomy Alignment Score [0-100]
    raw_tax_align = (
        0.72 * df['po_desc_consistency_score'] +
        28.0 * (1.0 - df['is_unstandardized_matkl']) -
        35.0 * df['is_nlp_mtart_conflict'] -
        28.0 * df['is_po_matkl_mismatch'] -
        24.0 * df['is_missing_valuation_or_uom'] -
        18.0 * (df['dup_desc_diff_matkl_count'] > 1).astype(float)
    )
    df['taxonomy_alignment_score'] = np.round(np.clip(raw_tax_align, 0.0, 100.0), 2)

    # Composite NLP Material Auto-Classification Confidence Score [0-100] & Misclassification Risk Index [0-100]
    raw_conf = (
        0.34 * df['text_clarity_score'] +
        0.36 * df['semantic_confidence_score'] +
        0.30 * df['taxonomy_alignment_score']
    )
    df['nlp_classification_confidence_score'] = np.round(np.clip(raw_conf, 0.0, 100.0), 2)
    df['nlp_misclassification_risk_index'] = np.round(100.0 - df['nlp_classification_confidence_score'], 2)

    # Discrete 1-5 Quintile Ratings (T-S-A: Text Clarity, Semantic Confidence, Taxonomy Alignment)
    df['t_rating'] = np.clip(np.ceil(df['text_clarity_score'] / 20.0), 1, 5).astype(int)
    df['s_rating'] = np.clip(np.ceil(df['semantic_confidence_score'] / 20.0), 1, 5).astype(int)
    df['a_rating'] = np.clip(np.ceil(df['taxonomy_alignment_score'] / 20.0), 1, 5).astype(int)
    df['classification_tier_code'] = (
        "T" + df['t_rating'].astype(str) +
        "-S" + df['s_rating'].astype(str) +
        "-A" + df['a_rating'].astype(str)
    )

    # 10. Post-Hoc Benchmark Classification Segment & Exception Flag (Y)
    # Strictly excluded from unsupervised clustering training; used ONLY for post-hoc evaluation & ranking of the 5 models
    p22 = float(np.percentile(df['nlp_classification_confidence_score'], 22.0))
    p52 = float(np.percentile(df['nlp_classification_confidence_score'], 52.0))
    p78 = float(np.percentile(df['nlp_classification_confidence_score'], 78.0))

    seg_conditions = [
        (df['nlp_classification_confidence_score'] >= p78),
        (df['nlp_classification_confidence_score'] >= p52),
        (df['nlp_classification_confidence_score'] >= p22),
    ]
    seg_choices = [
        'High-Confidence Auto-Classified Core Materials',
        'Verified Technical Spares & MRO Taxonomy',
        'Moderate-Confidence / Needs Spec Enrichment',
    ]
    df['benchmark_material_segment'] = np.select(
        seg_conditions,
        seg_choices,
        default='Misclassified / Ambiguous Material Exceptions'
    )

    risk_cutoff = round(100.0 - p22, 2)
    df['is_classification_exception'] = (
        (df['nlp_misclassification_risk_index'] >= risk_cutoff) |
        ((df['nlp_misclassification_risk_index'] >= (risk_cutoff - 7.5)) & (
            (df['is_nlp_mtart_conflict'] == 1) |
            (df['is_po_matkl_mismatch'] == 1) |
            (df['is_cryptic_or_short_desc'] == 1)
        ))
    ).astype(int)

    df.drop(columns=['_norm_desc'], inplace=True, errors='ignore')
    return df


def get_feature_matrix(df: pd.DataFrame):
    """
    Extracts 32 numerical NLP & Material Classification features for Unsupervised Clustering.
    """
    feature_cols = [
        'nlp_classification_confidence_score',
        'nlp_misclassification_risk_index',
        'text_clarity_score',
        'semantic_confidence_score',
        'taxonomy_alignment_score',
        'nlp_spare_machinery_affinity',
        'nlp_elec_inst_affinity',
        'nlp_consumable_fastener_affinity',
        'nlp_finished_paint_cement_affinity',
        'nlp_packaging_raw_affinity',
        'desc_char_length',
        'desc_word_count',
        'desc_avg_word_len',
        'desc_digit_ratio',
        'desc_alpha_ratio',
        'desc_spec_token_count',
        'tfidf_active_ngrams',
        'tfidf_semantic_energy',
        'svd_semantic_comp_1',
        'svd_semantic_comp_2',
        'svd_semantic_comp_3',
        'svd_semantic_comp_4',
        'svd_semantic_comp_5',
        'po_desc_consistency_score',
        'dup_desc_count',
        'dup_desc_diff_matkl_count',
        'is_cryptic_or_short_desc',
        'is_unstandardized_matkl',
        'is_po_matkl_mismatch',
        'is_nlp_mtart_conflict',
        'log_effective_unit_price',
        'matkl_price_zscore'
    ]

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df['is_classification_exception'].values
    return X, y, feature_cols


def prepare_train_test_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Extracts 32 NLP & Taxonomy features, performs stratified train/test split for post-hoc evaluation,
    and applies RobustScaler (fit strictly on X_train without labels).
    """
    X, y, feature_cols = get_feature_matrix(df)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=test_size, random_state=random_state, stratify=y
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
