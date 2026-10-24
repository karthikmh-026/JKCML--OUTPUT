"""
Preprocessing & NLP Feature Engineering Module - Use Case 26: Breakdown Root Cause Analysis (NLP)
Transforms raw SAP Plant Maintenance Order Master Data (AUFK), Order Header Scheduling (AFKO),
Order Confirmations & Technician Execution Logs (AFRU), Spare Parts Goods Movements (AUFM, MAKT),
and Work Center Master (CRHD) into 32 domain-engineered numerical NLP & Plant Maintenance features
to extract root causes from maintenance text and identify critical breakdown and unverified root-cause exceptions.
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


# NLP Lexical Taxonomies for SAP Plant Maintenance (PM) Breakdown Root Cause Extraction (J.K. Cement)
REGEX_MECH_BEARING_DRIVE = re.compile(
    r'\b(BEARING|BRG|PLUMMER|HOUSING|GEAR\s*BOX|GEARBOX|GEAR|PINION|GIRTH|COUPLING|FLUID\s*COUPLING|'
    r'SHAFT|ROLLER|PULLROD|PULL\s*ROD|PULLEY|SPROCKET|IMPELLER|ROTOR|STATOR|FAN|BLOWER|CRUSHER|'
    r'HAMMER|HRP|POLYCOM|DIAPHRAGM|GRINDING|MILL|KILN|TYRE|THRUST|TRUNNION|VIBRATION|ALIGNMENT|'
    r'BALANCING|SEIZED|SEIZURE|BROKEN|CRACK|CRACKED|BENT|LOOSE|BOLT|ANCHOR|FASTENER)\b',
    re.IGNORECASE
)

REGEX_ELEC_MOTOR_TRIP = re.compile(
    r'\b(MOTOR|TRIP|TRIPPED|TRIPPING|OVERCURRENT|OVER\s*CURRENT|OVERLOAD|EARTH\s*FAULT|SHORT\s*CIRCUIT|'
    r'LRS|CONTACTOR|BREAKER|MCCB|ACB|VCB|RELAY|FUSE|VFD|DRIVE|INVERTER|THYRISTOR|CABLE|WIRE|TERMINAL|'
    r'WINDING|INSULATION|TRANSFORMER|TRAFO|HT|LT|MCC|PCC|PANEL|BUSBAR|LIGHT|LIGHTING|MAST|POLE|'
    r'EARTHING|HOIST|BRAKE|BREAK|CARBON\s*BRUSH|SLIP\s*RING|BATTERY|UPS|GENERATOR|DG|ALTERNATOR)\b',
    re.IGNORECASE
)

REGEX_CONVEYOR_BELT_WEAR = re.compile(
    r'\b(BELT|V\-BELT|CONVEYOR|BC|BUCKET|ELEVATOR|BE|CHAIN|LINK|SPLICING|SPLICE|JOINT|VULCANIZING|'
    r'SKIRT|SCRAPER|IDLER|GARLAND|CHUTE|HOPPER|LINER|WEAR\s*PLATE|PLATE|BAFFLE|WELDING|WELDER|'
    r'FABRICATION|PATCH|BAG|FILTER|BAG\s*FILTER|RABH|ESP|CLOTH|CAGE|AIR\s*SLIDE|CANVAS|SILO|SPOUT|'
    r'EXTRACTOR|RECLAIMER|HARROW|STACKER|APRON|FEEDER|DBC|DRAG\s*CHAIN|SCREW|CASING|SHELL|DUCT|'
    r'DAMAGED|WORN|WORN\s*OUT|TEAR|TORN|CHOKED|CHOKING|JAM|JAMMED|BLOCKED|CLEANING)\b',
    re.IGNORECASE
)

REGEX_HYDR_PNEUM_LEAK = re.compile(
    r'\b(OIL|LUBE|LUBRICATION|GREASE|GREASING|LEAK|LEAKAGE|SEEPAGE|SEAL|OIL\s*SEAL|GASKET|O\-RING|'
    r'PACKING|GLAND|HYDRAULIC|HYD|POWER\s*PACK|ACCUMULATOR|CYLINDER|PISTON|BELLOW|ROD|VALVE|SOV|'
    r'SOLENOID|NRV|ROTARY\s*VALVE|RAL|FLAP|DAMPER|GATE|ACTUATOR|PNEUMATIC|AIR\s*BLASTER|CANNON|'
    r'COMPRESSOR|BLOWER|AIR\s*DRYER|DRYER|PUMP|WATER|COOLING|HOSE|PIPE|PIPELINE|TUBE|NOZZLE|'
    r'FILTER\s*ELEMENT|STRAINER|PRESSURE\s*DROP|LOW\s*PRESSURE|TEMP\s*HIGH|OVERHEAT|HEATING)\b',
    re.IGNORECASE
)

REGEX_INST_SENSOR_PLC = re.compile(
    r'\b(CALIBRATION|CALI|CALIB|WEIGH|WEIGHFEEDER|WEIGH\s*FEEDER|PFISTER|SCHENCK|SCALE|WEIGHBRIDGE|'
    r'WEIGH\s*BRIDGE|PACKER|ROTO\s*PACKER|LOAD\s*CELL|LOADCELL|SENSOR|PROXY|PROXIMITY|PULSAR|ZERO\s*SPEED|'
    r'SWITCH|LIMIT\s*SWITCH|PULL\s*CORD|SWAY|EMERGENCY|RTD|THERMOCOUPLE|TEMP\s*SENSOR|PYROMETER|SCANNER|'
    r'TRANSMITTER|DP\s*TRANSMITTER|GAUGE|LEVEL|FLOW\s*METER|METER|PLC|DCS|SCADA|CCR|PID|INTERLOCK|'
    r'AUTO\s*PROCESS|COMMUNICATION|PROFIBUS|MODBUS|MODULE|CARD|I/O|GAS\s*ANALYZER|STROBOSCOPE|CCTV)\b',
    re.IGNORECASE
)

REGEX_FAILURE_SYMPTOM_TOKENS = re.compile(
    r'\b(BROKEN|DAMAGED|DAMAGE|LEAK|LEAKAGE|TRIP|TRIPPED|WORN|WORN\s*OUT|HIGH\s*TEMP|TEMP\s*HIGH|'
    r'OVERCURRENT|OVERLOAD|SEIZED|JAMMED|CHOKED|TEAR|TORN|CRACK|CRACKED|FAULT|ISSUE|PROBLEM|'
    r'NOT\s*WORKING|NOT\s*STARTED|ABNORMAL|NOISE|VIBRATION|BENT|BURNT|FAILURE|FAILED|REPLACEMENT|REPAIR)\b',
    re.IGNORECASE
)

REGEX_EQUIPMENT_TAG = re.compile(
    r'\b([0-9]{2,3}[A-Z]{2,3}[0-9]{2,4}|[A-Z][0-9][A-Z][0-9]{2}|(?:RM|CM|KL|KM|CR|BC|BE|FN|BF|ESP|RABH|PACKER|DBC|HRP|LRS|CCR)[\-\s_]*[0-9]{1,3})\b',
    re.IGNORECASE
)

REGEX_VAGUE_CONFIRMATION = re.compile(
    r'^(WORK\s*DONE|ORDER\s*CLOSED|ITEMS?\s*ISSUED|MATERIAL\s*ISSUES?|GENERAL\s*ITEMS?\s*ISSUE|'
    r'DONE|OK|CHECKED|POINTS\s*CHECK|TO\s*BE\s*CHE[CK]HED.*|CREATED\s*BY\s*MISTAKE)$',
    re.IGNORECASE
)


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw SAP Plant Maintenance Order Master Data (AUFK), Order Header (AFKO), Confirmations (AFRU),
    Spare Parts Goods Movements (AUFM/MAKT), and Work Center Master (CRHD),
    extracts Breakdown Root Causes and Failure Modes from maintenance text using NLP, and engineers
    32 domain-specific features for Unsupervised Breakdown Root Cause Analysis.

    Missing value handling: If data is present, format and retain it. If missing/null/nan/none/empty,
    set to blank string ("") for text/categorical columns or 0.0 for numeric columns.
    """
    df = df.copy()

    # 1. Clean string columns - sanitize any null/none/nan into blank ""
    str_cols = [
        'company_code', 'company_name', 'company_city', 'company_currency',
        'order_number', 'order_type', 'order_category', 'maintenance_text',
        'confirmation_text', 'spare_parts_texts', 'plant', 'plant_name', 'plant_city',
        'work_center', 'work_center_cat', 'wc_person_resp', 'cost_center',
        'profit_center', 'business_area', 'location_code', 'order_priority',
        'mrp_controller', 'maintenance_supervisor', 'created_by',
        'created_date', 'release_date', 'teco_date', 'close_date',
        'basic_start_date', 'basic_finish_date', 'actual_start_date', 'actual_finish_date',
        'exec_start_date', 'exec_end_date',
        'phase_created', 'phase_released', 'phase_teco', 'phase_closed',
        'duration_unit', 'variance_reason_code', 'final_conf_flag',
        'primary_spare_matnr', 'spare_uom', 'storage_location'
    ]
    for col in str_cols:
        if col in df.columns:
            s = (
                df[col]
                .fillna('')
                .astype(str)
                .str.replace('\ufffd', ' ', regex=False)
                .str.replace(r'\s+', ' ', regex=True)
                .str.strip()
            )
            # Replace string representations of missing values with true blank ""
            s = s.replace(['None', 'none', 'NaN', 'nan', 'null', 'NULL', '<NA>', '<na>', 'NaT', 'nat'], '')
            df[col] = s
        else:
            df[col] = ''

    # 2. Clean numeric columns
    num_cols = [
        'confirmation_count', 'total_actual_duration', 'total_labor_hours',
        'goods_movement_count', 'distinct_spares_count', 'total_spare_cost_inr',
        'total_spare_qty'
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).abs()
        else:
            df[col] = 0.0

    # Map SAP PM Order Type Description
    order_type_map = {
        'PM01': 'PM01 - Breakdown Maintenance Order',
        'PM02': 'PM02 - Preventive Maintenance Order',
        'PM03': 'PM03 - Corrective / General Maintenance Order',
        'PM04': 'PM04 - Refurbishment Maintenance Order',
        'PM05': 'PM05 - Calibration Maintenance Order',
        'PM07': 'PM07 - Shutdown / Overhaul Maintenance Order',
        'PM10': 'PM10 - Condition / Inspection Maintenance Order'
    }
    df['order_type_desc'] = df['order_type'].map(lambda x: order_type_map.get(x, f"{x} - Plant Maintenance Order" if x else ""))

    # 3. Construct Consolidated NLP Maintenance & Breakdown Text Corpus
    # Combines AUFK-KTEXT (Maintenance Order Short Text), AFRU-LTXA1 (Technician Confirmation Text),
    # and AUFM/MAKT-MAKTX (Consumed Spare Parts Descriptions)
    combined_corpus = (
        df['maintenance_text'] + " " +
        df['confirmation_text'] + " " +
        df['spare_parts_texts']
    ).str.replace(r'\s+', ' ', regex=True).str.strip()
    df['combined_maintenance_corpus'] = combined_corpus

    # 4. Extract Plant Equipment Tag & Failure Symptom from Maintenance Text
    eq_matches = df['maintenance_text'].str.findall(REGEX_EQUIPMENT_TAG)
    df['equipment_tag_count'] = eq_matches.str.len().fillna(0).clip(0, 5).astype(float)
    # If equipment tag is found in text, put it; if not present, put blank "" (no placeholder)
    df['extracted_equipment_tag'] = eq_matches.apply(
        lambda m: m[0].upper() if isinstance(m, list) and len(m) > 0 else ''
    )

    symp_matches = combined_corpus.str.findall(REGEX_FAILURE_SYMPTOM_TOKENS)
    df['failure_symptom_token_count'] = symp_matches.str.len().fillna(0).clip(0, 12).astype(float)

    # 5. Lexical & Structural NLP Features from Maintenance Text (AUFK-KTEXT + AFRU-LTXA1)
    df['maint_text_char_length'] = df['maintenance_text'].str.len().clip(0, 120).astype(float)
    ktext_tokens = df['maintenance_text'].str.findall(r'[A-Za-z0-9]+')
    df['maint_text_word_count'] = ktext_tokens.str.len().fillna(0).clip(0, 25).astype(float)

    df['conf_text_char_length'] = df['confirmation_text'].str.len().clip(0, 120).astype(float)
    alpha_chars = df['maintenance_text'].str.count(r'[A-Za-z]').fillna(0).astype(float)
    digit_chars = df['maintenance_text'].str.count(r'[0-9]').fillna(0).astype(float)
    safe_len = np.maximum(1.0, df['maint_text_char_length'])

    df['maint_text_digit_ratio'] = (digit_chars / safe_len).clip(0.0, 1.0)
    df['maint_text_alpha_ratio'] = (alpha_chars / safe_len).clip(0.0, 1.0)

    # Equipment & Text Recurrence Frequency (Chronic / Repeat Breakdown Pattern Detection)
    norm_ktext = df['maintenance_text'].str.upper().str.replace(r'[^A-Z0-9]', '', regex=True)
    df['_norm_ktext'] = np.where(norm_ktext == '', 'EMPTY_KTEXT', norm_ktext)
    df['repeat_text_order_count'] = df.groupby('_norm_ktext')['order_number'].transform('count').clip(1, 200).astype(float)
    
    # Equipment order frequency only calculated where equipment tag is present; otherwise 0.0
    has_eq = df['extracted_equipment_tag'] != ''
    eq_freq = df[has_eq].groupby('extracted_equipment_tag')['order_number'].transform('count').clip(1, 500).astype(float)
    df['equipment_order_frequency'] = 0.0
    df.loc[has_eq, 'equipment_order_frequency'] = eq_freq

    # 6. TF-IDF Vectorization & TruncatedSVD (Latent Semantic Analysis) on Maintenance Text Corpus
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=350,
        min_df=5,
        sublinear_tf=True,
        token_pattern=r'(?u)\b[A-Za-z0-9][A-Za-z0-9\-\./_]{1,}\b'
    )
    corpus_list = combined_corpus.replace('', 'MAINTENANCE_OPERATION').tolist()
    tfidf_matrix = tfidf.fit_transform(corpus_list)

    nnz_per_row = np.diff(tfidf_matrix.indptr).astype(float)
    l1_energy = np.asarray(tfidf_matrix.sum(axis=1)).ravel().astype(float)
    df['tfidf_active_ngrams'] = np.clip(nnz_per_row, 0.0, 40.0)
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

    # 7. Unsupervised NLP Root Cause Domain Affinity Scores [0-100]
    mech_hits = combined_corpus.str.findall(REGEX_MECH_BEARING_DRIVE).str.len().fillna(0).astype(float)
    elec_hits = combined_corpus.str.findall(REGEX_ELEC_MOTOR_TRIP).str.len().fillna(0).astype(float)
    conv_hits = combined_corpus.str.findall(REGEX_CONVEYOR_BELT_WEAR).str.len().fillna(0).astype(float)
    hydr_hits = combined_corpus.str.findall(REGEX_HYDR_PNEUM_LEAK).str.len().fillna(0).astype(float)
    inst_hits = combined_corpus.str.findall(REGEX_INST_SENSOR_PLC).str.len().fillna(0).astype(float)

    # Work Center Discipline Cues (VAPLZ: MECH-*, ELEC-*, INST-*, PROD-*, DEPT-*)
    wc_upper = df['work_center'].str.upper()
    wc_is_mech = wc_upper.str.startswith(('MECH', 'CRU', 'MIN')).astype(float)
    wc_is_elec = wc_upper.str.startswith(('ELEC', 'PWR', 'WHR', 'CPP')).astype(float)
    wc_is_inst = (wc_upper.str.startswith(('INST', 'AUT')) | (df['order_type'] == 'PM05')).astype(float)

    df['nlp_mechanical_bearing_drive_affinity'] = np.clip(
        mech_hits * 34.0 + wc_is_mech * 18.0 + df['equipment_tag_count'] * 6.0, 0.0, 100.0
    )
    df['nlp_electrical_motor_trip_affinity'] = np.clip(
        elec_hits * 36.0 + wc_is_elec * 20.0 + df['equipment_tag_count'] * 6.0, 0.0, 100.0
    )
    df['nlp_conveyor_belt_wear_affinity'] = np.clip(
        conv_hits * 34.0 + wc_is_mech * 15.0 + df['equipment_tag_count'] * 6.0, 0.0, 100.0
    )
    df['nlp_hydraulic_pneumatic_leak_affinity'] = np.clip(
        hydr_hits * 35.0 + (wc_is_mech * 12.0 + wc_is_inst * 10.0) + df['equipment_tag_count'] * 6.0, 0.0, 100.0
    )
    df['nlp_instrumentation_plc_calib_affinity'] = np.clip(
        inst_hits * 36.0 + wc_is_inst * 22.0 + df['equipment_tag_count'] * 6.0, 0.0, 100.0
    )

    affinity_stack = np.vstack([
        df['nlp_mechanical_bearing_drive_affinity'].values,
        df['nlp_electrical_motor_trip_affinity'].values,
        df['nlp_conveyor_belt_wear_affinity'].values,
        df['nlp_hydraulic_pneumatic_leak_affinity'].values,
        df['nlp_instrumentation_plc_calib_affinity'].values
    ]).T  # shape (N, 5)

    sorted_aff = np.sort(affinity_stack, axis=1)
    top1_aff = sorted_aff[:, -1]
    top2_aff = sorted_aff[:, -2]
    dominant_domain_idx = np.argmax(affinity_stack, axis=1)

    domain_labels = np.array([
        'Mechanical Drive, Bearing, Gearbox & Roller Failure',
        'Electrical Motor, VFD Drive, Contactor & Power Trip',
        'Conveyor Belt, Bucket Elevator, Chute & Liner Wear',
        'Hydraulic, Pneumatic, Lube Oil & Valve Leakage',
        'Instrumentation, Weighfeeder, Sensor & PLC Fault'
    ])
    df['extracted_root_cause_domain'] = domain_labels[dominant_domain_idx]

    # 8. Extract Specific Granular Root Cause Code & Description from Maintenance Text
    corpus_upper = combined_corpus.str.upper()

    rc_conds = [
        corpus_upper.str.contains(r'\b(BEARING|BRG|PLUMMER|TRUNNION)\b', regex=True, na=False),
        corpus_upper.str.contains(r'\b(GEAR\s*BOX|GEARBOX|GEAR|PINION|GIRTH)\b', regex=True, na=False),
        corpus_upper.str.contains(r'\b(COUPLING|SHAFT|ROLLER|PULLROD|PULLEY|IMPELLER|ROTOR|HAMMER|HRP|POLYCOM)\b', regex=True, na=False),
        corpus_upper.str.contains(r'\b(TRIP|TRIPPED|OVERCURRENT|OVERLOAD|EARTH\s*FAULT|SHORT\s*CIRCUIT)\b', regex=True, na=False),
        corpus_upper.str.contains(r'\b(LRS|CONTACTOR|BREAKER|MCCB|ACB|RELAY|VFD|DRIVE|INVERTER|PANEL)\b', regex=True, na=False),
        corpus_upper.str.contains(r'\b(MOTOR|WINDING|CABLE|TRANSFORMER|HOIST|BRAKE|LIGHTING|POLE|EARTHING)\b', regex=True, na=False),
        corpus_upper.str.contains(r'\b(BELT|V\-BELT|BC|SPLICE|SPLICING|IDLER|PULLEY|SKIRT|SCRAPER)\b', regex=True, na=False),
        corpus_upper.str.contains(r'\b(BUCKET|ELEVATOR|CHAIN|LINK|DBC|DRAG\s*CHAIN|RECLAIMER|HARROW|APRON)\b', regex=True, na=False),
        corpus_upper.str.contains(r'\b(BAG|FILTER|CLOTH|CAGE|AIR\s*SLIDE|CHUTE|LINER|PLATE|WELDING|CHOKED|JAM)\b', regex=True, na=False),
        corpus_upper.str.contains(r'\b(OIL|LUBE|LUBRICATION|GREASE|LEAK|LEAKAGE|SEAL|GASKET|POWER\s*PACK|HYDRAULIC)\b', regex=True, na=False),
        corpus_upper.str.contains(r'\b(VALVE|SOV|SOLENOID|NRV|CYLINDER|PISTON|BELLOW|DAMPER|FLAP|PUMP|COMPRESSOR|DRYER)\b', regex=True, na=False),
        corpus_upper.str.contains(r'\b(CALIBRATION|CALI|CALIB|WEIGH|PFISTER|SCALE|WEIGHBRIDGE|PACKER|LOADCELL)\b', regex=True, na=False),
        corpus_upper.str.contains(r'\b(PROXY|SENSOR|PULSAR|SWITCH|EMERGENCY|RTD|THERMOCOUPLE|TRANSMITTER|PLC|AUTO|CCR)\b', regex=True, na=False),
    ]
    rc_codes = [
        'RC-MECH-01',
        'RC-MECH-02',
        'RC-MECH-03',
        'RC-ELEC-01',
        'RC-ELEC-02',
        'RC-ELEC-03',
        'RC-CONV-01',
        'RC-CONV-02',
        'RC-CONV-03',
        'RC-HYDR-01',
        'RC-HYDR-02',
        'RC-INST-01',
        'RC-INST-02',
    ]
    default_rc_code = pd.Series(dominant_domain_idx, index=df.index).map({
        0: 'RC-MECH-03',
        1: 'RC-ELEC-03',
        2: 'RC-CONV-03',
        3: 'RC-HYDR-01',
        4: 'RC-INST-02'
    })
    df['extracted_root_cause_code'] = pd.Series(
        np.select(rc_conds, rc_codes, default=default_rc_code), index=df.index
    ).astype(str)

    rc_desc_map = {
        'RC-MECH-01': 'Bearing Overheating, Seizure & Housing Wear',
        'RC-MECH-02': 'Gearbox Internal Gear, Pinion & Transmission Wear',
        'RC-MECH-03': 'Mechanical Drive Shaft, Coupling, Roller & Pullrod Fatigue',
        'RC-ELEC-01': 'Motor Overcurrent, Thermal Overload & Earth-Fault Trip',
        'RC-ELEC-02': 'LRS Starter, Contactor, VFD Drive & Switchgear Fault',
        'RC-ELEC-03': 'Electrical Motor Winding, Power Cable & Brake Assembly Fault',
        'RC-CONV-01': 'Conveyor Belt Tear, Splice Failure & Tracking Misalignment',
        'RC-CONV-02': 'Bucket Elevator / Drag Chain Link Elongation & Breakage',
        'RC-CONV-03': 'Bag Filter Cloth Tear, Air-Slide Choking & Chute Liner Abrasion',
        'RC-HYDR-01': 'Hydraulic Power Pack, Lube Oil System & Shaft Seal Leakage',
        'RC-HYDR-02': 'Pneumatic Cylinder Bellow, Solenoid Valve (SOV) & Pump Failure',
        'RC-INST-01': 'Weighfeeder, Packer & Loadcell Calibration Drift',
        'RC-INST-02': 'Proximity Sensor, Emergency Switch & PLC Interlock Fault'
    }
    df['extracted_root_cause'] = df['extracted_root_cause_code'].map(
        lambda x: rc_desc_map.get(x, 'General Plant Equipment Wear & Maintenance')
    )

    # 9. Operational Downtime, Turnaround & Spare Parts Financial Exposure Features
    created_dt = pd.to_datetime(df['created_date'], errors='coerce')
    teco_dt = pd.to_datetime(df['teco_date'], errors='coerce')
    basic_end_dt = pd.to_datetime(df['basic_finish_date'], errors='coerce')
    ref_end_dt = teco_dt.fillna(basic_end_dt)

    turnaround_days = (ref_end_dt - created_dt).dt.days.fillna(0.0).clip(0.0, 365.0).astype(float)
    df['order_turnaround_days'] = turnaround_days
    df['log_actual_duration_hrs'] = np.log1p(np.maximum(0.0, df['total_actual_duration'] + df['total_labor_hours']))
    df['log_spare_cost_inr'] = np.log1p(np.maximum(0.0, df['total_spare_cost_inr']))

    # 10. Plant Maintenance RCA Governance & Exception Signal Flags
    # Flag 1: Breakdown / Emergency Order or Acute Failure Symptom in Text
    df['is_breakdown_or_emergency_order'] = (
        (df['order_type'] == 'PM01') |
        (df['order_priority'].isin({'1', '2'})) |
        (df['failure_symptom_token_count'] >= 1)
    ).astype(int)

    # Flag 2: Vague, Missing, or Non-Diagnostic Technician Confirmation Text (AFRU-LTXA1)
    is_vague_ltxa1 = (
        (df['confirmation_text'] == '') |
        (df['conf_text_char_length'] <= 10) |
        df['confirmation_text'].str.match(REGEX_VAGUE_CONFIRMATION, na=False)
    )
    df['is_vague_or_missing_rca_text'] = (
        is_vague_ltxa1 &
        ((df['maint_text_word_count'] <= 3) | (df['failure_symptom_token_count'] == 0) | (df['order_type'].isin({'PM01', 'PM03', 'PM07'})))
    ).astype(int)

    # Flag 3: Work Center Discipline vs NLP Root Cause Domain Mismatch
    df['is_work_center_domain_mismatch'] = (
        ((dominant_domain_idx == 1) & (wc_is_mech == 1) & (elec_hits >= 1) & (mech_hits == 0)) |
        ((dominant_domain_idx.isin([0, 2]) if hasattr(dominant_domain_idx, 'isin') else np.isin(dominant_domain_idx, [0, 2])) & (wc_is_elec == 1) & ((mech_hits + conv_hits) >= 1) & (elec_hits == 0)) |
        ((dominant_domain_idx == 4) & (wc_is_mech == 1) & (inst_hits >= 1) & (mech_hits == 0)) |
        (df['work_center'] == '')
    ).astype(int)

    # Flag 4: Unclosed / Backlog Breakdown or High-Cost Spare Part Consumption Without RCA Verification
    df['is_unclosed_or_high_cost_breakdown'] = (
        ((df['phase_teco'] == '') & (df['phase_closed'] == '') & (df['is_breakdown_or_emergency_order'] == 1)) |
        ((df['total_spare_cost_inr'] >= 10000.0) & is_vague_ltxa1) |
        ((df['equipment_order_frequency'] >= 15) & (df['is_breakdown_or_emergency_order'] == 1))
    ).astype(int)

    # 11. Three Component Scores [0-100] & Composite NLP Root Cause Confidence / Breakdown Severity Scores
    total_keyword_hits = mech_hits + elec_hits + conv_hits + hydr_hits + inst_hits

    # (A) Maintenance Text Diagnostic Clarity Score [0-100]
    ktext_len_score = np.clip((df['maint_text_char_length'] / 34.0) * 100.0, 0.0, 100.0)
    ktext_word_score = np.clip((df['maint_text_word_count'] / 5.0) * 100.0, 0.0, 100.0)
    conf_bonus = np.where(~is_vague_ltxa1, np.clip((df['conf_text_char_length'] / 28.0) * 28.0, 8.0, 28.0), 0.0)
    eq_bonus = np.clip(df['equipment_tag_count'] * 12.0, 0.0, 18.0)

    raw_text_clarity = (
        0.36 * ktext_len_score +
        0.32 * ktext_word_score +
        conf_bonus +
        eq_bonus -
        28.0 * df['is_vague_or_missing_rca_text']
    )
    df['text_diagnostic_clarity_score'] = np.round(np.clip(raw_text_clarity, 0.0, 100.0), 2)

    # (B) NLP Root Cause Semantic Specificity Score [0-100]
    margin_signal = np.clip(top1_aff - 0.35 * top2_aff, 0.0, 100.0)
    raw_rca_spec = (
        22.0 +
        0.48 * top1_aff +
        0.26 * margin_signal +
        np.clip(df['tfidf_semantic_energy'] * 9.0, 0.0, 20.0) -
        22.0 * (total_keyword_hits == 0).astype(float) -
        18.0 * df['is_vague_or_missing_rca_text']
    )
    df['rca_semantic_specificity_score'] = np.round(np.clip(raw_rca_spec, 0.0, 100.0), 2)

    # (C) Plant Maintenance Execution & Work-Center Alignment Score [0-100]
    has_teco_or_closed = ((df['phase_teco'] == 'X') | (df['phase_closed'] == 'X')).astype(float)
    has_conf = (df['confirmation_count'] > 0).astype(float)
    raw_pm_align = (
        52.0 +
        24.0 * has_teco_or_closed +
        18.0 * has_conf +
        8.0 * (~is_vague_ltxa1).astype(float) -
        30.0 * df['is_work_center_domain_mismatch'] -
        24.0 * df['is_unclosed_or_high_cost_breakdown'] -
        18.0 * (df['order_type'] == 'PM01').astype(float) -
        10.0 * np.clip(df['failure_symptom_token_count'], 0.0, 2.0)
    )
    df['pm_execution_alignment_score'] = np.round(np.clip(raw_pm_align, 0.0, 100.0), 2)

    # Composite NLP Root Cause Confidence Score [0-100] & Breakdown Severity Risk Index [0-100]
    raw_conf = (
        0.34 * df['text_diagnostic_clarity_score'] +
        0.36 * df['rca_semantic_specificity_score'] +
        0.30 * df['pm_execution_alignment_score']
    )
    df['nlp_rca_confidence_score'] = np.round(np.clip(raw_conf, 0.0, 100.0), 2)
    df['breakdown_severity_risk_index'] = np.round(100.0 - df['nlp_rca_confidence_score'], 2)

    # Discrete 1-5 Quintile Ratings (D-R-E: Diagnostic Clarity, Root-Cause Specificity, Execution Alignment)
    df['d_rating'] = np.clip(np.ceil(df['text_diagnostic_clarity_score'] / 20.0), 1, 5).astype(int)
    df['r_rating'] = np.clip(np.ceil(df['rca_semantic_specificity_score'] / 20.0), 1, 5).astype(int)
    df['e_rating'] = np.clip(np.ceil(df['pm_execution_alignment_score'] / 20.0), 1, 5).astype(int)
    df['rca_tier_code'] = (
        "D" + df['d_rating'].astype(str) +
        "-R" + df['r_rating'].astype(str) +
        "-E" + df['e_rating'].astype(str)
    )

    # 12. Post-Hoc Benchmark Breakdown Root Cause Segment & Exception Flag (Y)
    # Strictly excluded from unsupervised clustering training; used ONLY for post-hoc evaluation & ranking of the 5 models
    p22 = float(np.percentile(df['nlp_rca_confidence_score'], 22.0))
    p52 = float(np.percentile(df['nlp_rca_confidence_score'], 52.0))
    p78 = float(np.percentile(df['nlp_rca_confidence_score'], 78.0))

    seg_conditions = [
        (df['nlp_rca_confidence_score'] >= p78),
        (df['nlp_rca_confidence_score'] >= p52),
        (df['nlp_rca_confidence_score'] >= p22),
    ]
    seg_choices = [
        'Well-Documented Preventive & Calibrated PM Orders',
        'Verified Corrective Maintenance with Clear Root Cause',
        'Moderate-Clarity Maintenance / Chronic Wear Orders',
    ]
    df['benchmark_rca_segment'] = np.select(
        seg_conditions,
        seg_choices,
        default='Critical Breakdown & Unverified Root-Cause Exceptions'
    )

    risk_cutoff = round(100.0 - p22, 2)
    df['is_breakdown_rca_exception'] = (
        (df['breakdown_severity_risk_index'] >= risk_cutoff) |
        ((df['breakdown_severity_risk_index'] >= (risk_cutoff - 6.0)) & (
            (df['is_breakdown_or_emergency_order'] == 1) |
            (df['is_unclosed_or_high_cost_breakdown'] == 1) |
            (df['is_work_center_domain_mismatch'] == 1)
        ))
    ).astype(int)

    df.drop(columns=['_norm_ktext'], inplace=True, errors='ignore')
    return df


def get_feature_matrix(df: pd.DataFrame):
    """
    Extracts 32 numerical NLP & Plant Maintenance features for Unsupervised Breakdown Root Cause Clustering.
    """
    feature_cols = [
        'nlp_rca_confidence_score',
        'breakdown_severity_risk_index',
        'text_diagnostic_clarity_score',
        'rca_semantic_specificity_score',
        'pm_execution_alignment_score',
        'nlp_mechanical_bearing_drive_affinity',
        'nlp_electrical_motor_trip_affinity',
        'nlp_conveyor_belt_wear_affinity',
        'nlp_hydraulic_pneumatic_leak_affinity',
        'nlp_instrumentation_plc_calib_affinity',
        'maint_text_char_length',
        'maint_text_word_count',
        'conf_text_char_length',
        'maint_text_digit_ratio',
        'maint_text_alpha_ratio',
        'equipment_tag_count',
        'failure_symptom_token_count',
        'tfidf_active_ngrams',
        'tfidf_semantic_energy',
        'svd_semantic_comp_1',
        'svd_semantic_comp_2',
        'svd_semantic_comp_3',
        'svd_semantic_comp_4',
        'svd_semantic_comp_5',
        'repeat_text_order_count',
        'equipment_order_frequency',
        'confirmation_count',
        'log_actual_duration_hrs',
        'distinct_spares_count',
        'log_spare_cost_inr',
        'order_turnaround_days',
        'is_breakdown_or_emergency_order'
    ]

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df['is_breakdown_rca_exception'].values
    return X, y, feature_cols


def prepare_train_test_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Extracts 32 NLP & Plant Maintenance features, performs stratified train/test split for post-hoc evaluation,
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
