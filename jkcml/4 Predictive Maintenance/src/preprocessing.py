"""
Preprocessing & Feature Engineering Module - Use Case 4: Predictive Maintenance Classification
(Predict Equipment Failures Before Breakdown)
Cleans raw SAP Plant Maintenance data from AUFK, AFKO, AFRU, AUFM, CRHD, T001W, T001,
engineers 32 domain-specific operational & degradation features, and applies RobustScaler
for 100% unsupervised machine learning.
"""
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    'planned_duration_days',
    'actual_duration_days',
    'duration_variance_days',
    'start_delay_days',
    'completion_delay_days',
    'teco_cycle_days',
    'schedule_adherence_ratio',
    'total_actual_duration_hrs',
    'total_labor_hours',
    'confirmation_count',
    'has_confirmations',
    'duration_per_confirmation',
    'is_final_confirmed',
    'has_variance_reason',
    'spare_parts_item_count',
    'total_spare_parts_cost',
    'total_parts_quantity',
    'cost_per_spare_part',
    'has_spare_parts',
    'high_spare_cost_flag',
    'work_center_frequency',
    'work_center_breakdown_ratio',
    'work_center_avg_cost',
    'priority_numeric',
    'is_phase_completed',
    'is_phase_closed',
    'has_long_text',
    'nlp_vibration_bearing_flag',
    'nlp_overheating_leakage_flag',
    'nlp_electrical_motor_flag',
    'nlp_wear_damage_flag',
    'composite_nlp_risk_score'
]


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw SAP Plant Maintenance records into 32 domain features capturing
    equipment operating stress, execution anomalies, spare part intensity, and degradation signals.
    """
    df = df.copy()

    # 1. Date Parsing & Duration Features
    date_cols = [
        'creation_date', 'change_date', 'release_date', 'teco_date', 'close_date',
        'planned_start_date', 'planned_finish_date', 'actual_start_date',
        'actual_finish_date', 'confirmed_finish_date', 'exec_start_date', 'exec_end_date'
    ]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Planned duration
    df['planned_duration_days'] = (df['planned_finish_date'] - df['planned_start_date']).dt.days.fillna(1.0)
    df['planned_duration_days'] = df['planned_duration_days'].clip(lower=0.0, upper=365.0)

    # Actual duration (fallback to teco - creation if dates missing)
    act_dur = (df['actual_finish_date'] - df['actual_start_date']).dt.days
    act_fallback = (df['teco_date'] - df['creation_date']).dt.days
    df['actual_duration_days'] = act_dur.fillna(act_fallback).fillna(1.0).clip(lower=0.0, upper=365.0)

    # Duration variance
    df['duration_variance_days'] = (df['actual_duration_days'] - df['planned_duration_days']).clip(lower=-60.0, upper=365.0)

    # Start and completion delays
    df['start_delay_days'] = (df['actual_start_date'] - df['planned_start_date']).dt.days.fillna(0.0).clip(lower=-30.0, upper=180.0)
    df['completion_delay_days'] = (df['actual_finish_date'] - df['planned_finish_date']).dt.days.fillna(0.0).clip(lower=-30.0, upper=180.0)

    # TECO cycle days
    df['teco_cycle_days'] = (df['teco_date'] - df['creation_date']).dt.days.fillna(0.0).clip(lower=0.0, upper=365.0)

    # Schedule adherence ratio
    df['schedule_adherence_ratio'] = (df['planned_duration_days'] / (df['actual_duration_days'] + 1.0)).clip(lower=0.0, upper=10.0)

    # 2. Labor & Execution Confirmations
    df['total_actual_duration_hrs'] = pd.to_numeric(df['total_actual_duration'], errors='coerce').fillna(0.0).clip(lower=0.0, upper=1000.0)
    df['total_labor_hours'] = pd.to_numeric(df['total_labor_hours'], errors='coerce').fillna(0.0).clip(lower=0.0, upper=1000.0)
    df['confirmation_count'] = pd.to_numeric(df['confirmation_count'], errors='coerce').fillna(0).astype(int).clip(lower=0, upper=100)
    df['has_confirmations'] = (df['confirmation_count'] > 0).astype(float)
    df['duration_per_confirmation'] = df['total_actual_duration_hrs'] / np.maximum(1, df['confirmation_count'])
    df['is_final_confirmed'] = df['final_conf_flag'].astype(str).str.upper().eq('X').astype(float)
    df['has_variance_reason'] = df['variance_reason_code'].astype(str).str.strip().ne('').astype(float)

    # 3. Spare Parts & Replacement Intensity
    df['spare_parts_item_count'] = pd.to_numeric(df['spare_parts_item_count'], errors='coerce').fillna(0).astype(int).clip(lower=0, upper=50)
    df['total_spare_parts_cost'] = pd.to_numeric(df['total_spare_parts_cost'], errors='coerce').fillna(0.0).clip(lower=0.0, upper=5000000.0)
    df['total_parts_quantity'] = pd.to_numeric(df['total_parts_quantity'], errors='coerce').fillna(0.0).clip(lower=0.0, upper=5000.0)
    df['cost_per_spare_part'] = df['total_spare_parts_cost'] / np.maximum(1, df['spare_parts_item_count'])
    df['has_spare_parts'] = (df['spare_parts_item_count'] > 0).astype(float)

    cost_q75 = df['total_spare_parts_cost'].quantile(0.75) if len(df) > 0 else 1000.0
    df['high_spare_cost_flag'] = (df['total_spare_parts_cost'] > max(5000.0, cost_q75)).astype(float)

    # 4. Equipment & Work Center Context
    wc_counts = df['work_center'].value_counts().to_dict()
    df['work_center_frequency'] = df['work_center'].map(wc_counts).fillna(1.0).astype(float)

    wc_bd_ratio = df.groupby('work_center')['order_type'].apply(lambda s: (s == 'PM03').mean()).to_dict()
    df['work_center_breakdown_ratio'] = df['work_center'].map(wc_bd_ratio).fillna(0.20).astype(float)

    wc_avg_cost = df.groupby('work_center')['total_spare_parts_cost'].mean().to_dict()
    df['work_center_avg_cost'] = df['work_center'].map(wc_avg_cost).fillna(0.0).astype(float)

    # Priority mapping
    prio_map = {'1': 1.0, '2': 2.0, '3': 3.0, '4': 4.0}
    df['priority_numeric'] = df['priority_code'].astype(str).map(prio_map).fillna(3.0)

    # Phase indicators
    df['is_phase_completed'] = df['phase_completed'].astype(str).str.upper().eq('X').astype(float)
    df['is_phase_closed'] = df['phase_closed'].astype(str).str.upper().eq('X').astype(float)
    df['has_long_text'] = df['long_text_flag'].astype(str).str.upper().eq('X').astype(float)

    # 5. NLP Maintenance Text Risk Signals
    text_corpus = (
        df['order_description'].fillna('') + ' ' +
        df['confirmation_text'].fillna('') + ' ' +
        df['spare_parts_descriptions'].fillna('')
    ).str.lower()

    pattern_vibration = r'(?:bearing|vibration|noise|sound|loose|shake|pulley|imbalance|play|eccentric)'
    pattern_thermal = r'(?:heat|overheat|hot|temperature|leak|seepage|seizure|smoke|burnt|burning|seal|gasket)'
    pattern_electrical = r'(?:motor|trip|tripping|tripped|cable|fuse|starter|coil|breaker|short|relay|spark|winding)'
    pattern_wear = r'(?:damage|wear|worn|crack|broken|wilted|bent|fracture|corrosion|rust|pitted|jam|stuck)'
    pattern_emergency = r'(?:breakdown|urgent|emergency|stop|stoppage|seized|failed|failure|abnormal|choke)'

    df['nlp_vibration_bearing_flag'] = text_corpus.str.contains(pattern_vibration, regex=True).astype(float)
    df['nlp_overheating_leakage_flag'] = text_corpus.str.contains(pattern_thermal, regex=True).astype(float)
    df['nlp_electrical_motor_flag'] = text_corpus.str.contains(pattern_electrical, regex=True).astype(float)
    df['nlp_wear_damage_flag'] = text_corpus.str.contains(pattern_wear, regex=True).astype(float)
    nlp_emergency = text_corpus.str.contains(pattern_emergency, regex=True).astype(float)

    df['composite_nlp_risk_score'] = (
        0.25 * df['nlp_vibration_bearing_flag'] +
        0.25 * df['nlp_overheating_leakage_flag'] +
        0.20 * df['nlp_electrical_motor_flag'] +
        0.20 * df['nlp_wear_damage_flag'] +
        0.10 * nlp_emergency
    )

    # 6. Post-Hoc Audit Benchmark Exception Label (Used Strictly for Evaluation/Ranking, Never for Training)
    # Failure exception occurs if breakdown maintenance PM03, or emergency failure stoppage,
    # or severe operational wear with high spare parts and duration spikes
    is_bd = df['order_type'].isin(['PM03', 'PM01'])
    severe_stress = (df['total_spare_parts_cost'] > 10000.0) & (df['duration_variance_days'] > 5.0)
    nlp_failure = (nlp_emergency == 1.0) & (df['composite_nlp_risk_score'] >= 0.40)

    df['is_equipment_failure_exception'] = (is_bd | severe_stress | nlp_failure).astype(int)

    return df


def prepare_train_test_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42) -> dict:
    """
    Prepares unsupervised training and testing splits.
    Applies RobustScaler to the 32 input features X.
    No labels Y are used during training.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df['is_equipment_failure_exception'].values

    # Handle any remaining NaN/Inf
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'X_train_df': X_train,
        'X_test_df': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_cols': FEATURE_COLUMNS
    }
