"""
Production Inference and Reporting Module - Use Case 4: Predictive Maintenance Classification
(Predict Equipment Failures Before Breakdown)
Applies the #1 Champion Model to score all extracted plant maintenance records, generates
prescriptive maintenance actions, handles missing values (leaving unpopulated fields blank),
and exports formatted executive multi-tab Excel workbooks and CSVs.
"""
import os
import shutil
import joblib
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


def score_with_single_model(champion_segmenter, full_featured_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores all maintenance records with the #1 Champion Model:
    - failure_risk_score [0.0 - 1.0]
    - predicted_failure_exception (1 = Failure Imminent / Exception, 0 = Normal)
    - equipment_health_tier (Critical Failure Imminent, High Breakdown Risk, Moderate Wear, Healthy)
    - risk_category (Critical, High, Medium, Low)
    - prescriptive_action
    """
    df = full_featured_df.copy()
    feature_cols = champion_segmenter.feature_cols

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_scaled = champion_segmenter.scaler.transform(X)

    scores = champion_segmenter.predict_risk_scores(X_scaled)
    preds = champion_segmenter.predict_exception(X_scaled)

    df['failure_risk_score'] = np.round(scores, 4)
    df['predicted_failure_exception'] = preds.astype(int)

    # Health Tiers
    df['equipment_health_tier'] = [champion_segmenter.assign_health_tier(s) for s in scores]

    # Risk Categories
    def assign_category(score):
        if score >= 0.70:
            return "Critical Risk"
        elif score >= 0.45:
            return "High Risk"
        elif score >= 0.25:
            return "Medium Risk"
        else:
            return "Low Risk"

    df['failure_risk_category'] = [assign_category(s) for s in scores]

    # Prescriptive Actions (optimized with zip for fast processing on 200k+ rows)
    actions = []
    vib_col = df['nlp_vibration_bearing_flag'].values
    therm_col = df['nlp_overheating_leakage_flag'].values
    elec_col = df['nlp_electrical_motor_flag'].values
    wear_col = df['nlp_wear_damage_flag'].values
    cost_col = df['total_spare_parts_cost'].values
    var_col = df['duration_variance_days'].values

    for s, vib, therm, elec, wear, cost, dur_v in zip(scores, vib_col, therm_col, elec_col, wear_col, cost_col, var_col):
        row_dict = {
            'failure_risk_score': s,
            'nlp_vibration_bearing_flag': vib,
            'nlp_overheating_leakage_flag': therm,
            'nlp_electrical_motor_flag': elec,
            'nlp_wear_damage_flag': wear,
            'total_spare_parts_cost': cost,
            'duration_variance_days': dur_v
        }
        actions.append(champion_segmenter.prescribe_action(row_dict))
    df['prescriptive_action'] = actions

    # Format human-readable output columns
    output_cols_lead = [
        'order_number',
        'order_type',
        'order_description',
        'work_center',
        'plant_code',
        'plant_name',
        'company_code',
        'company_name',
        'predicted_failure_exception',
        'failure_risk_score',
        'equipment_health_tier',
        'failure_risk_category',
        'prescriptive_action',
        'actual_duration_days',
        'planned_duration_days',
        'duration_variance_days',
        'total_spare_parts_cost',
        'spare_parts_item_count',
        'total_labor_hours',
        'confirmation_count',
        'composite_nlp_risk_score',
        'creation_date',
        'teco_date'
    ]

    existing_lead = [c for c in output_cols_lead if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_lead]
    df = df[existing_lead + other_cols]

    return df


def clean_missing_for_export(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans missing values so that:
    - If data is present, it is formatted properly.
    - If data is missing/null/nan/NaT, it is left completely blank (empty string).
    """
    clean_df = input_df.copy()

    # Clean date columns
    date_cols = [
        'creation_date', 'change_date', 'release_date', 'teco_date', 'close_date',
        'planned_start_date', 'planned_finish_date', 'actual_start_date',
        'actual_finish_date', 'confirmed_finish_date', 'sched_release_date',
        'exec_start_date', 'exec_end_date'
    ]
    for col in date_cols:
        if col in clean_df.columns:
            clean_df[col] = pd.to_datetime(clean_df[col], errors='coerce').dt.strftime('%Y-%m-%d').fillna('')

    # Clean string / text / object / float columns
    for col in clean_df.columns:
        if pd.api.types.is_float_dtype(clean_df[col]):
            clean_df[col] = clean_df[col].round(4)
        elif not pd.api.types.is_numeric_dtype(clean_df[col]):
            clean_df[col] = clean_df[col].fillna('').astype(str).replace(
                {'nan': '', 'None': '', 'NaT': '', '<NA>': '', 'null': '', 'NULL': ''}
            ).str.strip()

    return clean_df


def save_single_model_pipeline(champion_segmenter, model_path: str, scaler_path: str):
    """Serializes the champion model instance and scaler to disk."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    bundle = {
        'model_key': champion_segmenter.model_key,
        'model': champion_segmenter.model,
        'threshold': champion_segmenter.threshold,
        'meta': champion_segmenter.meta,
        'feature_cols': champion_segmenter.feature_cols
    }
    joblib.dump(bundle, model_path)
    joblib.dump(champion_segmenter.scaler, scaler_path)
    print(f"[INFO] Champion Model Bundle saved to: {model_path}")
    print(f"[INFO] Scaler saved to: {scaler_path}")


def export_enriched_excel_report(
    enriched_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    fi_df: pd.DataFrame,
    output_excel_path: str
):
    """
    Exports a professional multi-sheet Excel report:
    1. Executive_Summary
    2. Enriched_Predictions
    3. Failure_Watchlist
    4. Model_Comparison
    5. Feature_Importance
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_excel_path)), exist_ok=True)

    total_orders = len(enriched_df)
    total_exceptions = int(enriched_df['predicted_failure_exception'].sum())
    exception_rate = (total_exceptions / max(1, total_orders)) * 100.0
    total_spare_cost = float(enriched_df['total_spare_parts_cost'].sum())
    exception_spare_cost = float(enriched_df[enriched_df['predicted_failure_exception'] == 1]['total_spare_parts_cost'].sum())

    champion_row = comparison_df.iloc[0]
    champion_name = champion_row['Model']
    champion_acc = champion_row['Accuracy'] * 100.0
    champion_f1 = champion_row['F1-Score']
    champion_auc = champion_row['ROC-AUC']

    # 1. Summary sheet data
    summary_data = [
        ["Predictive Maintenance Machine Learning Executive Report", ""],
        ["Use Case", "4: Predictive Maintenance Classification (Predict Equipment Failures Before Breakdown)"],
        ["Learning Methodology", "100% Unsupervised Machine Learning (Input-Only Training, No Target Labels)"],
        ["Selected #1 Champion Model", champion_name],
        ["Champion Model Accuracy", f"{champion_acc:.2f}%"],
        ["Champion F1-Score", f"{champion_f1:.4f}"],
        ["Champion ROC-AUC", f"{champion_auc:.4f}"],
        ["", ""],
        ["Key Operational Metrics", "Value"],
        ["Total Plant Maintenance Orders Analyzed", f"{total_orders:,}"],
        ["Predicted Equipment Failure Exceptions", f"{total_exceptions:,}"],
        ["Predicted Exception Rate", f"{exception_rate:.2f}%"],
        ["Total Spare Parts Replacement Cost (INR)", f"Rs. {total_spare_cost:,.2f}"],
        ["High-Risk Failure Exception Spare Cost (INR)", f"Rs. {exception_spare_cost:,.2f}"],
        ["Potential Unplanned Breakdown Cost at Risk", f"Rs. {exception_spare_cost * 1.8:,.2f}"],
        ["", ""],
        ["Equipment Health Tier Breakdown", "Order Count"],
    ]

    tier_counts = enriched_df['equipment_health_tier'].value_counts()
    for tier, cnt in tier_counts.items():
        summary_data.append([tier, f"{cnt:,} ({(cnt/total_orders)*100:.1f}%)"])

    summary_df = pd.DataFrame(summary_data, columns=["KPI Indicator", "Value"])

    # Watchlist subset
    watchlist_raw = enriched_df[
        (enriched_df['predicted_failure_exception'] == 1) |
        (enriched_df['equipment_health_tier'].isin(['Critical Failure Imminent', 'High Breakdown Risk']))
    ].sort_values(by='failure_risk_score', ascending=False)

    # Clean missing values for display: if present put value, if missing put blank
    clean_enriched = clean_missing_for_export(enriched_df)
    clean_watchlist = clean_missing_for_export(watchlist_raw)

    print(f"[INFO] Writing multi-sheet Excel file to: {output_excel_path} ({total_orders:,} rows)...")
    # Save to Excel with openpyxl
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Executive_Summary', index=False)
        clean_enriched.to_excel(writer, sheet_name='Enriched_Predictions', index=False)
        clean_watchlist.to_excel(writer, sheet_name='Failure_Watchlist', index=False)
        comparison_df.to_excel(writer, sheet_name='Model_Comparison', index=False)
        fi_df.to_excel(writer, sheet_name='Feature_Importance', index=False)

    # Style workbook headers
    wb = openpyxl.load_workbook(output_excel_path)
    header_fill = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    thin_border = Border(
        left=Side(style='thin', color='D9D9D9'),
        right=Side(style='thin', color='D9D9D9'),
        top=Side(style='thin', color='D9D9D9'),
        bottom=Side(style='thin', color='D9D9D9')
    )

    for ws in wb.worksheets:
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")

        if ws.max_row <= 1000:
            for col in ws.columns:
                max_len = 0
                col_letter = get_column_letter(col[0].column)
                for cell in col:
                    cell.border = thin_border
                    val_str = str(cell.value or '')
                    if len(val_str) > max_len:
                        max_len = len(val_str)
                ws.column_dimensions[col_letter].width = min(42, max(max_len + 3, 11))
        else:
            for col in ws.iter_cols(min_row=1, max_row=min(50, ws.max_row)):
                col_letter = get_column_letter(col[0].column)
                max_len = max(len(str(cell.value or '')) for cell in col)
                ws.column_dimensions[col_letter].width = min(42, max(max_len + 3, 13))

    wb.save(output_excel_path)
    print(f"[INFO] Formatted Multi-Sheet Excel Report saved to: {output_excel_path}")
