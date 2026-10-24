"""
Inference & Excel Reporting Module - Use Case 13: Cost Center Spend Forecasting (Time Series)
Applies the #1 Ranked Champion Time-Series ML Model (Gradient Boosting Time-Series Forecaster)
to forecast departmental spending, quantify forecast variances, and generate executive 3-sheet
Excel reports matching the enterprise SAP template.
"""
import os
import datetime
import joblib
import pandas as pd
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


def score_with_single_model(df: pd.DataFrame, model, scaler, feature_cols: list) -> pd.DataFrame:
    """
    Applies the champion Time-Series forecasting model to compute:
    - forecasted_spend_inr: ML forecasted departmental spend in INR
    - spend_forecast_variance_inr: actual_spend_inr - forecasted_spend_inr
    - spend_forecast_variance_pct: percentage variance relative to forecast
    - forecast_accuracy_pct: sample-level forecasting accuracy % [0 - 100%]
    - spend_overrun_prob: continuous spend overrun probability in [0.0, 1.0] (8 decimals)
    - predicted_spend_exception: binary exception flag (1 for severe overrun, 0 for on-forecast)
    - risk_category: executive triage tier (Critical Risk, High Risk, Medium Risk, Low Risk)
    """
    df_out = df.copy()

    X = df_out[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_scaled = scaler.transform(X)

    if hasattr(model, 'predict'):
        raw_preds = model.predict(X_scaled)
    else:
        raw_preds = model.model.predict(X_scaled)

    forecasted = np.maximum(0.0, np.round(raw_preds, 2))
    df_out['forecasted_spend_inr'] = forecasted

    # Actual vs Forecasted Variance
    actual = df_out['actual_spend_inr'].values
    variance_inr = actual - forecasted
    safe_forecast = np.maximum(100.0, forecasted)

    variance_pct = (variance_inr / safe_forecast) * 100.0
    df_out['spend_forecast_variance_inr'] = np.round(variance_inr, 2)
    df_out['spend_forecast_variance_pct'] = np.round(np.clip(variance_pct, -100.0, 999.0), 2)

    # Individual Forecast Accuracy % = max(0, 1 - abs(actual - forecast) / max(actual, forecast)) * 100
    safe_denom = np.maximum(100.0, np.maximum(np.abs(actual), np.abs(forecasted)))
    acc_pct = np.clip(1.0 - (np.abs(actual - forecasted) / safe_denom), 0.0, 1.0) * 100.0
    df_out['forecast_accuracy_pct'] = np.round(acc_pct, 2)

    # Continuous Spend Overrun Probability via Sigmoid on relative variance and z-score
    z_signal = df_out['cc_spend_zscore'].values if 'cc_spend_zscore' in df_out.columns else np.zeros(len(df_out))
    rel_dev = np.clip((actual - forecasted) / safe_forecast, -2.0, 4.0)
    logit = 1.65 * rel_dev + 0.35 * z_signal - 0.20
    probs = 1.0 / (1.0 + np.exp(-logit))
    df_out['spend_overrun_prob'] = np.round(np.clip(probs, 0.00000001, 0.99999999), 8)

    # Predicted Spend Exception Flag
    df_out['predicted_spend_exception'] = (
        ((df_out['spend_forecast_variance_pct'] >= 35.0) & (df_out['actual_spend_inr'] >= 25000.0)) |
        (df_out['spend_overrun_prob'] >= 0.75)
    ).astype(int)

    # Executive Risk Category
    conditions = [
        (df_out['spend_forecast_variance_pct'] >= 50.0) & (df_out['actual_spend_inr'] >= 50000.0),
        (df_out['spend_forecast_variance_pct'] >= 30.0) & (df_out['actual_spend_inr'] >= 20000.0),
        (df_out['spend_forecast_variance_pct'] >= 15.0) | (df_out['spend_overrun_prob'] >= 0.60),
    ]
    choices = ['Critical Risk', 'High Risk', 'Medium Risk']
    df_out['risk_category'] = np.select(conditions, choices, default='Low Risk')

    # Sort so highest overrun probability and largest spend variance appear at the top
    df_out = df_out.sort_values(
        by=['spend_overrun_prob', 'spend_forecast_variance_inr', 'actual_spend_inr'],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return df_out


def export_enriched_excel_report(
    df: pd.DataFrame,
    output_path: str = "ML_13_Cost_Center_Spend_Forecasting_Enriched_Output.xlsx",
    insight_id: str = "ML_13",
    use_case: str = "Cost Center Spend Forecasting",
    description: str = "Forecast departmental spending using Time-Series ML models (Gradient Boosting, Random Forest, Bayesian AutoRegressive, Ridge, Holt-Winters EWMA) trained on 2025 historical spend to forecast 2026 expenditure across SAP Cost Centers (RSEG/RBKP/EKKN), Cost Center Master (CSKS), G/L Cost Elements (SKAT/SKA1), and Maintenance Orders (AUFK).",
    feature_importance_df: pd.DataFrame = None
) -> str:
    """
    Generates the executive enriched Excel report matching the enterprise SAP reference template:
    - Sheet 1: ML_13_Enriched_Output(Predict)
      - Rows 1-5: Metadata block (Insight ID, Use Case, Description, Generated On, Row Count)
      - Row 6: Blank row
      - Row 7: Enterprise SAP column headers
      - Row 8+: Enriched SAP Cost Center Spend records with 2025-2026 actual spend, ML forecasted spend, forecast variance, accuracy %, overrun probabilities, and risk tiers
    - Sheet 2: Feature_Importance (All 32 engineered Time-Series features with mathematical rationale)
    - Sheet 3: Columns Ignored (Audit justification for excluded non-feature columns)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    column_mapping = [
        ('company_code', 'Company Code (BUKRS)'),
        ('company_name', 'Company Name (BUTXT)'),
        ('doc_number', 'Accounting Doc (BELNR)'),
        ('fiscal_year', 'Fiscal Year (GJAHR)'),
        ('line_item', 'Line Item (BUZEI)'),
        ('po_number', 'Purchase Order (EBELN)'),
        ('po_item', 'PO Item (EBELP)'),
        ('cost_center', 'Cost Center (KOSTL)'),
        ('cost_center_name', 'Cost Center Name (CSKS)'),
        ('department_code', 'Cost Center Category/Dept'),
        ('plant', 'Plant (WERKS)'),
        ('plant_name', 'Plant Name (T001W)'),
        ('gl_account', 'G/L Cost Element (SAKNR)'),
        ('gl_description', 'G/L Cost Element Description (SKAT)'),
        ('department_spend_category', 'Departmental Spend Category'),
        ('order_id', 'Order ID (AUFNR)'),
        ('order_desc', 'Order Description (AUFK)'),
        ('posting_date', 'Posting Date (BUDAT)'),
        ('year_month', 'Year-Month Period'),
        ('posting_year', 'Posting Year'),
        ('debit_credit_ind', 'Debit/Credit (SHKZG)'),
        ('currency', 'Currency (WAERS)'),
        ('line_item_text', 'Line Item Text (SGTXT)'),
        ('vendor_code', 'Vendor Code (LIFNR)'),
        ('vendor_name', 'Vendor Name (LFA1)'),
        ('invoice_ref', 'Invoice Reference (XBLNR)'),
        ('actual_spend_inr', 'Actual Spend INR (WRBTR)'),
        ('forecasted_spend_inr', 'Forecasted Spend INR (ML)'),
        ('spend_forecast_variance_inr', 'Spend Forecast Variance INR'),
        ('spend_forecast_variance_pct', 'Spend Forecast Variance (%)'),
        ('forecast_accuracy_pct', 'Forecast Accuracy (%)'),
        ('lag_1_month_spend_inr', 'Lag-1 Month Spend INR (t-1)'),
        ('rolling_mean_3m_spend_inr', 'Rolling 3M Mean Spend INR'),
        ('ewma_spend_inr', 'EWMA Trend Spend INR'),
        ('cc_mom_spend_growth_pct', 'MoM Spend Growth (%)'),
        ('cc_seasonality_index', 'Seasonality Index'),
        ('cc_po_commitment_value_inr', 'Open PO Commitments INR (EKKN)'),
        ('cc_pm_order_count', 'Maintenance Orders (AUFK)'),
        ('spend_overrun_prob', 'Spend Overrun Exception Prob'),
        ('predicted_spend_exception', 'Predicted Spend Exception'),
        ('risk_category', 'Risk Category')
    ]

    export_cols = []
    display_headers = []
    for internal_col, display_name in column_mapping:
        if internal_col in df.columns:
            export_cols.append(internal_col)
            display_headers.append(display_name)

    df_export = df[export_cols].copy()

    for dt_col in ['posting_date']:
        if dt_col in df_export.columns:
            df_export[dt_col] = pd.to_datetime(df_export[dt_col], errors='coerce').dt.strftime('%d-%m-%Y').fillna('')

    for num_col in [
        'actual_spend_inr', 'forecasted_spend_inr', 'spend_forecast_variance_inr',
        'spend_forecast_variance_pct', 'forecast_accuracy_pct', 'lag_1_month_spend_inr',
        'rolling_mean_3m_spend_inr', 'ewma_spend_inr', 'cc_mom_spend_growth_pct',
        'cc_seasonality_index', 'cc_po_commitment_value_inr'
    ]:
        if num_col in df_export.columns:
            df_export[num_col] = df_export[num_col].round(2)

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "ML_13_Enriched_Output(Predict)"

    font_meta_label = Font(name="Calibri", size=11, bold=True, color="000000")
    font_meta_val = Font(name="Calibri", size=11, bold=False, color="000000")

    gen_time_str = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
    row_count_str = f"{len(df_export):,}"

    metadata_rows = [
        ("Insight ID", insight_id),
        ("Use Case", use_case),
        ("Description", description),
        ("Generated On", gen_time_str),
        ("Row Count", row_count_str)
    ]

    for r_idx, (label, val) in enumerate(metadata_rows, start=1):
        cell_lbl = ws.cell(row=r_idx, column=1, value=label)
        cell_val = ws.cell(row=r_idx, column=2, value=val)
        cell_lbl.font = font_meta_label
        cell_val.font = font_meta_val

    # Row 6 left blank intentionally as per enterprise SAP template

    header_row_idx = 7
    font_header = Font(name="Calibri", size=10, bold=True, color="000000")
    fill_header = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
    thin_side = Side(border_style="thin", color="D3D3D3")
    border_header = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)

    for col_idx, header_text in enumerate(display_headers, start=1):
        cell = ws.cell(row=header_row_idx, column=col_idx, value=header_text)
        cell.font = font_header
        cell.fill = fill_header
        cell.border = border_header
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=False)

    ws.row_dimensions[header_row_idx].height = 24

    font_data = Font(name="Calibri", size=10, color="000000")
    data_values = df_export.values.tolist()

    for r_offset, row_vals in enumerate(data_values, start=8):
        for c_idx, val in enumerate(row_vals, start=1):
            cell = ws.cell(row=r_offset, column=c_idx, value=val)
            cell.font = font_data
            cell.border = border_header

            header_name = display_headers[c_idx - 1]
            if "Prob" in header_name:
                cell.alignment = Alignment(horizontal="right")
                cell.number_format = "0.00000000"
            elif any(k in header_name for k in ["Spend", "Variance", "Growth", "Index", "Commitments", "Accuracy"]):
                cell.alignment = Alignment(horizontal="right")
                if isinstance(val, (int, float)):
                    cell.number_format = "#,##0.00"
            elif any(k in header_name for k in ["Orders", "Year", "Line Item", "Exception", "PO Item"]):
                cell.alignment = Alignment(horizontal="center")
            elif header_name in [
                "Company Code (BUKRS)", "Accounting Doc (BELNR)", "Fiscal Year (GJAHR)",
                "Cost Center (KOSTL)", "Plant (WERKS)", "G/L Cost Element (SAKNR)",
                "Posting Date (BUDAT)", "Year-Month Period", "Debit/Credit (SHKZG)",
                "Currency (WAERS)", "Vendor Code (LIFNR)", "Purchase Order (EBELN)"
            ]:
                cell.alignment = Alignment(horizontal="center")
            elif header_name == "Risk Category":
                cell.alignment = Alignment(horizontal="center")
                if val == "Critical Risk":
                    cell.font = Font(name="Calibri", size=10, bold=True, color="9C0006")
                    cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
                elif val == "High Risk":
                    cell.font = Font(name="Calibri", size=10, bold=True, color="9C6500")
                    cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
                elif val == "Medium Risk":
                    cell.font = Font(name="Calibri", size=10, color="006100")
                    cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")

    for c_idx, header_text in enumerate(display_headers, start=1):
        col_letter = get_column_letter(c_idx)
        max_len = len(str(header_text))
        for r_sample in range(8, min(108, len(data_values) + 8)):
            val = ws.cell(row=r_sample, column=c_idx).value
            if val is not None:
                max_len = max(max_len, len(str(val)))
        ws.column_dimensions[col_letter].width = min(max(max_len + 3, 11), 44)

    ws.freeze_panes = "A8"

    # -------------------------------------------------------------
    # Sheet 2: Feature_Importance
    # -------------------------------------------------------------
    if feature_importance_df is None and os.path.exists("reports/feature_importance.csv"):
        feature_importance_df = pd.read_csv("reports/feature_importance.csv")

    if feature_importance_df is not None and len(feature_importance_df) > 0:
        ws_feat = wb.create_sheet(title="Feature_Importance")

        cell_title = ws_feat.cell(row=1, column=1, value=f"{insight_id} | Feature_Importance")
        cell_title.font = Font(name="Calibri", size=11, bold=True, color="000000")

        c_feat = ws_feat.cell(row=2, column=1, value="feature")
        c_imp = ws_feat.cell(row=2, column=2, value="importance")
        c_expl = ws_feat.cell(row=2, column=3, value="Direct Mathematical Influence On Cost Center Spend Forecasting & Budget Governance:")

        font_feat_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
        fill_feat_header = PatternFill(start_color="2F5597", end_color="2F5597", fill_type="solid")

        c_feat.font = font_feat_header
        c_feat.fill = fill_feat_header
        c_feat.border = border_header
        c_feat.alignment = Alignment(horizontal="left", vertical="center")

        c_imp.font = font_feat_header
        c_imp.fill = fill_feat_header
        c_imp.border = border_header
        c_imp.alignment = Alignment(horizontal="right", vertical="center")

        c_expl.font = font_feat_header
        c_expl.fill = fill_feat_header
        c_expl.border = border_header
        c_expl.alignment = Alignment(horizontal="left", vertical="center")

        ws_feat.row_dimensions[2].height = 22

        col_feat_name = 'feature' if 'feature' in feature_importance_df.columns else feature_importance_df.columns[0]
        col_imp_name = 'importance' if 'importance' in feature_importance_df.columns else feature_importance_df.columns[1]
        has_expl = len(feature_importance_df.columns) > 2
        col_expl_name = feature_importance_df.columns[2] if has_expl else None

        for idx, r_data in feature_importance_df.iterrows():
            curr_row = idx + 3
            cell_f = ws_feat.cell(row=curr_row, column=1, value=str(r_data[col_feat_name]))
            cell_f.font = font_data
            cell_f.border = border_header
            cell_f.alignment = Alignment(horizontal="left", vertical="center")

            imp_val = float(r_data[col_imp_name])
            cell_i = ws_feat.cell(row=curr_row, column=2, value=imp_val)
            cell_i.font = font_data
            cell_i.border = border_header
            cell_i.alignment = Alignment(horizontal="right", vertical="center")
            cell_i.number_format = "0.000000000"

            expl_val = str(r_data[col_expl_name]) if col_expl_name and pd.notna(r_data[col_expl_name]) else ""
            cell_e = ws_feat.cell(row=curr_row, column=3, value=expl_val)
            cell_e.font = font_data
            cell_e.border = border_header
            cell_e.alignment = Alignment(horizontal="left", vertical="center")

        ws_feat.column_dimensions['A'].width = 38
        ws_feat.column_dimensions['B'].width = 18
        ws_feat.column_dimensions['C'].width = 92

    # -------------------------------------------------------------
    # Sheet 3: Columns Ignored
    # -------------------------------------------------------------
    ws_ign = wb.create_sheet(title="Columns Ignored")

    cell_ign_title = ws_ign.cell(row=1, column=1, value="Columns Ignored / Excluded from Model Training:")
    cell_ign_title.font = Font(name="Calibri", size=11, bold=True, color="000000")

    ign_headers = ["Ignored Column", "SAP Field Name", "Why It Was Ignored"]
    for c_idx, h_text in enumerate(ign_headers, start=1):
        c_cell = ws_ign.cell(row=3, column=c_idx, value=h_text)
        c_cell.font = Font(name="Calibri", size=10, bold=True, color="000000")
        c_cell.border = border_header
        c_cell.alignment = Alignment(horizontal="left", vertical="center")

    ws_ign.row_dimensions[3].height = 20

    ignored_rows = [
        ("Accounting Document Primary Keys", "BELNR, GJAHR, BUZEI, XBLNR", "Unique SAP invoice document and line item identifiers. Contain no temporal or economic predictive signal."),
        ("Organizational Identifiers", "BUKRS, KOSTL, WERKS, SAKNR, LIFNR, EBELN, EBELP", "Cost center, company code, plant, and vendor IDs retained for departmental grouping and drill-down; not raw numerical regressors."),
        ("Raw Descriptive Text Strings", "TXT20, TXT50, SGTXT, NAME1, Name, Description, KTEXT", "Unstructured text descriptions mapped into structured departmental spend categories; raw text strings excluded from numeric ML regressor."),
        ("Raw Calendar Date Strings", "BUDAT, BLDAT", "Raw date strings partitioned into training/forecast horizons and transformed into cyclical harmonic Fourier features and posting lag days."),
        ("Currency & Posting Indicators", "WAERS, SHKZG", "Categorical indicators already accounted for in directional sign and local currency conversion."),
        ("Direct Target & Variance Labels", "actual_spend_inr, forecasted_spend_inr, spend_forecast_variance_inr", "Actual spend is the forecast target variable (Y); variance fields computed post-prediction to prevent target leakage.")
    ]

    for r_offset, (col_name, sap_field, reason) in enumerate(ignored_rows, start=4):
        c1 = ws_ign.cell(row=r_offset, column=1, value=col_name)
        c2 = ws_ign.cell(row=r_offset, column=2, value=sap_field)
        c3 = ws_ign.cell(row=r_offset, column=3, value=reason)

        for c_cell in [c1, c2, c3]:
            c_cell.font = font_data
            c_cell.border = border_header
            c_cell.alignment = Alignment(horizontal="left", vertical="center")

    ws_ign.column_dimensions['A'].width = 34
    ws_ign.column_dimensions['B'].width = 30
    ws_ign.column_dimensions['C'].width = 98

    actual_path = output_path
    save_candidates = [
        output_path,
        "ML_13_Cost_Center_Spend_Forecasting_Enriched_Output_Clean.xlsx",
        "ML_13_Cost_Center_Spend_Forecasting_Enriched_Output_Final.xlsx",
        os.path.join("reports", os.path.basename(output_path)),
        f"ML_13_Cost_Center_Spend_Forecasting_Enriched_Output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    ]

    saved = False
    for candidate in save_candidates:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(candidate)), exist_ok=True)
            wb.save(candidate)
            actual_path = candidate
            saved = True
            break
        except PermissionError:
            print(f"[WARNING] '{candidate}' is currently open in Excel / locked. Trying next file...")
            continue

    if not saved:
        fallback = f"ML_13_Cost_Center_Spend_Output_{int(datetime.datetime.now().timestamp())}.xlsx"
        wb.save(fallback)
        actual_path = fallback

    print(f"[SUCCESS] Executive Enriched Excel Report saved to: {actual_path}")
    print(f"[INFO] Total Rows: {len(df_export):,}, Total Columns: {len(display_headers)}, Sheets: {wb.sheetnames}")
    return actual_path


def save_single_model_pipeline(model, scaler, feature_cols, suite=None, comparison_df=None, output_dir: str = "models"):
    """
    Saves the selected #1 Champion Time-Series Forecasting Model, scaler, and 5-model benchmark bundle to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    champ_name = getattr(model, 'champion_name', 'Gradient Boosting Time-Series Forecaster')
    actual_model = getattr(model, 'model', model)

    bundle = {
        'model': actual_model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'model_name': champ_name,
        'benchmark_ranking': comparison_df.to_dict(orient='records') if comparison_df is not None else None
    }
    champ_path = os.path.join(output_dir, "cost_center_spend_champion_model.joblib")
    bundle_path = os.path.join(output_dir, "cost_center_spend_model_bundle.joblib")
    scaler_path = os.path.join(output_dir, "robust_scaler.joblib")

    joblib.dump(bundle, champ_path)
    joblib.dump(bundle, bundle_path)
    joblib.dump(scaler, scaler_path)

    # Save individual models if available
    if suite is not None:
        if 'Random Forest Time-Series Forecaster' in suite.models:
            joblib.dump(suite.models['Random Forest Time-Series Forecaster'], os.path.join(output_dir, "champion_rf.joblib"))
        if 'Gradient Boosting Time-Series Forecaster' in suite.models:
            joblib.dump(suite.models['Gradient Boosting Time-Series Forecaster'], os.path.join(output_dir, "challenger_gbr.joblib"))

    print(f"[SUCCESS] Saved champion Time-Series Cost Center Spend model bundle to: {champ_path}")
