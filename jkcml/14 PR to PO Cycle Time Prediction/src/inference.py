"""
Inference & Excel Reporting Module - Use Case 14: PR to PO Cycle Time Prediction
Category: Regression
Objective: Estimate procurement processing time.

Handles scoring SAP Purchase Requisitions using the #1 Ranked Champion Regression Model
and generating executive, formatted 3-sheet Excel reports matching the enterprise SAP reference template:
- Sheet 1: ML_14_Enriched_Output(Predict)
- Sheet 2: Feature_Importance
- Sheet 3: Columns Ignored

Engineered for full multi-schema datasets (HANA_P2 + HANA_P3, 529,000+ rows) with
constant-memory streaming and strict missing value handling:
If data is present, writes it cleanly; if missing/null/nan, writes a clean blank cell.
"""
import os
import shutil
import datetime
import joblib
import pandas as pd
import numpy as np
import xlsxwriter


def score_with_single_model(df: pd.DataFrame, model, scaler, feature_cols: list) -> pd.DataFrame:
    """
    Applies the selected #1 Champion Regression ML model to predict:
    - predicted_cycle_time_days: estimated procurement processing turnaround in days (1 decimal)
    - actual_cycle_time_days: actual cycle turnaround in days (if closed, else NaN)
    - cycle_time_variance_days: variance (predicted - actual) in days (if closed, else NaN)
    - cycle_prediction_accuracy_pct: accuracy percentage on benchmark records (if closed, else NaN)
    - sla_status: operational lead time category
    - procurement_risk_tier: executive risk category (Critical Risk, High Risk, Medium Risk, Low Risk)
    """
    df_out = df.copy()

    X = df_out[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    df_out['predicted_cycle_time_days'] = np.round(np.clip(preds, 0.0, None), 1)

    # Actual and Variance: for open PRs, these must be NaN (exported as blank)
    has_actual = df_out['actual_cycle_time_days'].notna()
    df_out['cycle_time_variance_days'] = np.where(
        has_actual,
        np.round(df_out['predicted_cycle_time_days'] - df_out['actual_cycle_time_days'], 1),
        np.nan
    )

    # Accuracy % on known closed records
    denom = np.maximum(df_out['actual_cycle_time_days'], 1.0)
    abs_err = np.abs(df_out['predicted_cycle_time_days'] - df_out['actual_cycle_time_days'])
    df_out['cycle_prediction_accuracy_pct'] = np.where(
        has_actual,
        np.round(np.maximum(0.0, 100.0 - (abs_err / denom * 100.0)), 1),
        np.nan
    )

    # Operational SLA Status based on predicted turnaround days
    p_days = df_out['predicted_cycle_time_days']
    conditions_sla = [
        p_days < 7.0,
        (p_days >= 7.0) & (p_days <= 21.0),
        (p_days > 21.0) & (p_days <= 45.0)
    ]
    choices_sla = [
        'Expedited (< 7 Days)',
        'Standard On-Time (7 - 21 Days)',
        'Moderate Delay (21 - 45 Days)'
    ]
    df_out['sla_status'] = np.select(conditions_sla, choices_sla, default='Critical Bottleneck (> 45 Days)')

    # Executive Risk Tiers
    is_open = df_out['actual_cycle_time_days'].isna()
    has_block = df_out.get('has_release_block', pd.Series(0, index=df_out.index)) == 1
    del_flag = df_out.get('deletion_flag', pd.Series(0, index=df_out.index)) == 1
    pr_age = df_out.get('pr_age_days', pd.Series(0, index=df_out.index)) > 60
    no_vendor = df_out.get('has_assigned_vendor', pd.Series(1, index=df_out.index)) == 0

    conditions_risk = [
        (p_days > 45.0) | (is_open & (has_block | del_flag | pr_age)),
        (p_days >= 30.0) | (is_open & no_vendor),
        (p_days >= 15.0)
    ]
    choices_risk = ['Critical Risk', 'High Risk', 'Medium Risk']
    df_out['procurement_risk_tier'] = np.select(conditions_risk, choices_risk, default='Low Risk')

    # Sort descending by predicted cycle turnaround days and item monetary value
    df_out = df_out.sort_values(
        by=['predicted_cycle_time_days', 'pr_item_value'],
        ascending=[False, False]
    ).reset_index(drop=True)

    return df_out


def is_blank_val(val) -> bool:
    """Helper to detect missing/null/nan values across strings and floats."""
    if val is None:
        return True
    if isinstance(val, (float, np.floating)):
        return np.isnan(val) or np.isinf(val)
    if isinstance(val, (int, np.integer)):
        return False
    if isinstance(val, (datetime.date, datetime.datetime, pd.Timestamp)):
        return pd.isna(val)
    s = str(val).strip().lower()
    return s in ('', 'nan', 'none', 'nat', 'null', '<na>')


def format_date_str(val) -> str:
    """Format dates to YYYY-MM-DD or empty string if missing."""
    if is_blank_val(val):
        return ""
    if isinstance(val, (datetime.date, datetime.datetime, pd.Timestamp)):
        try:
            return val.strftime('%Y-%m-%d')
        except Exception:
            return ""
    s = str(val).strip()
    if len(s) >= 10 and s[4] == '-' and s[7] == '-':
        return s[:10]
    return s


def export_enriched_excel_report(
    df: pd.DataFrame,
    output_path: str = "ML_14_PR_to_PO_Cycle_Time_Prediction_Enriched_Output_Clean.xlsx",
    insight_id: str = "ML_14",
    use_case: str = "PR to PO Cycle Time Prediction",
    category: str = "Regression",
    objective: str = "Estimate procurement processing time.",
    feature_importance_df: pd.DataFrame = None
) -> str:
    """
    Generates the executive enriched Excel report matching the enterprise SAP reference template:
    - Sheet 1: ML_14_Enriched_Output(Predict)
    - Sheet 2: Feature_Importance
    - Sheet 3: Columns Ignored

    Uses XlsxWriter with constant_memory mode for fast streaming of 529,000+ rows.
    Implements strict missing value handling: if data is present, writes it cleanly;
    if missing/null/nan, writes a clean blank cell.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) if os.path.dirname(output_path) else ".", exist_ok=True)

    column_mapping = [
        ('company_code', 'Company Code'),
        ('company_name', 'Company Name'),
        ('company_city', 'Company City'),
        ('plant', 'Plant'),
        ('plant_name', 'Plant Name'),
        ('plant_city', 'Plant City'),
        ('pr_number', 'PR Number (BANFN)'),
        ('pr_item', 'PR Item (BNFPO)'),
        ('doc_type', 'PR Document Type (BSART)'),
        ('processing_status', 'Processing Status (STATU)'),
        ('deletion_indicator', 'Deletion Indicator (LOEKZ)'),
        ('release_indicator', 'Release Indicator (FRGKZ)'),
        ('release_group', 'Release Group (FRGGR)'),
        ('release_strategy', 'Release Strategy (FRGST)'),
        ('purchasing_group', 'Purchasing Group (EKGRP)'),
        ('requisitioner', 'Requisitioner (AFNAM)'),
        ('requisition_date', 'Requisition Date (BADAT)'),
        ('creation_date', 'Creation Date (ERDAT)'),
        ('delivery_date', 'Delivery Date (LFDAT)'),
        ('release_date', 'Release Date (FRGDT)'),
        ('po_number', 'PO Number (EBELN)'),
        ('po_item', 'PO Item (EBELP)'),
        ('po_date', 'PO Date (BEDAT)'),
        ('material_id', 'Material (MATNR)'),
        ('material_desc', 'Material Description'),
        ('material_type', 'Material Type (MTART)'),
        ('material_group', 'Material Group (MATKL)'),
        ('vendor_id', 'Vendor (LIFNR)'),
        ('vendor_name', 'Vendor Name'),
        ('quantity', 'Requisition Quantity'),
        ('unit_of_measure', 'Unit of Measure'),
        ('unit_price', 'Unit Price (PREIS)'),
        ('pr_item_value', 'Requisition Value (INR)'),
        ('planned_delivery_days', 'Planned Lead Time (Days)'),
        ('actual_cycle_time_days', 'Actual Cycle Time (Days)'),
        ('predicted_cycle_time_days', 'Est Procurement Cycle Time (Days)'),
        ('cycle_time_variance_days', 'Cycle Time Variance (Days)'),
        ('cycle_prediction_accuracy_pct', 'Cycle Prediction Accuracy (%)'),
        ('sla_status', 'SLA Status'),
        ('procurement_risk_tier', 'Procurement Risk Category')
    ]

    export_cols = [src for src, _ in column_mapping if src in df.columns]
    display_headers = [tgt for src, tgt in column_mapping if src in df.columns]
    df_export = df[export_cols].copy()

    # Create temporary file to ensure safe write without permission clashes
    temp_target = output_path + ".tmp.xlsx"
    workbook = xlsxwriter.Workbook(temp_target, {'constant_memory': True, 'strings_to_numbers': False})

    # ==================== FORMAT DEFINITIONS ====================
    # Metadata Formats
    fmt_meta_label = workbook.add_format({'font_name': 'Calibri', 'font_size': 10, 'bold': True, 'font_color': '#1F497D'})
    fmt_meta_val = workbook.add_format({'font_name': 'Calibri', 'font_size': 10, 'font_color': '#000000'})

    # Header Format
    fmt_header = workbook.add_format({
        'font_name': 'Calibri', 'font_size': 10, 'bold': True,
        'bg_color': '#F2F2F2', 'font_color': '#000000',
        'align': 'center', 'valign': 'vcenter',
        'border': 1, 'border_color': '#D3D3D3'
    })

    # Data Cell Formats
    border_kw = {'border': 1, 'border_color': '#D3D3D3', 'font_name': 'Calibri', 'font_size': 10, 'valign': 'vcenter'}
    fmt_text_left = workbook.add_format({**border_kw, 'align': 'left'})
    fmt_text_center = workbook.add_format({**border_kw, 'align': 'center'})
    fmt_text_right = workbook.add_format({**border_kw, 'align': 'right'})
    fmt_num_1dec = workbook.add_format({**border_kw, 'align': 'right', 'num_format': '#,##0.0'})
    fmt_num_2dec = workbook.add_format({**border_kw, 'align': 'right', 'num_format': '#,##0.00'})
    fmt_num_pct = workbook.add_format({**border_kw, 'align': 'right', 'num_format': '0.0"%"'})
    fmt_blank = workbook.add_format({**border_kw, 'align': 'center'})

    # Risk Formats
    fmt_risk_critical = workbook.add_format({**border_kw, 'align': 'center', 'bold': True, 'font_color': '#9C0006', 'bg_color': '#FFC7CE'})
    fmt_risk_high = workbook.add_format({**border_kw, 'align': 'center', 'bold': True, 'font_color': '#9C6500', 'bg_color': '#FFEB9C'})
    fmt_risk_med = workbook.add_format({**border_kw, 'align': 'center', 'font_color': '#006100', 'bg_color': '#C6EFCE'})
    fmt_risk_low = workbook.add_format({**border_kw, 'align': 'center', 'font_color': '#006100', 'bg_color': '#C6EFCE'})

    # ==================== SHEET 1: ML_14_Enriched_Output(Predict) ====================
    ws1 = workbook.add_worksheet(f"{insight_id}_Enriched_Output(Predict)")
    ws1.hide_gridlines(0)

    metadata_rows = [
        ("Insight ID", f": {insight_id}"),
        ("Use Case", f": {use_case}"),
        ("Category", f": {category}"),
        ("Objective", f": {objective}"),
        ("Generated On", f": {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
        ("Total Records Analyzed", f": {len(df_export):,} Requisitions")
    ]

    for r_idx, (label, val) in enumerate(metadata_rows):
        ws1.write(r_idx, 0, label, fmt_meta_label)
        ws1.write(r_idx, 1, val, fmt_meta_val)

    # Row 7 is blank index 6; Table Header at Row 8 (0-indexed row 7)
    header_row_idx = 7
    ws1.set_row(header_row_idx, 24)

    # Pre-determine column types for high-speed row writing
    col_types = []
    for h in display_headers:
        if "Accuracy" in h:
            col_types.append('pct')
        elif any(k in h for k in ["Variance", "Actual Cycle", "Est Procurement Cycle", "Planned Lead", "Quantity"]):
            col_types.append('num_1dec')
        elif any(k in h for k in ["Value", "Unit Price"]):
            col_types.append('num_2dec')
        elif "Risk Category" in h:
            col_types.append('risk')
        elif any(k in h for k in ["Date", "BADAT", "ERDAT", "LFDAT", "FRGDT", "BEDAT"]):
            col_types.append('date')
        elif any(k in h for k in ["Code", "Plant", "PR Number", "PR Item", "Document Type", "Status", "Indicator", "Group", "PO Number", "PO Item", "Material (", "Vendor (", "SLA Status"]):
            col_types.append('center')
        else:
            col_types.append('left')

    for col_idx, header_text in enumerate(display_headers):
        ws1.write(header_row_idx, col_idx, header_text, fmt_header)

    # Write Data Rows
    col_widths = [max(len(h) + 4, 12) for h in display_headers]

    print(f"[INFO] Streaming {len(df_export):,} rows into Excel sheet with strict blank-value handling...")
    data_records = df_export.to_dict(orient='records')

    for row_offset, row_dict in enumerate(data_records, start=8):
        for col_idx, (src_col, col_t) in enumerate(zip(export_cols, col_types)):
            val = row_dict[src_col]

            # Missing value check: if missing/null/nan -> write blank cell
            if is_blank_val(val):
                ws1.write_blank(row_offset, col_idx, None, fmt_blank)
                continue

            if col_t == 'pct':
                try:
                    num_val = float(val)
                    if np.isnan(num_val):
                        ws1.write_blank(row_offset, col_idx, None, fmt_blank)
                    else:
                        ws1.write_number(row_offset, col_idx, num_val, fmt_num_pct)
                except Exception:
                    ws1.write_blank(row_offset, col_idx, None, fmt_blank)

            elif col_t == 'num_1dec':
                try:
                    num_val = float(val)
                    if np.isnan(num_val):
                        ws1.write_blank(row_offset, col_idx, None, fmt_blank)
                    else:
                        ws1.write_number(row_offset, col_idx, num_val, fmt_num_1dec)
                except Exception:
                    ws1.write_blank(row_offset, col_idx, None, fmt_blank)

            elif col_t == 'num_2dec':
                try:
                    num_val = float(val)
                    if np.isnan(num_val):
                        ws1.write_blank(row_offset, col_idx, None, fmt_blank)
                    else:
                        ws1.write_number(row_offset, col_idx, num_val, fmt_num_2dec)
                except Exception:
                    ws1.write_blank(row_offset, col_idx, None, fmt_blank)

            elif col_t == 'risk':
                s_val = str(val).strip()
                if s_val == "Critical Risk":
                    ws1.write_string(row_offset, col_idx, s_val, fmt_risk_critical)
                elif s_val == "High Risk":
                    ws1.write_string(row_offset, col_idx, s_val, fmt_risk_high)
                elif s_val == "Medium Risk":
                    ws1.write_string(row_offset, col_idx, s_val, fmt_risk_med)
                else:
                    ws1.write_string(row_offset, col_idx, s_val, fmt_risk_low)

            elif col_t == 'date':
                d_str = format_date_str(val)
                if d_str == "":
                    ws1.write_blank(row_offset, col_idx, None, fmt_blank)
                else:
                    ws1.write_string(row_offset, col_idx, d_str, fmt_text_center)

            elif col_t == 'center':
                s_val = str(val).strip()
                if s_val.lower() in ('nan', 'none', 'nat', 'null', '<na>', ''):
                    ws1.write_blank(row_offset, col_idx, None, fmt_blank)
                else:
                    ws1.write_string(row_offset, col_idx, s_val, fmt_text_center)

            else:  # left
                s_val = str(val).strip()
                if s_val.lower() in ('nan', 'none', 'nat', 'null', '<na>', ''):
                    ws1.write_blank(row_offset, col_idx, None, fmt_blank)
                else:
                    ws1.write_string(row_offset, col_idx, s_val, fmt_text_left)

    # Set column widths and freeze panes at row 9 (below header row 8)
    for c_idx, width in enumerate(col_widths):
        ws1.set_column(c_idx, c_idx, min(max(width, 12), 42))

    ws1.freeze_panes(8, 0)

    # ==================== SHEET 2: Feature_Importance ====================
    if feature_importance_df is None and os.path.exists("reports/feature_importance.csv"):
        feature_importance_df = pd.read_csv("reports/feature_importance.csv")

    ws2 = workbook.add_worksheet("Feature_Importance")
    ws2.hide_gridlines(0)

    fmt_fi_title = workbook.add_format({'font_name': 'Calibri', 'font_size': 11, 'bold': True, 'font_color': '#000000'})
    fmt_fi_header = workbook.add_format({
        'font_name': 'Calibri', 'font_size': 11, 'bold': True,
        'bg_color': '#2F5597', 'font_color': '#FFFFFF',
        'align': 'left', 'valign': 'vcenter', 'border': 1, 'border_color': '#D3D3D3'
    })
    fmt_fi_header_right = workbook.add_format({
        'font_name': 'Calibri', 'font_size': 11, 'bold': True,
        'bg_color': '#2F5597', 'font_color': '#FFFFFF',
        'align': 'right', 'valign': 'vcenter', 'border': 1, 'border_color': '#D3D3D3'
    })
    fmt_fi_num = workbook.add_format({**border_kw, 'align': 'right', 'num_format': '0.000000000'})

    ws2.write(0, 0, f"{insight_id} | Feature_Importance ({category})", fmt_fi_title)
    ws2.write(1, 0, "feature", fmt_fi_header)
    ws2.write(1, 1, "importance", fmt_fi_header_right)
    ws2.write(1, 2, "Direct Mathematical Influence On Procurement Processing Time (Days):", fmt_fi_header)
    ws2.set_row(1, 22)

    if feature_importance_df is not None and len(feature_importance_df) > 0:
        col_f = 'Feature Name' if 'Feature Name' in feature_importance_df.columns else ('feature' if 'feature' in feature_importance_df.columns else feature_importance_df.columns[0])
        col_i = 'Importance Score' if 'Importance Score' in feature_importance_df.columns else ('importance' if 'importance' in feature_importance_df.columns else feature_importance_df.columns[1])
        if 'Business Impact & Procurement Rationale' in feature_importance_df.columns:
            col_e = 'Business Impact & Procurement Rationale'
        elif len(feature_importance_df.columns) > 3:
            col_e = feature_importance_df.columns[3]
        elif len(feature_importance_df.columns) > 2 and feature_importance_df.columns[2] not in (col_f, col_i):
            col_e = feature_importance_df.columns[2]
        else:
            col_e = None

        for idx, r_data in feature_importance_df.iterrows():
            curr_r = idx + 2
            ws2.write_string(curr_r, 0, str(r_data[col_f]), fmt_text_left)
            ws2.write_number(curr_r, 1, float(r_data[col_i]), fmt_fi_num)
            expl_str = str(r_data[col_e]) if col_e and pd.notna(r_data[col_e]) else ""
            ws2.write_string(curr_r, 2, expl_str, fmt_text_left)

    ws2.set_column(0, 0, 32)
    ws2.set_column(1, 1, 18)
    ws2.set_column(2, 2, 86)

    # ==================== SHEET 3: Columns Ignored ====================
    ws3 = workbook.add_worksheet("Columns Ignored")
    ws3.hide_gridlines(0)

    ws3.write(0, 0, "Columns Ignored / Excluded from Model Training:", fmt_fi_title)

    ign_headers = ["Ignored Column", "SAP Field Name", "Why It Was Ignored"]
    for c_idx, h_text in enumerate(ign_headers):
        ws3.write(2, c_idx, h_text, fmt_header)
    ws3.set_row(2, 20)

    ignored_rows = [
        ("PR Document & Item Number", "BANFN, BNFPO", "Sequential administrative SAP requisition document numbers. They possess no intrinsic statistical correlation with procurement processing turnaround."),
        ("Purchase Order Number & Item", "EBELN, EBELP", "Target conversion identifiers. In open requisitions awaiting cycle estimation, this is completely unpopulated and must not leak into features."),
        ("Requisitioner Name Text", "AFNAM", "Free-text user employee identifiers with thousands of unique variations; captured indirectly through purchasing group velocity."),
        ("Material Text Description", "MAKTX, TXZ01", "Free-form textual item description retained strictly for human audit review; material group and type capture the structural semantics."),
        ("Vendor & Plant Descriptive Names", "NAME1 (LFA1), NAME1 (T001W)", "Display names retained for executive reporting; categorical frequency features capture the plant and supplier operational density."),
        ("Company Code", "BUKRS (1000)", "Single operating corporate entity (J.K. CEMENT LIMITED) across the dataset with zero mathematical variance."),
        ("Company & Vendor City / Country", "ORT01, LAND1, REGIO", "Administrative geographic descriptors with minimal variance across domestic plant locations."),
        ("Currency Code", "WAERS", "Constant transactional currency (INR) across domestic plant procurement documents."),
        ("Raw Date Timestamps", "BADAT, ERDAT, LFDAT, FRGDT, BEDAT", "Raw calendar timestamps cannot be directly fed into regression estimators; transformed into lead time, delivery horizon, and calendar cyclical features."),
        ("Target Variable", "actual_cycle_time_days", "Target variable to predict (BEDAT - BADAT); isolated as y during supervised regression training to prevent target leakage.")
    ]

    for r_offset, (col_name, sap_field, reason) in enumerate(ignored_rows, start=3):
        ws3.write_string(r_offset, 0, col_name, fmt_text_left)
        ws3.write_string(r_offset, 1, sap_field, fmt_text_left)
        ws3.write_string(r_offset, 2, reason, fmt_text_left)

    ws3.set_column(0, 0, 30)
    ws3.set_column(1, 1, 28)
    ws3.set_column(2, 2, 98)

    workbook.close()

    # Move temp file to final target with fallback handling
    save_candidates = [
        output_path,
        "ML_14_PR_to_PO_Cycle_Time_Prediction_Enriched_Output_Clean.xlsx",
        "ML_14_PR_to_PO_Cycle_Time_Prediction_Enriched_Output.xlsx",
        os.path.join("reports", os.path.basename(output_path)),
        f"ML_14_PR_to_PO_Cycle_Time_Output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    ]

    saved = False
    actual_path = output_path
    for candidate in save_candidates:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(candidate)) if os.path.dirname(candidate) else ".", exist_ok=True)
            if os.path.exists(candidate):
                os.remove(candidate)
            shutil.copy2(temp_target, candidate)
            actual_path = candidate
            saved = True
            break
        except Exception as e:
            print(f"[WARNING] Could not save directly to '{candidate}': {e}. Trying next candidate...")
            continue

    if not saved:
        actual_path = f"ML_14_PR_to_PO_Output_{int(datetime.datetime.now().timestamp())}.xlsx"
        shutil.copy2(temp_target, actual_path)

    # Clean up temp target
    if os.path.exists(temp_target):
        try:
            os.remove(temp_target)
        except Exception:
            pass

    print(f"[SUCCESS] Executive Enriched Excel Report saved to: {actual_path}")
    print(f"[INFO] Total Rows: {len(df_export):,}, Total Columns: {len(display_headers)}")
    return actual_path


def save_single_model_pipeline(model, scaler, feature_cols, comparison_df=None, output_dir: str = "models"):
    """
    Saves the selected #1 Champion Regression Model pipeline and benchmark metadata to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    bundle = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'model_name': model.champion_name if hasattr(model, 'champion_name') else 'Champion Model',
        'benchmark_ranking': comparison_df.to_dict(orient='records') if comparison_df is not None else None
    }
    champ_path = os.path.join(output_dir, "champion_pr_cycle_regressor.joblib")
    joblib.dump(bundle, champ_path)
    print(f"[SUCCESS] Champion Regression Model pipeline successfully saved to: {champ_path}")
    return champ_path
