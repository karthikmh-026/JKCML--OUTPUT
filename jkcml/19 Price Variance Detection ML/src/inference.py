"""
Inference & Excel Reporting Module
Handles scoring procurement transactions using the single champion ML model (Isolation Forest)
and generating executive, formatted Excel reports matching the enterprise SAP template.
Uses high-performance xlsxwriter with constant-memory streaming for hundreds of thousands of rows.
"""
import os
import datetime
import joblib
import pandas as pd
import numpy as np
import xlsxwriter


def score_with_single_model(df: pd.DataFrame, model, scaler, feature_cols: list) -> pd.DataFrame:
    """
    Applies the single champion ML model (Isolation Forest) to calculate:
    - price_variance_prob: continuous anomaly probability in [0.0, 1.0] (8 decimal precision)
    - predicted_variance: binary flag (1 for unusual price anomaly, 0 for normal)
    - risk_category: executive risk triage (Critical Risk, High Risk, Medium Risk, Low Risk)
    """
    df_out = df.copy()

    # Prepare feature matrix (fill NaNs for model calculation only)
    X = df_out[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_scaled = scaler.transform(X)

    # Inference using SingleModelPriceVarianceDetector
    probs, raw_scores = model.predict_proba(X_scaled)
    preds = model.predict(X_scaled)

    df_out['price_variance_prob'] = np.round(probs, 8)
    df_out['predicted_variance'] = preds.astype(int)

    # Executive Risk Triage rules
    conditions = [
        (df_out['price_variance_prob'] >= 0.85) | ((df_out['predicted_variance'] == 1) & (df_out['price_variance_pct'] >= 40.0)),
        (df_out['price_variance_prob'] >= 0.70) | ((df_out['predicted_variance'] == 1) & (df_out['price_variance_pct'] >= 20.0)),
        (df_out['price_variance_prob'] >= 0.50) | (df_out['predicted_variance'] == 1),
    ]
    choices = ['Critical Risk', 'High Risk', 'Medium Risk']
    df_out['risk_category'] = np.select(conditions, choices, default='Low Risk')

    # Sort descending so critical risks and largest percentage/amount variances appear at the top
    df_out = df_out.sort_values(
        by=['price_variance_prob', 'price_variance_pct', 'price_variance_abs'],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return df_out


def export_enriched_excel_report(
    df: pd.DataFrame,
    output_path: str = "ML_19_Price_Variance_Detection_Enriched_Output.xlsx",
    insight_id: str = "ML_19",
    use_case: str = "Price Variance Detection",
    description: str = "Flag unusual purchase prices, enriched with purchasing document (EKPO, EKKO), info records (EINA, EINE), material master (MARA, MAKT), vendor master (LFA1), and plant (T001W).",
    feature_importance_df: pd.DataFrame = None
) -> str:
    """
    Generates the executive enriched Excel report matching the reference screenshots:
    - Sheet 1: ML_19_Enriched_Output(Predict)
      - Rows 1-5: Metadata block
      - Row 6: Blank row
      - Row 7: Enterprise column headers (all 38 columns)
      - Row 8+: Enriched procurement data, missing values written as clean blank cells
      - Frozen panes at row 8, auto-fit column widths, risk color fills
    - Sheet 2: Feature_Importance (32 domain features, 9 decimal places, blue banner)
    - Sheet 3: Columns Ignored (11 business reasons explaining excluded non-feature fields)
    Uses streaming xlsxwriter for high speed and minimal memory footprint.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    column_mapping = [
        ('company_code', 'Company Code'),
        ('company_name', 'Company Name'),
        ('company_city', 'Company City'),
        ('company_country', 'Company Country'),
        ('company_currency', 'Company Currency'),
        ('vendor_id', 'Vendor'),
        ('vendor_name', 'Vendor Name'),
        ('vendor_city', 'Vendor City'),
        ('vendor_region', 'Vendor Region'),
        ('vendor_country', 'Vendor Country'),
        ('plant', 'Plant'),
        ('plant_name', 'Plant Name'),
        ('plant_city', 'Plant City'),
        ('material_id', 'Material'),
        ('material_desc', 'Material Description'),
        ('material_type', 'Material Type'),
        ('material_group', 'Material Group'),
        ('po_number', 'Document Number'),
        ('po_item', 'Item Number'),
        ('doc_type', 'Document Type'),
        ('po_date', 'PO Date'),
        ('order_quantity', 'Order Quantity'),
        ('unit_of_measure', 'Order Unit'),
        ('po_unit_price', 'Net Price'),
        ('po_price_unit', 'Price Unit'),
        ('effective_po_price', 'Effective PO Price'),
        ('info_record_id', 'Info Record Number'),
        ('info_rec_price', 'Info Record Net Price'),
        ('info_rec_price_unit', 'Info Record Price Unit'),
        ('effective_info_price', 'Effective Info Price'),
        ('reference_price', 'Reference Price'),
        ('price_variance_abs', 'Price Variance (Amount)'),
        ('price_variance_pct', 'Price Variance (%)'),
        ('po_net_value', 'Amount (Local Currency)'),
        ('currency', 'Currency'),
        ('price_variance_prob', 'Price Variance Probability'),
        ('predicted_variance', 'Predicted Price Variance'),
        ('risk_category', 'Risk Category')
    ]

    export_cols = []
    display_headers = []
    for internal_col, display_name in column_mapping:
        if internal_col in df.columns:
            export_cols.append(internal_col)
            display_headers.append(display_name)

    df_export = df[export_cols].copy()

    # Clean date format
    if 'po_date' in df_export.columns:
        df_export['po_date'] = pd.to_datetime(df_export['po_date'], errors='coerce').dt.strftime('%d-%m-%Y 00:00')

    # Round numerical fields cleanly
    for num_col in ['order_quantity', 'po_unit_price', 'effective_po_price', 'info_rec_price',
                    'effective_info_price', 'reference_price', 'price_variance_abs', 'po_net_value',
                    'price_variance_pct']:
        if num_col in df_export.columns:
            df_export[num_col] = pd.to_numeric(df_export[num_col], errors='coerce').round(2)

    # Determine safe output path in case target is locked
    save_candidates = [
        output_path,
        "ML_19_Price_Variance_Detection_Enriched_Output_Clean.xlsx",
        "ML_19_Price_Variance_Detection_Enriched_Output_Final.xlsx",
        os.path.join("reports", os.path.basename(output_path)),
        f"ML_19_Price_Variance_Detection_Enriched_Output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    ]

    target_path = output_path
    for candidate in save_candidates:
        try:
            test_dir = os.path.dirname(os.path.abspath(candidate))
            os.makedirs(test_dir, exist_ok=True)
            if os.path.exists(candidate):
                with open(candidate, 'r+b'):
                    pass
            target_path = candidate
            break
        except (PermissionError, OSError):
            continue

    print(f"[INFO] Streaming {len(df_export):,} rows into Excel workbook: {target_path}...")
    wb = xlsxwriter.Workbook(target_path, {'constant_memory': True})

    # ==========================================
    # STYLES & FORMATS
    # ==========================================
    border_color = '#D3D3D3'
    fmt_meta_lbl = wb.add_format({'bold': True, 'font_name': 'Calibri', 'font_size': 11, 'font_color': '#000000'})
    fmt_meta_val = wb.add_format({'bold': False, 'font_name': 'Calibri', 'font_size': 11, 'font_color': '#000000'})

    fmt_header = wb.add_format({
        'bold': True, 'font_name': 'Calibri', 'font_size': 10, 'font_color': '#000000',
        'bg_color': '#F2F2F2', 'border': 1, 'border_color': border_color,
        'align': 'center', 'valign': 'vcenter'
    })

    fmt_blank = wb.add_format({'border': 1, 'border_color': border_color})
    fmt_text = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'border_color': border_color, 'align': 'left', 'valign': 'vcenter'})
    fmt_center = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'border_color': border_color, 'align': 'center', 'valign': 'vcenter'})
    fmt_num = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'border_color': border_color, 'align': 'right', 'valign': 'vcenter', 'num_format': '#,##0.00'})
    fmt_prob = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'border_color': border_color, 'align': 'right', 'valign': 'vcenter', 'num_format': '0.00000000'})

    # Risk formats
    fmt_risk_critical = wb.add_format({
        'font_name': 'Calibri', 'font_size': 10, 'bold': True, 'font_color': '#9C0006',
        'bg_color': '#FFC7CE', 'border': 1, 'border_color': border_color, 'align': 'center', 'valign': 'vcenter'
    })
    fmt_risk_high = wb.add_format({
        'font_name': 'Calibri', 'font_size': 10, 'bold': True, 'font_color': '#9C6500',
        'bg_color': '#FFEB9C', 'border': 1, 'border_color': border_color, 'align': 'center', 'valign': 'vcenter'
    })
    fmt_risk_medium = wb.add_format({
        'font_name': 'Calibri', 'font_size': 10, 'font_color': '#006100',
        'bg_color': '#C6EFCE', 'border': 1, 'border_color': border_color, 'align': 'center', 'valign': 'vcenter'
    })
    fmt_risk_low = fmt_center

    # ==========================================
    # SHEET 1: ML_19_Enriched_Output(Predict)
    # ==========================================
    ws1 = wb.add_worksheet("ML_19_Enriched_Output(Predict)")
    ws1.set_row(6, 24)

    # Rows 1-5: Metadata
    gen_time_str = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
    ws1.write(0, 0, "Insight ID", fmt_meta_lbl)
    ws1.write(0, 1, insight_id, fmt_meta_val)
    ws1.write(1, 0, "Use Case", fmt_meta_lbl)
    ws1.write(1, 1, use_case, fmt_meta_val)
    ws1.write(2, 0, "Description", fmt_meta_lbl)
    ws1.write(2, 1, description, fmt_meta_val)
    ws1.write(3, 0, "Generated On", fmt_meta_lbl)
    ws1.write(3, 1, gen_time_str, fmt_meta_val)
    ws1.write(4, 0, "Row Count", fmt_meta_lbl)
    ws1.write(4, 1, f"{len(df_export):,}", fmt_meta_val)

    # Row 7: Header row
    for c_idx, h_text in enumerate(display_headers):
        ws1.write(6, c_idx, h_text, fmt_header)

    # Helper function to test if a value is missing
    def is_missing(val):
        if val is None:
            return True
        if pd.isna(val):
            return True
        s = str(val).strip()
        if s == '' or s.lower() in ('nan', 'none', '<na>', 'nat', 'null'):
            return True
        return False

    center_headers = {"Company Code", "Vendor", "Plant", "Material", "Document Number", "Item Number", "Predicted Price Variance", "PO Date"}
    num_headers = {"Order Quantity", "Net Price", "Price Unit", "Effective PO Price", "Info Record Net Price",
                   "Info Record Price Unit", "Effective Info Price", "Reference Price", "Price Variance (Amount)",
                   "Price Variance (%)", "Amount (Local Currency)"}

    # Data rows (Row 8 onwards)
    data_matrix = df_export.values
    for r_idx, row_vals in enumerate(data_matrix, start=7):
        for c_idx, val in enumerate(row_vals):
            h_name = display_headers[c_idx]

            if is_missing(val):
                ws1.write_blank(r_idx, c_idx, None, fmt_blank)
                continue

            if h_name == "Risk Category":
                s_val = str(val).strip()
                if s_val == "Critical Risk":
                    ws1.write_string(r_idx, c_idx, s_val, fmt_risk_critical)
                elif s_val == "High Risk":
                    ws1.write_string(r_idx, c_idx, s_val, fmt_risk_high)
                elif s_val == "Medium Risk":
                    ws1.write_string(r_idx, c_idx, s_val, fmt_risk_medium)
                else:
                    ws1.write_string(r_idx, c_idx, s_val, fmt_risk_low)
            elif h_name == "Price Variance Probability":
                try:
                    ws1.write_number(r_idx, c_idx, float(val), fmt_prob)
                except (ValueError, TypeError):
                    ws1.write_string(r_idx, c_idx, str(val), fmt_prob)
            elif h_name in num_headers:
                try:
                    ws1.write_number(r_idx, c_idx, float(val), fmt_num)
                except (ValueError, TypeError):
                    ws1.write_string(r_idx, c_idx, str(val), fmt_num)
            elif h_name in center_headers:
                ws1.write_string(r_idx, c_idx, str(val), fmt_center)
            else:
                ws1.write_string(r_idx, c_idx, str(val), fmt_text)

    # Set column widths
    for c_idx, h_text in enumerate(display_headers):
        w = max(len(h_text) + 3, 12)
        if "Description" in h_name or "Name" in h_name:
            w = max(w, 24)
        if "Amount" in h_name or "Variance" in h_name or "Probability" in h_name:
            w = max(w, 18)
        ws1.set_column(c_idx, c_idx, min(w, 40))

    ws1.freeze_panes(7, 0)

    # ==========================================
    # SHEET 2: Feature_Importance
    # ==========================================
    if feature_importance_df is None and os.path.exists("reports/feature_importance.csv"):
        feature_importance_df = pd.read_csv("reports/feature_importance.csv")

    if feature_importance_df is not None and len(feature_importance_df) > 0:
        ws2 = wb.add_worksheet("Feature_Importance")
        ws2.set_row(1, 22)

        fmt_feat_hdr = wb.add_format({
            'bold': True, 'font_name': 'Calibri', 'font_size': 11, 'font_color': '#FFFFFF',
            'bg_color': '#2F5597', 'border': 1, 'border_color': border_color,
            'align': 'left', 'valign': 'vcenter'
        })
        fmt_feat_hdr_r = wb.add_format({
            'bold': True, 'font_name': 'Calibri', 'font_size': 11, 'font_color': '#FFFFFF',
            'bg_color': '#2F5597', 'border': 1, 'border_color': border_color,
            'align': 'right', 'valign': 'vcenter'
        })
        fmt_imp_num = wb.add_format({
            'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'border_color': border_color,
            'align': 'right', 'valign': 'vcenter', 'num_format': '0.000000000'
        })

        # Row 1: Title
        ws2.write(0, 0, f"{insight_id} | Feature_Importance", fmt_meta_lbl)

        # Row 2: Headers
        ws2.write(1, 0, "feature", fmt_feat_hdr)
        ws2.write(1, 1, "importance", fmt_feat_hdr_r)
        ws2.write(1, 2, "Direct Mathematical Influence On Price Fairness:", fmt_feat_hdr)

        col_feat_name = 'feature' if 'feature' in feature_importance_df.columns else feature_importance_df.columns[0]
        col_imp_name = 'importance' if 'importance' in feature_importance_df.columns else feature_importance_df.columns[1]
        col_expl_name = feature_importance_df.columns[2] if len(feature_importance_df.columns) > 2 else None

        for idx, r_data in feature_importance_df.iterrows():
            r_num = idx + 2
            ws2.write_string(r_num, 0, str(r_data[col_feat_name]), fmt_text)
            try:
                ws2.write_number(r_num, 1, float(r_data[col_imp_name]), fmt_imp_num)
            except (ValueError, TypeError):
                ws2.write_string(r_num, 1, str(r_data[col_imp_name]), fmt_imp_num)
            
            expl_str = str(r_data[col_expl_name]) if col_expl_name and pd.notna(r_data[col_expl_name]) else ""
            ws2.write_string(r_num, 2, expl_str, fmt_text)

        ws2.set_column(0, 0, 32)
        ws2.set_column(1, 1, 18)
        ws2.set_column(2, 2, 70)

    # ==========================================
    # SHEET 3: Columns Ignored
    # ==========================================
    ws3 = wb.add_worksheet("Columns Ignored")
    ws3.set_row(2, 20)

    # Row 1: Title
    ws3.write(0, 0, "Columns Ignored / Excluded from Model Training:", fmt_meta_lbl)

    # Row 3: Headers
    ign_headers = ["Ignored Column", "SAP Field Name", "Why It Was Ignored"]
    for c_idx, h_text in enumerate(ign_headers):
        ws3.write(2, c_idx, h_text, fmt_header)

    ignored_rows = [
        ("PO Number & Line Item", "EBELN, EBELP", "Arbitrary sequential keys (e.g., 6000034396). They have no statistical relation to whether a price is fair."),
        ("Vendor Code & Plant Code", "LIFNR, WERKS", "High cardinality IDs (hundreds of vendors and plants). If fed directly, models memorize specific vendor IDs rather than learning pricing behavior. (Captured indirectly via vendor_price_diff and plant_mat_price_diff)."),
        ("Material Code", "MATNR", "Thousands of unique parts. Captured statistically via reference_price and median grouping rather than raw categorical IDs."),
        ("Text Names & Descriptions", "TXZ01, NAME1, MAKTX", "Unstructured text strings (e.g., \"BUCKET CYLINDER-SHORT ARM\"). Decision trees require numerical features, not free-text strings."),
        ("Company Code & Name", "BUKRS, BUTXT", "Single operating entity (1000 - J.K. CEMENT LIMITED). Constant across all transactions with zero mathematical variance."),
        ("Geographic Master Fields", "ORT01, REGIO, LAND1", "Descriptive municipal and regional locations. Macro-geography is captured via is_domestic_vendor."),
        ("Currency", "WAERS", "100% of rows are INR (Indian Rupee). Zero mathematical variance."),
        ("Document Type", "BSART", "Qualitative classification (e.g., ZAN, ZSTD). Financial impact is captured directly by effective unit price and net value."),
        ("PO Date", "AEDAT", "Raw dates are timestamps, not prices. (Dates were used only to sort historical baseline medians and extract cyclical features)."),
        ("Info Record ID", "INFNR", "Sequential database key. Contractual pricing status is captured numerically via has_info_contract and contract_price_diff."),
        ("Target Label", "is_variance_exception", "Strictly excluded during training so the model remains 100% unsupervised and does not cheat.")
    ]

    for idx, (col_name, sap_field, reason) in enumerate(ignored_rows, start=3):
        ws3.write_string(idx, 0, col_name, fmt_text)
        ws3.write_string(idx, 1, sap_field, fmt_text)
        ws3.write_string(idx, 2, reason, fmt_text)

    ws3.set_column(0, 0, 28)
    ws3.set_column(1, 1, 24)
    ws3.set_column(2, 2, 95)

    wb.close()
    print(f"[SUCCESS] Executive Enriched Excel Report saved to: {target_path}")
    print(f"[INFO] Total Rows: {len(df_export):,}, Total Features: {len(display_headers)}, Sheets: ['ML_19_Enriched_Output(Predict)', 'Feature_Importance', 'Columns Ignored']")
    return target_path


def save_single_model_pipeline(model, scaler, feature_cols, output_dir: str = "models"):
    """
    Saves the single champion model (Isolation Forest) and scaler to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    bundle = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'model_name': 'Isolation Forest'
    }
    bundle_path = os.path.join(output_dir, "price_variance_champion_model.joblib")
    joblib.dump(bundle, bundle_path)
    print(f"[SUCCESS] Saved champion model bundle to: {bundle_path}")
