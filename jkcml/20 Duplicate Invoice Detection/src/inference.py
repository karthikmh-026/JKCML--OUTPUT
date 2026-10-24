"""
Inference & Excel Reporting Module - Use Case 20: Duplicate Invoice Detection
Handles scoring incoming SAP invoices using the #1 Ranked Champion ML Model (Isolation Forest)
and generating executive, formatted Excel reports matching the enterprise SAP reference template.
Optimized for high-volume enterprise datasets (100k+ to 1M+ rows) with exact missing value blank formatting.
"""
import os
import datetime
import joblib
import pandas as pd
import numpy as np
import xlsxwriter


def score_with_single_model(df: pd.DataFrame, model, scaler, feature_cols: list) -> pd.DataFrame:
    """
    Applies the selected #1 Champion ML model (Isolation Forest) to calculate:
    - duplicate_invoice_prob: continuous anomaly probability in [0.0, 1.0] (8 decimal precision)
    - predicted_duplicate: binary flag (1 for potential duplicate invoice anomaly, 0 for normal)
    - risk_category: executive risk triage (Critical Risk, High Risk, Medium Risk, Low Risk)
    """
    df_out = df.copy()

    X = df_out[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_scaled = scaler.transform(X)

    probs, raw_scores = model.predict_proba(X_scaled)
    preds = model.predict(X_scaled)

    df_out['duplicate_invoice_prob'] = np.round(probs, 8)
    df_out['predicted_duplicate'] = preds.astype(int)

    conditions = [
        (df_out['duplicate_invoice_prob'] >= 0.85) | ((df_out['predicted_duplicate'] == 1) & (df_out['is_exact_ref_dup'] == 1)),
        (df_out['duplicate_invoice_prob'] >= 0.70) | ((df_out['predicted_duplicate'] == 1) & (df_out['dup_similarity_index'] >= 35.0)),
        (df_out['duplicate_invoice_prob'] >= 0.50) | (df_out['predicted_duplicate'] == 1),
    ]
    choices = ['Critical Risk', 'High Risk', 'Medium Risk']
    df_out['risk_category'] = np.select(conditions, choices, default='Low Risk')

    df_out = df_out.sort_values(
        by=['duplicate_invoice_prob', 'dup_similarity_index', 'gross_invoice_amount'],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return df_out


def export_enriched_excel_report(
    df: pd.DataFrame,
    output_path: str = "ML_20_Duplicate_Invoice_Detection_Enriched_Output_Clean.xlsx",
    insight_id: str = "ML_20",
    use_case: str = "Duplicate Invoice Detection",
    description: str = "Identify potential duplicate payments, enriched with incoming invoice headers & items (RBKP, RSEG), purchasing documents (EKKO, EKPO), vendor master (LFA1, LFB1), material master (MARA, MAKT), company code (T001), and plant (T001W).",
    feature_importance_df: pd.DataFrame = None
) -> str:
    """
    Generates the executive enriched Excel report matching the enterprise SAP reference template:
    - Sheet 1: ML_20_Enriched_Output(Predict)
      - Rows 1-5: Metadata block (Insight ID, Use Case, Description, Generated On, Row Count)
      - Row 6: Blank row
      - Row 7: Enterprise SAP column headers
      - Row 8+: Enriched SAP invoice transactions with ML duplicate probabilities and risk tiers
    - Sheet 2: Feature_Importance (All 32 engineered features with mathematical rationale)
    - Sheet 3: Columns Ignored (Audit justification for excluded non-feature columns)
    Preserves exact missing values: if data is absent, output cell is left completely blank.
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
        ('double_inv_check_flag', 'Double Inv Check (REPRF)'),
        ('plant', 'Plant'),
        ('plant_name', 'Plant Name'),
        ('plant_city', 'Plant City'),
        ('material_id', 'Material'),
        ('material_desc', 'Material Description'),
        ('material_type', 'Material Type'),
        ('material_group', 'Material Group'),
        ('invoice_doc_number', 'Invoice Document (BELNR)'),
        ('fiscal_year', 'Fiscal Year'),
        ('invoice_item', 'Invoice Item'),
        ('doc_type', 'Document Type'),
        ('reference_invoice_no', 'Reference Invoice No (XBLNR)'),
        ('invoice_date', 'Invoice Date (BLDAT)'),
        ('posting_date', 'Posting Date (BUDAT)'),
        ('po_number', 'PO Number (EBELN)'),
        ('po_item', 'PO Item (EBELP)'),
        ('invoiced_qty', 'Invoiced Quantity'),
        ('unit_of_measure', 'Order Unit'),
        ('invoice_item_amount', 'Invoice Item Amount (WRBTR)'),
        ('tax_amount', 'Tax Amount (WMWST1)'),
        ('gross_invoice_amount', 'Gross Invoice Amount (RMWWR)'),
        ('po_net_value', 'PO Net Value (NETWR)'),
        ('currency', 'Currency'),
        ('payment_terms', 'Payment Terms'),
        ('posted_by_user', 'Posted By User'),
        ('exact_vendor_ref_count', 'Vendor+Ref Collision Count'),
        ('vendor_amt_date_count', 'Vendor+Amt+Date Collision Count'),
        ('dup_similarity_index', 'Duplicate Similarity Score'),
        ('duplicate_pattern_type', 'Duplicate Pattern Type'),
        ('duplicate_invoice_prob', 'Duplicate Invoice Probability'),
        ('predicted_duplicate', 'Predicted Duplicate Invoice'),
        ('risk_category', 'Risk Category')
    ]

    export_cols = []
    display_headers = []
    for internal_col, display_name in column_mapping:
        if internal_col in df.columns:
            export_cols.append(internal_col)
            display_headers.append(display_name)

    df_export = df[export_cols].copy()

    # Format dates
    for dt_col in ['invoice_date', 'posting_date']:
        if dt_col in df_export.columns:
            dt_series = pd.to_datetime(df_export[dt_col], errors='coerce')
            df_export[dt_col] = dt_series.dt.strftime('%d-%m-%Y 00:00').fillna('')

    # Clean text columns: replace NaN / 'nan' / 'None' with blank
    str_cols = [
        'company_code', 'company_name', 'company_city', 'company_country', 'company_currency',
        'vendor_id', 'vendor_name', 'vendor_city', 'vendor_region', 'vendor_country',
        'double_inv_check_flag', 'plant', 'plant_name', 'plant_city', 'material_id',
        'material_desc', 'material_type', 'material_group', 'invoice_doc_number', 'fiscal_year',
        'invoice_item', 'doc_type', 'reference_invoice_no', 'po_number', 'po_item',
        'unit_of_measure', 'currency', 'payment_terms', 'posted_by_user', 'duplicate_pattern_type',
        'risk_category'
    ]
    for c in str_cols:
        if c in df_export.columns:
            df_export[c] = df_export[c].fillna('').astype(str).str.strip()
            df_export[c] = df_export[c].replace({'nan': '', 'None': '', 'NaN': '', '<NA>': '', 'NULL': ''})

    actual_path = output_path
    save_candidates = [
        output_path,
        "ML_20_Duplicate_Invoice_Detection_Enriched_Output_Clean.xlsx",
        "ML_20_Duplicate_Invoice_Detection_Enriched_Output.xlsx",
        os.path.join("reports", os.path.basename(output_path)),
        f"ML_20_Duplicate_Invoice_Detection_Enriched_Output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    ]

    for candidate in save_candidates:
        try:
            wb = xlsxwriter.Workbook(candidate, {'constant_memory': True})
            actual_path = candidate
            break
        except Exception as e:
            print(f"[WARNING] Cannot open {candidate} for writing: {e}")
            continue

    # -------------------------------------------------------------
    # Formats
    # -------------------------------------------------------------
    fmt_meta_label = wb.add_format({'font_name': 'Calibri', 'font_size': 11, 'bold': True})
    fmt_meta_val = wb.add_format({'font_name': 'Calibri', 'font_size': 11, 'bold': False})

    fmt_header = wb.add_format({
        'font_name': 'Calibri', 'font_size': 10, 'bold': True, 'font_color': '#000000',
        'bg_color': '#F2F2F2', 'border': 1, 'border_color': '#D3D3D3',
        'align': 'center', 'valign': 'vcenter'
    })

    fmt_text_left = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'border_color': '#D3D3D3', 'align': 'left'})
    fmt_text_center = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'border_color': '#D3D3D3', 'align': 'center'})
    fmt_num_2dec = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'border_color': '#D3D3D3', 'align': 'right', 'num_format': '#,##0.00'})
    fmt_num_int = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'border_color': '#D3D3D3', 'align': 'center', 'num_format': '#,##0'})
    fmt_prob = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'border_color': '#D3D3D3', 'align': 'right', 'num_format': '0.00000000'})

    fmt_crit = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'bold': True, 'font_color': '#9C0006', 'bg_color': '#FFC7CE', 'border': 1, 'border_color': '#D3D3D3', 'align': 'center'})
    fmt_high = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'bold': True, 'font_color': '#9C6500', 'bg_color': '#FFEB9C', 'border': 1, 'border_color': '#D3D3D3', 'align': 'center'})
    fmt_med = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'font_color': '#006100', 'bg_color': '#C6EFCE', 'border': 1, 'border_color': '#D3D3D3', 'align': 'center'})
    fmt_low = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'border_color': '#D3D3D3', 'align': 'center'})

    # -------------------------------------------------------------
    # Sheet 1: ML_20_Enriched_Output(Predict)
    # -------------------------------------------------------------
    ws = wb.add_worksheet("ML_20_Enriched_Output(Predict)")

    gen_time_str = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
    row_count_str = f"{len(df_export):,}"

    metadata_rows = [
        ("Insight ID", insight_id),
        ("Use Case", use_case),
        ("Description", description),
        ("Generated On", gen_time_str),
        ("Row Count", row_count_str)
    ]

    for r_idx, (label, val) in enumerate(metadata_rows):
        ws.write(r_idx, 0, label, fmt_meta_label)
        ws.write(r_idx, 1, val, fmt_meta_val)

    # Row 6 left blank intentionally
    header_row_idx = 6  # 0-indexed -> row 7 in Excel
    ws.set_row(header_row_idx, 24)

    for col_idx, header_text in enumerate(display_headers):
        ws.write(header_row_idx, col_idx, header_text, fmt_header)

    # Determine column formats
    col_formats = []
    for header_name in display_headers:
        if "Probability" in header_name:
            col_formats.append(fmt_prob)
        elif any(k in header_name for k in ["Amount", "Value", "Score", "Quantity"]):
            col_formats.append(fmt_num_2dec)
        elif header_name in [
            "Company Code", "Vendor", "Plant", "Material", "Invoice Document (BELNR)",
            "Fiscal Year", "Invoice Item", "PO Number (EBELN)", "PO Item (EBELP)",
            "Predicted Duplicate Invoice", "Invoice Date (BLDAT)", "Posting Date (BUDAT)",
            "Double Inv Check (REPRF)", "Vendor+Ref Collision Count", "Vendor+Amt+Date Collision Count"
        ]:
            col_formats.append(fmt_text_center)
        elif header_name == "Risk Category":
            col_formats.append("RISK_CATEGORY")
        else:
            col_formats.append(fmt_text_left)

    # Write data rows
    data_values = df_export.values
    for r_offset, row_vals in enumerate(data_values, start=7):
        for c_idx, val in enumerate(row_vals):
            fmt = col_formats[c_idx]

            # Missing value handling: if pd.isna or None or empty string -> write blank cell
            if val is None or pd.isna(val) or val == '':
                ws.write_blank(r_offset, c_idx, None, fmt_text_left if fmt != "RISK_CATEGORY" else fmt_low)
                continue

            if fmt == "RISK_CATEGORY":
                val_str = str(val)
                if val_str == "Critical Risk":
                    ws.write(r_offset, c_idx, val_str, fmt_crit)
                elif val_str == "High Risk":
                    ws.write(r_offset, c_idx, val_str, fmt_high)
                elif val_str == "Medium Risk":
                    ws.write(r_offset, c_idx, val_str, fmt_med)
                else:
                    ws.write(r_offset, c_idx, val_str, fmt_low)
            elif fmt in [fmt_num_2dec, fmt_prob, fmt_num_int]:
                try:
                    num_val = float(val)
                    ws.write_number(r_offset, c_idx, num_val, fmt)
                except (ValueError, TypeError):
                    ws.write(r_offset, c_idx, str(val), fmt_text_left)
            else:
                ws.write(r_offset, c_idx, str(val), fmt)

    # Set column widths
    for c_idx, header_text in enumerate(display_headers):
        w = max(len(str(header_text)) + 4, 12)
        if any(k in header_text for k in ["Description", "Name", "Pattern"]):
            w = max(w, 24)
        ws.set_column(c_idx, c_idx, min(w, 40))

    ws.freeze_panes(7, 0)

    # -------------------------------------------------------------
    # Sheet 2: Feature_Importance
    # -------------------------------------------------------------
    if feature_importance_df is None and os.path.exists("reports/feature_importance.csv"):
        feature_importance_df = pd.read_csv("reports/feature_importance.csv")

    if feature_importance_df is not None and len(feature_importance_df) > 0:
        ws_feat = wb.add_worksheet("Feature_Importance")
        ws_feat.write(0, 0, f"{insight_id} | Feature_Importance", fmt_meta_label)

        fmt_feat_header = wb.add_format({
            'font_name': 'Calibri', 'font_size': 11, 'bold': True, 'font_color': '#FFFFFF',
            'bg_color': '#2F5597', 'border': 1, 'border_color': '#D3D3D3',
            'align': 'left', 'valign': 'vcenter'
        })
        fmt_feat_header_r = wb.add_format({
            'font_name': 'Calibri', 'font_size': 11, 'bold': True, 'font_color': '#FFFFFF',
            'bg_color': '#2F5597', 'border': 1, 'border_color': '#D3D3D3',
            'align': 'right', 'valign': 'vcenter'
        })
        fmt_imp_val = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'border_color': '#D3D3D3', 'align': 'right', 'num_format': '0.000000000'})

        ws_feat.set_row(1, 22)
        ws_feat.write(1, 0, "feature", fmt_feat_header)
        ws_feat.write(1, 1, "importance", fmt_feat_header_r)
        ws_feat.write(1, 2, "Direct Mathematical Influence On Duplicate Payment Risk:", fmt_feat_header)

        col_feat_name = 'feature' if 'feature' in feature_importance_df.columns else feature_importance_df.columns[0]
        col_imp_name = 'importance' if 'importance' in feature_importance_df.columns else feature_importance_df.columns[1]
        has_expl = len(feature_importance_df.columns) > 2
        col_expl_name = feature_importance_df.columns[2] if has_expl else None

        for idx, r_data in feature_importance_df.iterrows():
            curr_row = idx + 2
            ws_feat.write(curr_row, 0, str(r_data[col_feat_name]), fmt_text_left)
            ws_feat.write_number(curr_row, 1, float(r_data[col_imp_name]), fmt_imp_val)
            expl_val = str(r_data[col_expl_name]) if col_expl_name and pd.notna(r_data[col_expl_name]) else ""
            ws_feat.write(curr_row, 2, expl_val, fmt_text_left)

        ws_feat.set_column(0, 0, 32)
        ws_feat.set_column(1, 1, 18)
        ws_feat.set_column(2, 2, 80)

    # -------------------------------------------------------------
    # Sheet 3: Columns Ignored
    # -------------------------------------------------------------
    ws_ign = wb.add_worksheet("Columns Ignored")
    ws_ign.write(0, 0, "Columns Ignored / Excluded from Model Training:", fmt_meta_label)

    fmt_ign_header = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'bold': True, 'border': 1, 'border_color': '#D3D3D3', 'align': 'left'})
    ws_ign.set_row(2, 20)
    ws_ign.write(2, 0, "Ignored Column", fmt_ign_header)
    ws_ign.write(2, 1, "SAP Field Name", fmt_ign_header)
    ws_ign.write(2, 2, "Why It Was Ignored", fmt_ign_header)

    ignored_rows = [
        ("Invoice Document & Item", "BELNR, GJAHR, BUZEI", "Sequential SAP system document keys generated at posting time. They carry no intrinsic signal on whether an invoice is a duplicate."),
        ("PO Number & Line Item", "EBELN, EBELP", "Raw document identifiers. Multi-billing and over-invoicing behavior on the same PO item are captured mathematically via po_item_exact_amt_count and cum_po_invoiced_ratio."),
        ("Raw Reference Invoice String", "XBLNR", "Free-text supplier invoice string (e.g., '014/2025-26'). Transformed into exact_vendor_ref_count, norm_vendor_ref_count, and is_fuzzy_ref_dup."),
        ("Vendor Code & Plant Code", "LIFNR, WERKS", "High-cardinality categorical IDs. Feeding raw IDs causes overfitting to specific suppliers rather than learning duplicate collision patterns."),
        ("Material Code & Group", "MATNR, MATKL, MTART", "Thousands of unique material IDs. Captured numerically via vendor_mat_amt_count and effective_inv_unit_price."),
        ("Text Names & Descriptions", "NAME1, MAKTX, TXZ01", "Unstructured display strings (Vendor Name, Plant Name, Material Description) retained strictly for human audit readability."),
        ("Company Code & Name", "BUKRS, BUTXT", "Single operating entity across the dataset with zero mathematical variance."),
        ("Geographic Master Fields", "ORT01, REGIO, LAND1", "Descriptive municipal and regional text fields with no direct bearing on duplicate invoice submission."),
        ("Currency", "WAERS", "Constant transaction currency (INR) across domestic procurement invoices."),
        ("Raw Date Timestamps", "BLDAT, BUDAT, CPUDT, ZFBDT", "Raw calendar dates cannot be fed directly to distance/tree models. Engineered into days_since_same_vendor_amt, posting_lag_days, and entry_lag_days."),
        ("Audit Benchmark Label", "is_duplicate_exception", "Strictly excluded from training so all 5 ML models remain 100% unsupervised and learn purely from input feature distributions.")
    ]

    for r_offset, (col_name, sap_field, reason) in enumerate(ignored_rows, start=3):
        ws_ign.write(r_offset, 0, col_name, fmt_text_left)
        ws_ign.write(r_offset, 1, sap_field, fmt_text_left)
        ws_ign.write(r_offset, 2, reason, fmt_text_left)

    ws_ign.set_column(0, 0, 28)
    ws_ign.set_column(1, 1, 26)
    ws_ign.set_column(2, 2, 98)

    wb.close()
    print(f"[SUCCESS] Executive Enriched Excel Report saved to: {actual_path}")
    print(f"[INFO] Total Rows: {len(df_export):,}, Total Columns: {len(display_headers)}")
    return actual_path


def save_single_model_pipeline(model, scaler, feature_cols, suite=None, comparison_df=None, output_dir: str = "models"):
    """
    Saves the selected #1 Champion Model (Isolation Forest), scaler, and 5-model benchmark bundle to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    bundle = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'model_name': 'Isolation Forest',
        'benchmark_ranking': comparison_df.to_dict(orient='records') if comparison_df is not None else None
    }
    champ_path = os.path.join(output_dir, "duplicate_invoice_champion_model.joblib")
    bundle_path = os.path.join(output_dir, "duplicate_invoice_model_bundle.joblib")
    if_path = os.path.join(output_dir, "champion_isolation_forest.joblib")
    scaler_path = os.path.join(output_dir, "robust_scaler.joblib")

    joblib.dump(bundle, champ_path)
    joblib.dump(bundle, bundle_path)
    joblib.dump(model.model, if_path)
    joblib.dump(scaler, scaler_path)

    if suite is not None and 'dbscan' in suite.models:
        joblib.dump(suite.models['dbscan'], os.path.join(output_dir, "champion_dbscan.joblib"))

    print(f"[SUCCESS] Saved champion model bundle to: {champ_path}")
