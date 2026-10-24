"""
Inference & Excel Reporting Module - Use Case 23: Vendor Risk Segmentation (Clustering)
Handles scoring and segmenting SAP Vendor records using the #1 Ranked Champion Clustering Model
(K-Means Clustering) and generating executive 3-sheet Excel reports matching the enterprise SAP template.
Uses xlsxwriter with constant_memory=True for fast, memory-safe generation of up to 1,000,000+ vendor records.
"""
import os
import datetime
import joblib
import pandas as pd
import numpy as np
import xlsxwriter


def score_with_single_model(df: pd.DataFrame, model, scaler, feature_cols: list) -> pd.DataFrame:
    """
    Applies the selected #1 Champion Clustering ML model (K-Means Clustering) to calculate:
    - vendor_cluster_id: ordered cluster tier (0 = Strategic/High-Reliability, 1 = Preferred/Core, 2 = Watchlist/Moderate-Risk, 3 = High-Risk/Non-Compliant)
    - vendor_segment_name: human-readable Vendor Reliability & Risk Segment
    - vendor_risk_prob: continuous reliability/compliance risk probability in [0.0, 1.0] (8 decimals)
    - predicted_vendor_exception: binary flag (1 for High-Risk / Non-Compliant vendor, 0 for Reliable / Compliant)
    - risk_category: executive triage tier (Critical Risk, High Risk, Medium Risk, Low Risk)
    """
    df_out = df.copy()

    X = df_out[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_scaled = scaler.transform(X)

    probs, raw_scores, mapped_tiers = model.predict_proba(X_scaled)
    _, segment_names = model.predict_clusters(X_scaled)
    preds = model.predict(X_scaled)

    df_out['vendor_cluster_id'] = mapped_tiers.astype(int)
    df_out['vendor_segment_name'] = segment_names
    df_out['vendor_risk_prob'] = np.round(probs, 8)
    df_out['predicted_vendor_exception'] = preds.astype(int)

    conditions = [
        (df_out['vendor_cluster_id'] == 3) & (
            (df_out['is_vendor_blocked_or_flagged'] == 1) |
            (df_out['total_blocked_invoices'] > 0) |
            (df_out['open_ap_balance'] > 0) |
            (df_out['vendor_risk_index'] >= 80.0)
        ),
        (df_out['vendor_cluster_id'] == 3) | (df_out['vendor_risk_prob'] >= 0.72),
        (df_out['vendor_cluster_id'] == 2) | (df_out['vendor_risk_prob'] >= 0.48),
    ]
    choices = ['Critical Risk', 'High Risk', 'Medium Risk']
    df_out['risk_category'] = np.select(conditions, choices, default='Low Risk')

    # Sort so high-priority vendor accounts with financial/operational exposure are structured first
    df_out = df_out.sort_values(
        by=['vendor_risk_prob', 'total_blocked_invoices', 'open_ap_balance', 'effective_vendor_spend'],
        ascending=[False, False, False, False]
    ).reset_index(drop=True)

    return df_out


def export_enriched_excel_report(
    df: pd.DataFrame,
    output_path: str = "ML_23_Vendor_Risk_Segmentation_Enriched_Output.xlsx",
    insight_id: str = "ML_23",
    use_case: str = "Vendor Risk Segmentation",
    description: str = "Group vendors by reliability and risk using Unsupervised Clustering across SAP Vendor Master (LFA1, LFB1, LFBK), Purchase Orders (EKKO, EKPO), Invoice Receipts & Blocked Invoices (RBKP, RBKP_BLOCKED), Open Accounts Payable (BSIK), and Quality Inspection Lots (QALS).",
    feature_importance_df: pd.DataFrame = None
) -> str:
    """
    Generates the executive enriched Excel report matching the enterprise SAP reference template:
    - Sheet 1: ML_23_Enriched_Output(Predict)
      - Rows 1-5: Metadata block (Insight ID, Use Case, Description, Generated On, Row Count)
      - Row 6: Blank row
      - Row 7: Enterprise SAP column headers
      - Row 8+: Enriched SAP Vendor Risk records with cluster assignments, reliability/risk scores, probabilities, and risk tiers
    - Sheet 2: Feature_Importance (All 32 engineered vendor risk features with mathematical rationale)
    - Sheet 3: Columns Ignored (Audit justification for excluded non-feature columns)
    
    Uses xlsxwriter with constant_memory=True to write up to 1,000,000+ rows directly to disk without memory overflow.
    Missing/null values are written strictly as empty blank strings "".
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    column_mapping = [
        ('source_schema', 'Data Source (Schema)'),
        ('company_code', 'Company Code (BUKRS)'),
        ('company_name', 'Company Name (BUTXT)'),
        ('company_city', 'Company City'),
        ('company_country', 'Company Country'),
        ('currency', 'Currency (WAERS)'),
        ('purchasing_org', 'Purchasing Org (EKORG)'),
        ('purchasing_group', 'Purchasing Group (EKGRP)'),
        ('vendor_id', 'Vendor (LIFNR)'),
        ('vendor_name', 'Vendor Name (NAME1)'),
        ('vendor_city', 'Vendor City (ORT01)'),
        ('vendor_region', 'Region (REGIO)'),
        ('postal_code', 'Postal Code (PSTLZ)'),
        ('account_group', 'Account Group (KTOKK)'),
        ('account_group_desc', 'Account Group Description'),
        ('payment_terms', 'Payment Terms (ZTERM)'),
        ('recon_account', 'Recon Account (AKONT)'),
        ('dup_invoice_check_flag', 'Dup Inv Check (REPRF)'),
        ('gstin_tax_no', 'GSTIN (STCD3)'),
        ('pan_number', 'PAN (J_1IPANNO)'),
        ('posting_block', 'Posting Block (SPERR)'),
        ('purchasing_block', 'Purchasing Block (SPERM)'),
        ('payment_block', 'Payment Block (SPERZ/ZAHLS)'),
        ('deletion_flag', 'Deletion Flag (LOEVM)'),
        ('master_created_date', 'Created Date (ERDAT)'),
        ('first_po_date', 'First PO Date (BEDAT)'),
        ('last_po_date', 'Last PO Date (BEDAT)'),
        ('last_invoice_date', 'Last Invoice Date (BUDAT)'),
        ('recency_effective_days', 'Recency (Days)'),
        ('po_count', 'PO Count (EKKO)'),
        ('po_item_count', 'PO Items (EKPO)'),
        ('return_po_item_count', 'Return PO Items (RETPO)'),
        ('total_po_net_value', 'Total PO Net Spend (NETWR)'),
        ('invoice_count', 'Invoice Count (RBKP)'),
        ('total_invoice_gross_value', 'Total Invoiced Gross (RMWWR)'),
        ('total_blocked_invoices', 'Blocked Invoices (ZLSPR/MRM)'),
        ('reversed_invoices', 'Reversed Invoices (STBLG)'),
        ('open_ap_item_count', 'Open AP Items (BSIK)'),
        ('open_ap_balance', 'Open AP Balance (WRBTR)'),
        ('inspection_lot_count', 'Quality Lots (QALS)'),
        ('delivery_reliability_score', 'Delivery Reliability (0-100)'),
        ('invoice_compliance_score', 'Invoice Compliance (0-100)'),
        ('quality_governance_score', 'Quality & Governance (0-100)'),
        ('vendor_tier_code', 'Vendor Rating Code (D-I-Q)'),
        ('vendor_reliability_score', 'Vendor Reliability Score'),
        ('vendor_risk_index', 'Vendor Risk Index'),
        ('vendor_cluster_id', 'ML Cluster ID'),
        ('vendor_segment_name', 'ML Vendor Risk Segment'),
        ('vendor_risk_prob', 'Vendor Risk Probability'),
        ('predicted_vendor_exception', 'Predicted Risk Exception'),
        ('risk_category', 'Risk Category')
    ]

    export_cols = []
    display_headers = []
    for internal_col, display_name in column_mapping:
        if internal_col in df.columns:
            export_cols.append(internal_col)
            display_headers.append(display_name)

    df_export = df[export_cols].copy()

    # Pre-format date columns to DD-MM-YYYY strings or empty string
    for dt_col in ['master_created_date', 'first_po_date', 'last_po_date', 'last_invoice_date']:
        if dt_col in df_export.columns:
            df_export[dt_col] = pd.to_datetime(df_export[dt_col], errors='coerce', dayfirst=True).dt.strftime('%d-%m-%Y').fillna('')

    # Clean text columns: replace NaN/None/na with empty string ""
    for col in df_export.select_dtypes(include=['object', 'string']).columns:
        df_export[col] = df_export[col].fillna('').astype(str).str.strip()
        df_export[col] = df_export[col].replace({'nan': '', 'NaN': '', 'None': '', 'null': '', 'NaT': '', '<NA>': '', 'undefined': ''})

    # Prepare file target
    save_candidates = [
        output_path,
        "ML_23_Vendor_Risk_Segmentation_Enriched_Output_Clean.xlsx",
        "ML_23_Vendor_Risk_Segmentation_Enriched_Output_Final.xlsx",
        os.path.join("reports", os.path.basename(output_path)),
        f"ML_23_Vendor_Risk_Output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    ]

    actual_path = None
    wb = None
    for candidate in save_candidates:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(candidate)), exist_ok=True)
            wb = xlsxwriter.Workbook(candidate, {'constant_memory': True, 'use_zip64': True})
            actual_path = candidate
            break
        except Exception:
            continue

    if wb is None:
        actual_path = f"ML_23_Vendor_Risk_{int(datetime.datetime.now().timestamp())}.xlsx"
        wb = xlsxwriter.Workbook(actual_path, {'constant_memory': True, 'use_zip64': True})

    # Formats definition
    fmt_meta_lbl = wb.add_format({'font_name': 'Calibri', 'font_size': 11, 'bold': True, 'font_color': '#000000'})
    fmt_meta_val = wb.add_format({'font_name': 'Calibri', 'font_size': 11, 'bold': False, 'font_color': '#000000'})
    
    fmt_header = wb.add_format({
        'font_name': 'Calibri', 'font_size': 10, 'bold': True, 'font_color': '#000000',
        'bg_color': '#F2F2F2', 'border': 1, 'border_color': '#D3D3D3',
        'align': 'center', 'valign': 'vcenter'
    })

    fmt_data_left = wb.add_format({
        'font_name': 'Calibri', 'font_size': 10, 'font_color': '#000000',
        'border': 1, 'border_color': '#D3D3D3', 'align': 'left', 'valign': 'vcenter'
    })
    fmt_data_center = wb.add_format({
        'font_name': 'Calibri', 'font_size': 10, 'font_color': '#000000',
        'border': 1, 'border_color': '#D3D3D3', 'align': 'center', 'valign': 'vcenter'
    })
    fmt_data_currency = wb.add_format({
        'font_name': 'Calibri', 'font_size': 10, 'font_color': '#000000',
        'border': 1, 'border_color': '#D3D3D3', 'align': 'right', 'valign': 'vcenter',
        'num_format': '#,##0.00'
    })
    fmt_data_prob = wb.add_format({
        'font_name': 'Calibri', 'font_size': 10, 'font_color': '#000000',
        'border': 1, 'border_color': '#D3D3D3', 'align': 'right', 'valign': 'vcenter',
        'num_format': '0.00000000'
    })
    fmt_data_int = wb.add_format({
        'font_name': 'Calibri', 'font_size': 10, 'font_color': '#000000',
        'border': 1, 'border_color': '#D3D3D3', 'align': 'center', 'valign': 'vcenter',
        'num_format': '#,##0'
    })

    # Risk Category conditional formats
    fmt_crit = wb.add_format({
        'font_name': 'Calibri', 'font_size': 10, 'bold': True, 'font_color': '#9C0006',
        'bg_color': '#FFC7CE', 'border': 1, 'border_color': '#D3D3D3', 'align': 'center', 'valign': 'vcenter'
    })
    fmt_high = wb.add_format({
        'font_name': 'Calibri', 'font_size': 10, 'bold': True, 'font_color': '#9C6500',
        'bg_color': '#FFEB9C', 'border': 1, 'border_color': '#D3D3D3', 'align': 'center', 'valign': 'vcenter'
    })
    fmt_med = wb.add_format({
        'font_name': 'Calibri', 'font_size': 10, 'font_color': '#006100',
        'bg_color': '#C6EFCE', 'border': 1, 'border_color': '#D3D3D3', 'align': 'center', 'valign': 'vcenter'
    })

    # =============================================================
    # Sheet 1: ML_23_Enriched_Output(Predict)
    # =============================================================
    ws = wb.add_worksheet("ML_23_Enriched_Output(Predict)")
    ws.freeze_panes(7, 0)  # Freeze header row

    gen_time_str = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
    row_count_str = f"{len(df_export):,}"

    metadata_rows = [
        ("Insight ID", insight_id),
        ("Use Case", use_case),
        ("Description", description),
        ("Generated On", gen_time_str),
        ("Row Count", row_count_str)
    ]
    for r_idx, (lbl, val) in enumerate(metadata_rows):
        ws.write_string(r_idx, 0, lbl, fmt_meta_lbl)
        ws.write_string(r_idx, 1, val, fmt_meta_val)

    # Row 5 (0-indexed) left blank

    # Write Headers on Row 6 (0-indexed, display row 7)
    ws.set_row(6, 24)
    for c_idx, h_text in enumerate(display_headers):
        ws.write_string(6, c_idx, h_text, fmt_header)

    # Pre-classify column types for ultra-fast cell writing
    col_types = []
    for h_name in display_headers:
        if "Probability" in h_name:
            col_types.append('prob')
        elif any(k in h_name for k in ["Spend", "Gross", "Balance", "Score", "Index", "Reliability", "Compliance", "Governance"]):
            col_types.append('currency')
        elif any(k in h_name for k in ["Count", "Items", "Invoices", "Lots", "Days", "Cluster ID", "Exception"]):
            col_types.append('int')
        elif h_name == "Risk Category":
            col_types.append('risk')
        elif any(k in h_name for k in [
            "Company Code", "Currency", "Purchasing Org", "Purchasing Group", "Vendor (LIFNR)",
            "Region", "Postal Code", "Account Group", "Payment Terms", "Recon Account",
            "Dup Inv Check", "Posting Block", "Purchasing Block", "Payment Block", "Deletion Flag",
            "Vendor Rating Code", "Created Date", "First PO Date", "Last PO Date", "Last Invoice Date",
            "Data Source (Schema)"
        ]):
            col_types.append('center')
        else:
            col_types.append('left')

    # Stream write rows starting from row 7 (display row 8)
    data_matrix = df_export.values
    for r_idx, row_vals in enumerate(data_matrix, start=7):
        for c_idx, val in enumerate(row_vals):
            c_type = col_types[c_idx]

            # Missing value handling: if null, NaN, None, NaT, or empty string -> write blank cell ""
            if pd.isna(val) or val is None or val == '' or str(val).strip() in ['', 'nan', 'NaN', 'None', 'null', 'NaT', '<NA>', 'undefined']:
                ws.write_string(r_idx, c_idx, '', fmt_data_center if c_type in ['center', 'int'] else fmt_data_left)
                continue

            if c_type == 'risk':
                v_str = str(val).strip()
                if v_str == "Critical Risk":
                    ws.write_string(r_idx, c_idx, v_str, fmt_crit)
                elif v_str == "High Risk":
                    ws.write_string(r_idx, c_idx, v_str, fmt_high)
                elif v_str == "Medium Risk":
                    ws.write_string(r_idx, c_idx, v_str, fmt_med)
                else:
                    ws.write_string(r_idx, c_idx, v_str, fmt_data_center)
            elif c_type == 'prob':
                try:
                    ws.write_number(r_idx, c_idx, float(val), fmt_data_prob)
                except (ValueError, TypeError):
                    ws.write_string(r_idx, c_idx, str(val), fmt_data_left)
            elif c_type == 'currency':
                try:
                    ws.write_number(r_idx, c_idx, float(val), fmt_data_currency)
                except (ValueError, TypeError):
                    ws.write_string(r_idx, c_idx, str(val), fmt_data_left)
            elif c_type == 'int':
                try:
                    ws.write_number(r_idx, c_idx, int(round(float(val))), fmt_data_int)
                except (ValueError, TypeError):
                    ws.write_string(r_idx, c_idx, str(val), fmt_data_center)
            elif c_type == 'center':
                ws.write_string(r_idx, c_idx, str(val).strip(), fmt_data_center)
            else:
                ws.write_string(r_idx, c_idx, str(val).strip(), fmt_data_left)

    # Set column widths
    for c_idx, h_text in enumerate(display_headers):
        w = max(len(h_text) + 3, 12)
        if "Name" in h_text or "Description" in h_text or "Segment" in h_text:
            w = max(w, 28)
        ws.set_column(c_idx, c_idx, min(w, 45))

    # =============================================================
    # Sheet 2: Feature_Importance
    # =============================================================
    if feature_importance_df is None and os.path.exists("reports/feature_importance.csv"):
        feature_importance_df = pd.read_csv("reports/feature_importance.csv")

    ws_feat = wb.add_worksheet("Feature_Importance")
    fmt_feat_title = wb.add_format({'font_name': 'Calibri', 'font_size': 11, 'bold': True, 'font_color': '#000000'})
    fmt_feat_header = wb.add_format({
        'font_name': 'Calibri', 'font_size': 11, 'bold': True, 'font_color': '#FFFFFF',
        'bg_color': '#2F5597', 'border': 1, 'border_color': '#D3D3D3', 'align': 'left', 'valign': 'vcenter'
    })
    fmt_feat_header_r = wb.add_format({
        'font_name': 'Calibri', 'font_size': 11, 'bold': True, 'font_color': '#FFFFFF',
        'bg_color': '#2F5597', 'border': 1, 'border_color': '#D3D3D3', 'align': 'right', 'valign': 'vcenter'
    })

    ws_feat.write_string(0, 0, f"{insight_id} | Feature_Importance", fmt_feat_title)
    ws_feat.set_row(1, 22)
    ws_feat.write_string(1, 0, "feature", fmt_feat_header)
    ws_feat.write_string(1, 1, "importance", fmt_feat_header_r)
    ws_feat.write_string(1, 2, "Direct Mathematical Influence On Vendor Risk Segmentation & Reliability:", fmt_feat_header)

    if feature_importance_df is not None and len(feature_importance_df) > 0:
        col_f = 'feature' if 'feature' in feature_importance_df.columns else feature_importance_df.columns[0]
        col_i = 'importance' if 'importance' in feature_importance_df.columns else feature_importance_df.columns[1]
        has_expl = len(feature_importance_df.columns) > 2
        col_e = feature_importance_df.columns[2] if has_expl else None

        fmt_imp_num = wb.add_format({
            'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'border_color': '#D3D3D3',
            'align': 'right', 'valign': 'vcenter', 'num_format': '0.000000000'
        })

        for idx, r_data in feature_importance_df.iterrows():
            r_row = idx + 2
            ws_feat.write_string(r_row, 0, str(r_data[col_f]), fmt_data_left)
            ws_feat.write_number(r_row, 1, float(r_data[col_i]), fmt_imp_num)
            expl_val = str(r_data[col_e]) if col_e and pd.notna(r_data[col_e]) else ""
            ws_feat.write_string(r_row, 2, expl_val, fmt_data_left)

    ws_feat.set_column(0, 0, 32)
    ws_feat.set_column(1, 1, 18)
    ws_feat.set_column(2, 2, 88)

    # =============================================================
    # Sheet 3: Columns Ignored
    # =============================================================
    ws_ign = wb.add_worksheet("Columns Ignored")
    ws_ign.write_string(0, 0, "Columns Ignored / Excluded from Model Training:", fmt_feat_title)

    fmt_ign_header = wb.add_format({
        'font_name': 'Calibri', 'font_size': 10, 'bold': True, 'font_color': '#000000',
        'border': 1, 'border_color': '#D3D3D3', 'align': 'left', 'valign': 'vcenter'
    })
    ws_ign.set_row(2, 20)
    for c_i, h_t in enumerate(["Ignored Column", "SAP Field Name", "Why It Was Ignored"]):
        ws_ign.write_string(2, c_i, h_t, fmt_ign_header)

    ignored_rows = [
        ("Vendor Account Number", "LIFNR", "Unique alphanumeric SAP Vendor Master primary key. Carries no intrinsic mathematical signal on supply reliability or compliance risk."),
        ("Vendor Name & Address Text", "NAME1, ORT01, PSTLZ, STRAS", "Free-text supplier name and geographic location strings retained strictly for procurement report identification."),
        ("Tax & Statutory Registration IDs", "STCD3, J_1IPANNO", "High-cardinality GSTIN and PAN registration strings. Used to derive tax/governance compliance indicators and excluded as raw text."),
        ("Purchasing Organizational Codes", "EKORG, EKGRP, BSART", "Categorical purchasing organization and buyer group codes. Used to condition peer spend Z-scores (purch_org_spend_zscore) and retained for drill-down."),
        ("Account Group & Industry Codes", "KTOKK, TXT30, BRSCH", "Master data account classification codes retained for segment filtering rather than Euclidean centroid calculation."),
        ("Raw Transaction & Creation Dates", "BEDAT, AEDAT, BUDAT, ERDAT", "Raw calendar dates cannot be ingested directly by clustering algorithms. Engineered into recency_po_days, recency_invoice_days, recency_effective_days, vendor_tenure_days, and active_procurement_span_days."),
        ("Reconciliation Account & Payment Terms", "AKONT, ZTERM, ZWELS, FDGRV", "General Ledger reconciliation account and payment method codes retained as financial audit context."),
        ("Created By User ID", "ERNAM", "SAP user identifier who created the vendor master record. Carries no supplier reliability signal."),
        ("Company Code & Currency", "BUKRS, BUTXT, WAERS", "Enterprise entity metadata retained for multi-company reporting context."),
        ("Source Schema Metadata", "source_schema", "Internal tracking column denoting whether vendor record originated from HANA_P2, HANA_P3, or both. Excluded from clustering centroids."),
        ("Post-Hoc Benchmark Segment & Label", "benchmark_vendor_segment, is_vendor_risk_exception", "Strictly excluded from model training so all 5 clustering models remain 100% unsupervised and group vendors purely from input feature distributions.")
    ]

    for r_i, (col_n, sap_f, why_i) in enumerate(ignored_rows, start=3):
        ws_ign.write_string(r_i, 0, col_n, fmt_data_left)
        ws_ign.write_string(r_i, 1, sap_f, fmt_data_left)
        ws_ign.write_string(r_i, 2, why_i, fmt_data_left)

    ws_ign.set_column(0, 0, 32)
    ws_ign.set_column(1, 1, 28)
    ws_ign.set_column(2, 2, 98)

    wb.close()
    print(f"[SUCCESS] Executive Enriched Excel Report saved to: {actual_path}")
    print(f"[INFO] Total Rows: {len(df_export):,}, Total Columns: {len(display_headers)}, Engine: xlsxwriter (constant_memory=True)")
    return actual_path


def save_single_model_pipeline(model, scaler, feature_cols, suite=None, comparison_df=None, output_dir: str = "models"):
    """
    Saves the selected #1 Champion Clustering Model (K-Means Clustering), scaler, and 5-model benchmark bundle to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    bundle = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'model_name': 'K-Means Clustering',
        'benchmark_ranking': comparison_df.to_dict(orient='records') if comparison_df is not None else None
    }
    champ_path = os.path.join(output_dir, "vendor_risk_champion_model.joblib")
    bundle_path = os.path.join(output_dir, "vendor_risk_model_bundle.joblib")
    km_path = os.path.join(output_dir, "champion_kmeans.joblib")
    scaler_path = os.path.join(output_dir, "robust_scaler.joblib")

    joblib.dump(bundle, champ_path)
    joblib.dump(bundle, bundle_path)
    joblib.dump(model.model, km_path)
    joblib.dump(scaler, scaler_path)

    if suite is not None and 'gmm' in suite.models:
        joblib.dump(suite.models['gmm'], os.path.join(output_dir, "challenger_gmm.joblib"))

    print(f"[SUCCESS] Saved champion clustering model bundle to: {champ_path}")
