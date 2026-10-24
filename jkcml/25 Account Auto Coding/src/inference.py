"""
Inference & Excel Reporting Module - Use Case 25: Account Auto Coding (NLP)
Handles scoring and auto-coding SAP Financial Accounting line items from transaction text using
the #1 Ranked Champion Unsupervised Clustering Model (K-Means Clustering) and generating executive
3-sheet Excel reports matching the enterprise SAP template with streaming constant-memory xlsxwriter.
"""
import os
import datetime
import joblib
import pandas as pd
import numpy as np
import xlsxwriter


def score_with_single_model(df: pd.DataFrame, model, scaler, feature_cols: list) -> pd.DataFrame:
    """
    Applies the selected #1 Champion NLP Clustering ML model (K-Means Clustering) to calculate:
    - coding_cluster_id: ordered cluster tier (0 = High-Confidence Auto-Coded, 1 = Verified Expense/Statutory, 2 = Moderate-Confidence, 3 = Miscoded/Ambiguous)
    - coding_segment_name: human-readable Account Auto-Coding Segment
    - coding_exception_prob: continuous G/L miscoding/ambiguity probability in [0.0, 1.0] (8 decimals)
    - predicted_coding_exception: binary flag (1 for Miscoded / Ambiguous G/L coding exception, 0 for Verified Auto-Coded)
    - risk_category: executive triage tier (Critical Risk, High Risk, Medium Risk, Low Risk)
    """
    df_out = df.copy()

    X = df_out[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Process in memory-efficient chunks if dataset is large
    batch_size = 100000
    n_rows = len(df_out)
    all_probs = []
    all_raw_scores = []
    all_mapped_tiers = []
    all_segment_names = []
    all_preds = []

    for start_idx in range(0, n_rows, batch_size):
        end_idx = min(start_idx + batch_size, n_rows)
        X_chunk = X.iloc[start_idx:end_idx]
        X_chunk_scaled = scaler.transform(X_chunk)

        probs_chunk, raw_scores_chunk, mapped_tiers_chunk = model.predict_proba(X_chunk_scaled)
        _, segment_names_chunk = model.predict_clusters(X_chunk_scaled)
        preds_chunk = model.predict(X_chunk_scaled)

        all_probs.append(probs_chunk)
        all_raw_scores.append(raw_scores_chunk)
        all_mapped_tiers.append(mapped_tiers_chunk)
        all_segment_names.extend(segment_names_chunk)
        all_preds.append(preds_chunk)

    probs = np.concatenate(all_probs)
    mapped_tiers = np.concatenate(all_mapped_tiers)
    preds = np.concatenate(all_preds)

    df_out['coding_cluster_id'] = mapped_tiers.astype(int)
    df_out['coding_segment_name'] = all_segment_names
    df_out['coding_exception_prob'] = np.round(probs, 8)
    df_out['predicted_coding_exception'] = preds.astype(int)

    conditions = [
        (df_out['coding_cluster_id'] == 3) & (
            (df_out['is_nlp_gl_domain_conflict'] == 1) |
            (df_out['is_generic_provision_or_clearing_gl'] == 1) |
            (df_out['is_cryptic_or_code_only_sgtxt'] == 1) |
            (df_out['nlp_miscoding_risk_index'] >= 76.0)
        ),
        (df_out['coding_cluster_id'] == 3) | (df_out['coding_exception_prob'] >= 0.72),
        (df_out['coding_cluster_id'] == 2) | (df_out['coding_exception_prob'] >= 0.48),
    ]
    choices = ['Critical Risk', 'High Risk', 'Medium Risk']
    df_out['risk_category'] = np.select(conditions, choices, default='Low Risk')

    # Sort so highest G/L miscoding / ambiguity exceptions with highest financial exposure appear at the top
    df_out = df_out.sort_values(
        by=['coding_exception_prob', 'is_nlp_gl_domain_conflict', 'amount_local'],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return df_out


def export_enriched_excel_report(
    df: pd.DataFrame,
    output_path: str = "ML_25_Account_Auto_Coding_Enriched_Output.xlsx",
    insight_id: str = "ML_25",
    use_case: str = "Account Auto Coding",
    description: str = "Predict G/L accounts from transaction text using Unsupervised NLP (TF-IDF + LSA Semantic Embeddings + Lexical Chart-of-Accounts Taxonomy) & Clustering across SAP Accounting Document Segments (BSEG), G/L Account Master (SKA1, SKAT, SKB1), Posting Keys (TBSL), Document Types (T003T), and PO Short Texts (EKPO) across HANA_P2 and HANA_P3 with complete server coverage.",
    feature_importance_df: pd.DataFrame = None
) -> str:
    """
    Generates the executive enriched Excel report matching the enterprise SAP reference template
    using streaming constant-memory xlsxwriter to scale to 100% of the complete server dataset:
    - Sheet 1: ML_25_Enriched_Output(Predict)
      - Rows 1-5: Metadata block (Insight ID, Use Case, Description, Generated On, Row Count)
      - Row 6: Blank row
      - Row 7: Enterprise SAP column headers
      - Row 8+: Enriched SAP Account Auto-Coding records with NLP predicted G/L account, predicted G/L description,
                accounting domain, cluster assignments, confidence scores, probabilities, and risk tiers.
                Strict missing value handling: if missing, writes clean blank cell.
    - Sheet 2: Feature_Importance (All 32 engineered NLP & G/L Account Auto-Coding features with mathematical rationale)
    - Sheet 3: Columns Ignored (Audit justification for excluded non-feature columns)
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) if os.path.dirname(output_path) else ".", exist_ok=True)

    column_mapping = [
        ('schema_name', 'Schema Source'),
        ('company_code', 'Company Code (BUKRS)'),
        ('company_name', 'Company Name (BUTXT)'),
        ('doc_number', 'Document Number (BELNR)'),
        ('fiscal_year', 'Fiscal Year (GJAHR)'),
        ('line_item', 'Line Item (BUZEI)'),
        ('doc_type', 'Doc Type (BLART)'),
        ('doc_type_desc', 'Doc Type Description (LTEXT)'),
        ('posting_date', 'Posting Date (BUDAT)'),
        ('document_date', 'Document Date (BLDAT)'),
        ('posting_period', 'Period (MONAT)'),
        ('account_type', 'Acct Type (KOART)'),
        ('posting_key', 'Posting Key (BSCHL)'),
        ('posting_key_desc', 'Posting Key Name (TBSL)'),
        ('debit_credit_ind', 'Debit/Credit (SHKZG)'),
        ('transaction_text', 'Transaction Text (BSEG-SGTXT)'),
        ('po_short_text', 'PO Short Text (EKPO-TXZ01)'),
        ('gl_account', 'Posted GL Account (HKONT)'),
        ('gl_account_desc', 'Posted GL Description (SKAT-TXT50)'),
        ('gl_account_group', 'GL Account Group (SKA1-KTOKS)'),
        ('gl_account_type', 'GL Account Type (SKA1)'),
        ('predicted_gl_account', 'NLP Predicted GL Account (HKONT)'),
        ('predicted_gl_account_desc', 'NLP Predicted GL Description'),
        ('predicted_gl_domain', 'NLP Predicted Accounting Domain'),
        ('offsetting_gl_account', 'Offsetting Account (GKONT)'),
        ('offsetting_gl_desc', 'Offsetting GL Description'),
        ('cost_center', 'Cost Center (KOSTL)'),
        ('profit_center', 'Profit Center (PRCTR)'),
        ('plant', 'Plant (WERKS)'),
        ('vendor_id', 'Vendor ID (LIFNR)'),
        ('vendor_name', 'Vendor Name (LFA1-NAME1)'),
        ('po_number', 'PO Number (EBELN)'),
        ('hsn_sac_code', 'HSN/SAC Code'),
        ('doc_currency', 'Currency (WAERS)'),
        ('amount_local', 'Amount in Local Curr (DMBTR)'),
        ('sgtxt_char_length', 'Text Length (Chars)'),
        ('sgtxt_word_count', 'Text Word Count'),
        ('accounting_ref_token_count', 'Accounting Ref Tokens'),
        ('tfidf_active_ngrams', 'Active TF-IDF N-Grams'),
        ('text_clarity_score', 'Text Clarity Score (0-100)'),
        ('semantic_gl_confidence_score', 'NLP Semantic Confidence (0-100)'),
        ('coa_alignment_score', 'CoA Alignment Score (0-100)'),
        ('coding_tier_code', 'Coding Tier Code (T-S-C)'),
        ('nlp_coding_confidence_score', 'NLP Coding Confidence Score'),
        ('nlp_miscoding_risk_index', 'Miscoding Risk Index'),
        ('coding_cluster_id', 'ML Cluster ID'),
        ('coding_segment_name', 'ML Account Auto-Coding Segment'),
        ('coding_exception_prob', 'Coding Exception Prob'),
        ('predicted_coding_exception', 'Predicted Coding Exception'),
        ('risk_category', 'Risk Category')
    ]

    export_cols = []
    display_headers = []
    for internal_col, display_name in column_mapping:
        if internal_col in df.columns:
            export_cols.append(internal_col)
            display_headers.append(display_name)

    df_export = df[export_cols].copy()

    for dt_col in ['posting_date', 'document_date']:
        if dt_col in df_export.columns:
            df_export[dt_col] = pd.to_datetime(df_export[dt_col], errors='coerce').dt.strftime('%d-%m-%Y').fillna('')

    for num_col in [
        'amount_local', 'text_clarity_score', 'semantic_gl_confidence_score',
        'coa_alignment_score', 'nlp_coding_confidence_score', 'nlp_miscoding_risk_index'
    ]:
        if num_col in df_export.columns:
            df_export[num_col] = pd.to_numeric(df_export[num_col], errors='coerce').round(2)

    actual_path = output_path
    save_candidates = [
        output_path,
        "ML_25_Account_Auto_Coding_Enriched_Output_Clean.xlsx",
        "ML_25_Account_Auto_Coding_Enriched_Output_Final.xlsx",
        os.path.join("reports", os.path.basename(output_path)),
        f"ML_25_Account_Auto_Coding_Enriched_Output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    ]

    saved = False
    for candidate in save_candidates:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(candidate)) if os.path.dirname(candidate) else ".", exist_ok=True)
            wb = xlsxwriter.Workbook(candidate, {'constant_memory': True})
            wb.use_zip64()
            actual_path = candidate
            saved = True
            break
        except PermissionError:
            print(f"[WARNING] '{candidate}' is currently open in Excel / locked. Trying next file candidate...")
            continue

    if not saved:
        fallback = f"ML_25_Account_Coding_Output_{int(datetime.datetime.now().timestamp())}.xlsx"
        wb = xlsxwriter.Workbook(fallback, {'constant_memory': True})
        wb.use_zip64()
        actual_path = fallback

    # Format Styles
    meta_label_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 11, 'bold': True, 'font_color': '#000000'})
    meta_val_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 11, 'bold': False, 'font_color': '#000000'})
    header_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'bold': True, 'bg_color': '#F2F2F2', 'border': 1, 'align': 'center', 'valign': 'vcenter'})

    cell_text_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1})
    cell_center_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'align': 'center'})
    cell_num_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'align': 'right', 'num_format': '#,##0.00'})
    cell_int_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'align': 'center', 'num_format': '#,##0'})
    cell_prob_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'align': 'right', 'num_format': '0.00000000'})

    crit_risk_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'bold': True, 'font_color': '#9C0006', 'bg_color': '#FFC7CE', 'border': 1, 'align': 'center'})
    high_risk_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'bold': True, 'font_color': '#9C6500', 'bg_color': '#FFEB9C', 'border': 1, 'align': 'center'})
    med_risk_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'font_color': '#006100', 'bg_color': '#C6EFCE', 'border': 1, 'align': 'center'})
    low_risk_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'align': 'center'})

    # -------------------------------------------------------------
    # Sheet 1: ML_25_Enriched_Output(Predict)
    # -------------------------------------------------------------
    ws1 = wb.add_worksheet("ML_25_Enriched_Output(Predict)")
    ws1.set_row(6, 24)

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
        ws1.write(r_idx, 0, lbl, meta_label_fmt)
        ws1.write(r_idx, 1, val, meta_val_fmt)

    # Row 6 (index 5) is left blank intentionally as per enterprise SAP template

    # Row 7 (index 6): Headers
    for c_idx, header_text in enumerate(display_headers):
        ws1.write(6, c_idx, header_text, header_fmt)

    # Pre-determine column types for fast row-by-row cell formatting
    col_types = []
    for h in display_headers:
        if "Prob" in h:
            col_types.append("prob")
        elif any(k in h for k in ["Amount", "Score", "Confidence", "Index"]):
            col_types.append("num")
        elif any(k in h for k in ["Count", "Length", "Tokens", "N-Grams", "Cluster ID", "Exception"]) and "Prob" not in h:
            col_types.append("int")
        elif h in [
            "Schema Source", "Company Code (BUKRS)", "Document Number (BELNR)", "Fiscal Year (GJAHR)",
            "Line Item (BUZEI)", "Doc Type (BLART)", "Posting Date (BUDAT)",
            "Document Date (BLDAT)", "Period (MONAT)", "Acct Type (KOART)",
            "Posting Key (BSCHL)", "Debit/Credit (SHKZG)", "Posted GL Account (HKONT)",
            "GL Account Group (SKA1-KTOKS)", "GL Account Type (SKA1)",
            "NLP Predicted GL Account (HKONT)", "Offsetting Account (GKONT)",
            "Cost Center (KOSTL)", "Profit Center (PRCTR)", "Plant (WERKS)",
            "Vendor ID (LIFNR)", "PO Number (EBELN)", "HSN/SAC Code",
            "Currency (WAERS)", "Coding Tier Code (T-S-C)"
        ]:
            col_types.append("center")
        elif h == "Risk Category":
            col_types.append("risk")
        else:
            col_types.append("text")

    # Data Rows (index 7 onwards, capped at Excel maximum boundary 1,048,574)
    max_excel_row = 1048574
    data_matrix = df_export.values
    total_to_write = min(len(data_matrix), max_excel_row - 7)

    for r_offset in range(7, 7 + total_to_write):
        row_vals = data_matrix[r_offset - 7]
        for c_idx, val in enumerate(row_vals):
            ctype = col_types[c_idx]

            # Strict Missing Value Handling: if missing, write blank ""
            if val is None or pd.isna(val) or (isinstance(val, float) and np.isnan(val)):
                ws1.write_blank(r_offset, c_idx, None, cell_text_fmt)
                continue

            str_val = str(val).strip()
            if str_val in ('nan', 'None', 'NULL', '<NA>', 'NaT', ''):
                ws1.write_blank(r_offset, c_idx, None, cell_text_fmt)
                continue

            if ctype == "prob":
                try:
                    ws1.write_number(r_offset, c_idx, float(val), cell_prob_fmt)
                except (ValueError, TypeError):
                    ws1.write_string(r_offset, c_idx, str_val, cell_text_fmt)
            elif ctype == "num":
                try:
                    ws1.write_number(r_offset, c_idx, float(val), cell_num_fmt)
                except (ValueError, TypeError):
                    ws1.write_string(r_offset, c_idx, str_val, cell_text_fmt)
            elif ctype == "int":
                try:
                    ws1.write_number(r_offset, c_idx, int(round(float(val))), cell_int_fmt)
                except (ValueError, TypeError):
                    ws1.write_string(r_offset, c_idx, str_val, cell_text_fmt)
            elif ctype == "center":
                ws1.write_string(r_offset, c_idx, str_val, cell_center_fmt)
            elif ctype == "risk":
                if str_val == "Critical Risk":
                    ws1.write_string(r_offset, c_idx, str_val, crit_risk_fmt)
                elif str_val == "High Risk":
                    ws1.write_string(r_offset, c_idx, str_val, high_risk_fmt)
                elif str_val == "Medium Risk":
                    ws1.write_string(r_offset, c_idx, str_val, med_risk_fmt)
                else:
                    ws1.write_string(r_offset, c_idx, str_val, low_risk_fmt)
            else:
                ws1.write_string(r_offset, c_idx, str_val, cell_text_fmt)

    # Set column widths
    for c_idx, header_text in enumerate(display_headers):
        ws1.set_column(c_idx, c_idx, min(max(len(str(header_text)) + 3, 12), 44))

    ws1.freeze_panes(7, 0)

    # -------------------------------------------------------------
    # Sheet 2: Feature_Importance
    # -------------------------------------------------------------
    if feature_importance_df is None and os.path.exists("reports/feature_importance.csv"):
        feature_importance_df = pd.read_csv("reports/feature_importance.csv")

    if feature_importance_df is not None and len(feature_importance_df) > 0:
        ws2 = wb.add_worksheet("Feature_Importance")
        ws2.write(0, 0, f"{insight_id} | Feature_Importance", meta_label_fmt)

        feat_hdr_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 11, 'bold': True, 'font_color': '#FFFFFF', 'bg_color': '#2F5597', 'border': 1, 'valign': 'vcenter'})
        ws2.write(1, 0, "feature", feat_hdr_fmt)
        ws2.write(1, 1, "importance", feat_hdr_fmt)
        ws2.write(1, 2, "Direct Mathematical Influence On GL Account Auto Coding & CoA Governance:", feat_hdr_fmt)
        ws2.set_row(1, 22)

        feat_imp_fmt = wb.add_format({'font_name': 'Calibri', 'font_size': 10, 'border': 1, 'align': 'right', 'num_format': '0.000000000'})
        col_f = 'feature' if 'feature' in feature_importance_df.columns else feature_importance_df.columns[0]
        col_i = 'importance' if 'importance' in feature_importance_df.columns else feature_importance_df.columns[1]
        has_e = len(feature_importance_df.columns) > 2
        col_e = feature_importance_df.columns[2] if has_e else None

        for idx, r_data in feature_importance_df.iterrows():
            curr_row = idx + 2
            ws2.write_string(curr_row, 0, str(r_data[col_f]), cell_text_fmt)
            ws2.write_number(curr_row, 1, float(r_data[col_i]), feat_imp_fmt)
            expl_val = str(r_data[col_e]) if col_e and pd.notna(r_data[col_e]) else ""
            ws2.write_string(curr_row, 2, expl_val, cell_text_fmt)

        ws2.set_column(0, 0, 36)
        ws2.set_column(1, 1, 18)
        ws2.set_column(2, 2, 90)

    # -------------------------------------------------------------
    # Sheet 3: Columns Ignored
    # -------------------------------------------------------------
    ws3 = wb.add_worksheet("Columns Ignored")
    ws3.write(0, 0, "Columns Ignored / Excluded from Model Training:", meta_label_fmt)

    ign_headers = ["Ignored Column", "SAP Field Name", "Why It Was Ignored"]
    for c_idx, h_text in enumerate(ign_headers):
        ws3.write(2, c_idx, h_text, header_fmt)
    ws3.set_row(2, 20)

    ignored_rows = [
        ("Accounting Document & Line Item Keys", "BELNR, GJAHR, BUZEI", "Unique SAP FI document primary keys. Carry no intrinsic semantic signal on transaction narration or G/L account nature."),
        ("Raw Unvectorized Transaction Texts", "SGTXT, TXZ01, TXT20, TXT50", "Raw text strings cannot be fed directly into Euclidean/Gaussian clustering; transformed into TF-IDF n-grams, 5 TruncatedSVD semantic components, and lexical G/L domain affinities."),
        ("Assignment & Clearing Document IDs", "ZUONR, AUGBL, AUGDT", "Operational clearing and reference numbers retained strictly for FI audit drill-down."),
        ("Vendor, Customer & PO Identifiers", "LIFNR, KUNNR, EBELN, EBELP", "High-cardinality counterparty and purchasing document keys retained for sub-ledger traceability."),
        ("Raw Calendar Posting & Document Dates", "H_BUDAT, H_BLDAT, H_MONAT", "Calendar timestamps retained for fiscal period reporting; G/L account coding depends on transaction narration semantics."),
        ("Organizational & Location Codes", "BUKRS, WERKS, GSBER, BUPLA", "Company code, plant, and business place codes retained for organizational governance."),
        ("Payment Terms & Payment Method", "ZTERM, ZLSCH", "Treasury settlement parameters unrelated to G/L account classification from transaction text."),
        ("Tax Jurisdiction & Place of Supply", "PLC SUP, MWSKZ", "Regional GST place-of-supply metadata retained for tax compliance reporting."),
        ("Post-Hoc Benchmark Segment & Label", "benchmark_coding_segment, is_coding_exception", "Strictly excluded from model training so all 5 NLP clustering models remain 100% unsupervised and auto-code G/L accounts purely from input feature distributions.")
    ]

    for r_offset, (col_name, sap_field, reason) in enumerate(ignored_rows, start=3):
        ws3.write_string(r_offset, 0, col_name, cell_text_fmt)
        ws3.write_string(r_offset, 1, sap_field, cell_text_fmt)
        ws3.write_string(r_offset, 2, reason, cell_text_fmt)

    ws3.set_column(0, 0, 36)
    ws3.set_column(1, 1, 28)
    ws3.set_column(2, 2, 98)

    wb.close()
    print(f"[SUCCESS] Executive Enriched Excel Report saved to: {actual_path}")
    print(f"[INFO] Total Rows: {len(df_export):,}, Total Columns: {len(display_headers)}")
    return actual_path


def save_single_model_pipeline(model, scaler, feature_cols, suite=None, comparison_df=None, output_dir: str = "models"):
    """
    Saves the selected #1 Champion NLP Clustering Model (K-Means Clustering), scaler, and 5-model benchmark bundle to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    bundle = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'model_name': 'K-Means Clustering',
        'benchmark_ranking': comparison_df.to_dict(orient='records') if comparison_df is not None else None
    }
    champ_path = os.path.join(output_dir, "account_auto_coding_champion_model.joblib")
    bundle_path = os.path.join(output_dir, "account_auto_coding_model_bundle.joblib")
    km_path = os.path.join(output_dir, "champion_kmeans.joblib")
    scaler_path = os.path.join(output_dir, "robust_scaler.joblib")

    joblib.dump(bundle, champ_path)
    joblib.dump(bundle, bundle_path)
    joblib.dump(model.model, km_path)
    joblib.dump(scaler, scaler_path)

    if suite is not None and 'gmm' in suite.models:
        joblib.dump(suite.models['gmm'], os.path.join(output_dir, "challenger_gmm.joblib"))

    print(f"[SUCCESS] Saved champion NLP clustering model bundle to: {champ_path}")
