"""
Inference and Reporting Module - Use Case 5: Credit Risk Scoring
Runs end-to-end scoring using the selected #1 Champion Model,
generates Probability of Customer Default P(Default), Credit Risk Tiers,
and exports comprehensive executive Excel reports with Feature Importance,
Model Comparison, Default Watchlist, and Ignored Columns.
"""
import os
import joblib
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


FEATURE_DESCRIPTIONS = {
    'credit_risk_index': '(Composite Credit Delinquency Index combining overdue ratio, severe days past due >90d, and credit limit strain)',
    'distress_momentum_score': '(Distress Acceleration Signal measuring acute worsening of delinquent balance, delivery blocks, and order freezes)',
    'has_central_posting_block': '(SAP Master Posting Block [SPERR in KNA1] triggered by legal action, bankruptcy, or persistent non-payment)',
    'solvency_score': '(Customer Liquidity & Clearing Health inverse function of clearing delays, overdue balance, and payment velocity)',
    'credit_limit_utilization': '(Credit Limit Utilization Ratio of outstanding receivables against authorized limit [KNB1.VLIBB])',
    'open_ar_log_val': '(Total Outstanding Exposure Size measuring log1p of active debt balance)',
    'max_clearing_delay_days': '(Worst Historical Payment Delay in days past due observed across cleared invoices [BSAD])',
    'is_credit_limit_breached': '(Binary flag indicating customer receivables currently exceed authorized credit limit)',
    'unsecured_credit_gap': '(Unhedged Debt Gap representing excess balance above credit limit without collateral)',
    'open_ar_item_density': '(Unpaid Invoice Concentration measuring open invoice count relative to total transaction history)',
    'late_settlement_rate': '(Historical Delinquency Rate measuring proportion of cleared invoices settled after due date)',
    'settlement_velocity_index': '(Cash Collection Velocity measuring cleared document count relative to total transactions)',
    'high_dunning_exposure_flag': '(Dunning/Dispute Flag indicating accounts with active payment blocks [ZLSPR] or aging > 45 days)',
    'customer_tenure_years': '(Customer Relationship Age in years from creation date [ERDAT])',
    'order_to_billing_ratio': '(Sales Order Velocity Strain measuring new order volume placed against billed revenue)',
    'avg_clearing_delay_days': '(Average Historical Clearing Delay in days across all settled payments [BSAD])',
    'blocked_balance_ratio': '(Ratio of disputed or blocked receivables to total open balance)',
    'historical_default_signal': '(Frequency and severity index of past severe late payment events [>60d])',
    'billing_volatility_ratio': '(Coefficient of variation of monthly billing invoices indicating revenue irregularity)',
    'blocked_ar_rate': '(Proportion of open invoice items carrying an active payment block)',
    'blocked_order_rate': '(Proportion of sales orders blocked for credit or delivery holds)',
    'return_memo_rate': '(Frequency of credit memos and returns relative to billing transactions)',
    'cleared_to_open_ratio': '(Multiple of historical cleared debt volume relative to current open exposure)',
    'overdue_ar_log_val': '(Log1p of total overdue accounts receivable past contracted due date)',
    'max_overdue_days': '(Longest days past due across currently open invoices)',
    'avg_overdue_days': '(Average days past due across open invoices)',
    'severe_overdue_ratio': '(Ratio of debt aged beyond 90 days to total open receivables)',
    'overdue_ratio': '(Ratio of overdue receivables to total open receivables)',
    'has_deletion_flag': '(Deletion Flag [LOEVM in KNA1 or KNB1] indicating marked for decommissioning)',
    'has_cocode_posting_block': '(Company Code Posting Block [SPERR in KNB1] restricting financial postings in legal entity)',
    'payment_terms_risk_score': '(Inherent risk profile of payment terms [immediate vs extended 60-90 days credit])',
    'cash_flow_strain_ratio': '(Ratio of overdue debt relative to total commercial transaction volume)'
}

IGNORED_COLUMNS_INFO = [
    ("Customer Account Number", "KUNNR", "Unique alphanumeric SAP Customer Master primary key. Carries no intrinsic mathematical signal on credit risk or insolvency probability."),
    ("Customer Name & Address Text", "NAME1, ORT01, PSTLZ, STRAS", "Free-text commercial name and geographic address strings retained strictly for credit controller identification and reporting."),
    ("Document Dates", "BUDAT, BLDAT, ERDAT, FKDAT, AUGDT, ZFBDT", "Raw calendar timestamps converted into relative temporal features (customer tenure, days past due, clearing delay) to ensure time-invariance."),
    ("Sales & Billing Document Numbers", "BELNR, VBELN", "Arbitrary transactional tracking IDs with zero ordinal or predictive value."),
    ("Company Code Master Descriptions", "BUTXT, SPRAS", "Descriptive textual metadata representing company entity names."),
    ("Internal Status Flags", "BUZEI, GJAHR, MONAT", "Internal accounting line item counters and fiscal periods without standalone behavioral credit signal.")
]


def score_with_single_model(model, df_featured: pd.DataFrame, scaler, feature_cols: list) -> pd.DataFrame:
    """
    Applies the fitted #1 Champion model to score all customer records.
    Returns enriched DataFrame with predictions, probabilities, and risk categories.
    """
    X = df_featured[feature_cols].values
    X_scaled = scaler.transform(X)

    tiers, tier_names = model.predict_clusters(X_scaled)
    pd_scores, raw_scores = model.predict_proba(X_scaled)
    preds = model.predict(X_scaled)
    categories = model.assign_risk_category(pd_scores)

    df_out = df_featured.copy()
    df_out['predicted_credit_tier'] = tiers
    df_out['credit_risk_tier_name'] = tier_names
    df_out['probability_of_default'] = np.round(pd_scores, 4)
    df_out['default_risk_score'] = np.round(raw_scores, 4)
    df_out['predicted_default_exception'] = preds
    df_out['credit_risk_category'] = categories

    def assign_action(row):
        cat = row['credit_risk_category']
        overdue = row['overdue_ar_balance']
        if "Critical" in cat:
            return "IMMEDIATE ACTION: Freeze credit limit, block delivery, trigger legal dunning notice"
        elif "High" in cat:
            return "HIGH PRIORITY: Place on credit watch, require advance payment, expedite AR collections"
        elif "Medium" in cat:
            if overdue > 5000:
                return "MONITOR: Send polite payment reminder, cap credit exposure at current level"
            return "STANDARD REVIEW: Review quarterly, verify financial statements"
        return "ROUTINE: Healthy account, consider credit limit increase upon commercial request"

    df_out['recommended_credit_action'] = df_out.apply(assign_action, axis=1)
    return df_out


def export_enriched_excel_report(
    df_scored: pd.DataFrame,
    output_path: str = "ML_5_Credit_Risk_Scoring_Enriched_Output_Clean.xlsx",
    fi_df: pd.DataFrame = None,
    comp_df: pd.DataFrame = None
):
    """
    Exports a professional multi-sheet executive Excel workbook including:
    - Sheet 1: Executive_Summary
    - Sheet 2: ML_05_Enriched_Output(Predict)
    - Sheet 3: Default_Watchlist
    - Sheet 4: Feature_Importance
    - Sheet 5: Model_Comparison
    - Sheet 6: Columns Ignored
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # remove default sheet

    header_font = Font(name='Calibri', size=11, bold=True, color='FFFFFF')
    header_fill = PatternFill(start_color='1F497D', end_color='1F497D', fill_type='solid')
    title_font = Font(name='Calibri', size=14, bold=True, color='1F497D')
    sub_font = Font(name='Calibri', size=11, bold=True, color='333333')
    thin_border = Border(
        left=Side(style='thin', color='D9D9D9'),
        right=Side(style='thin', color='D9D9D9'),
        top=Side(style='thin', color='D9D9D9'),
        bottom=Side(style='thin', color='D9D9D9')
    )

    # -------------------------------------------------------------
    # SHEET 1: Executive_Summary
    # -------------------------------------------------------------
    ws1 = wb.create_sheet(title="Executive_Summary")
    ws1.views.sheetView[0].showGridLines = True

    ws1.cell(row=1, column=1, value="USE CASE 5: CREDIT RISK SCORING & CUSTOMER DEFAULT PROBABILITY").font = title_font
    ws1.cell(row=2, column=1, value="Unsupervised ML Customer Credit Assessment | Enterprise SAP HANA Finance & SD").font = sub_font

    total_cust = len(df_scored)
    crit_count = int((df_scored['credit_risk_category'].str.contains("Critical")).sum())
    high_count = int((df_scored['credit_risk_category'].str.contains("High")).sum())
    at_risk_count = crit_count + high_count
    at_risk_pct = (at_risk_count / max(1, total_cust)) * 100.0

    total_open_ar = float(df_scored['open_ar_balance'].sum())
    total_overdue = float(df_scored['overdue_ar_balance'].sum())
    overdue_at_risk = float(df_scored[df_scored['credit_risk_category'].str.contains("High|Critical")]['overdue_ar_balance'].sum())

    ws1.cell(row=4, column=1, value="KEY EXECUTIVE METRICS").font = sub_font
    kpis = [
        ("Total Scored SAP Customers", f"{total_cust:,}"),
        ("High & Critical Default Risk Accounts", f"{at_risk_count:,} ({at_risk_pct:.2f}%)"),
        ("Total Outstanding AR Exposure", f"INR {total_open_ar:,.2f}"),
        ("Total Overdue Receivables", f"INR {total_overdue:,.2f}"),
        ("Overdue Receivables at Imminent Default Risk", f"INR {overdue_at_risk:,.2f}"),
    ]
    for r_idx, (label, val) in enumerate(kpis, start=5):
        ws1.cell(row=r_idx, column=1, value=label).font = Font(name='Calibri', size=11, bold=True)
        ws1.cell(row=r_idx, column=2, value=val).font = Font(name='Calibri', size=11, bold=True, color='1F497D')

    ws1.cell(row=12, column=1, value="CREDIT RISK TIER BREAKDOWN").font = sub_font
    tier_headers = ["Credit Risk Tier", "Customer Count", "Share (%)", "Total Open AR (INR)", "Total Overdue (INR)", "Avg Default Probability"]
    for c_idx, h in enumerate(tier_headers, start=1):
        cell = ws1.cell(row=14, column=c_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center')

    tier_group = df_scored.groupby('credit_risk_tier_name').agg({
        'customer_id': 'count',
        'open_ar_balance': 'sum',
        'overdue_ar_balance': 'sum',
        'probability_of_default': 'mean'
    }).reset_index()

    for r_idx, row in tier_group.iterrows():
        ws_r = 15 + r_idx
        ws1.cell(row=ws_r, column=1, value=str(row['credit_risk_tier_name']))
        cnt = int(row['customer_id'])
        ws1.cell(row=ws_r, column=2, value=cnt)
        ws1.cell(row=ws_r, column=3, value=round((cnt / max(1, total_cust)) * 100.0, 2))
        ws1.cell(row=ws_r, column=4, value=round(float(row['open_ar_balance']), 2))
        ws1.cell(row=ws_r, column=5, value=round(float(row['overdue_ar_balance']), 2))
        ws1.cell(row=ws_r, column=6, value=round(float(row['probability_of_default']), 4))

    for col in ws1.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws1.column_dimensions[col_letter].width = max(max_len + 3, 14)

    # -------------------------------------------------------------
    # SHEET 2: ML_05_Enriched_Output(Predict)
    # -------------------------------------------------------------
    ws2 = wb.create_sheet(title="ML_05_Enriched_Output(Predict)")
    ws2.views.sheetView[0].showGridLines = True

    export_cols = [
        ("customer_id", "Customer ID"),
        ("customer_name", "Customer Name"),
        ("company_code", "Company Code"),
        ("company_name", "Company Name"),
        ("country", "Country"),
        ("city", "City"),
        ("account_group_name", "Account Group"),
        ("credit_limit", "Credit Limit (INR)"),
        ("open_ar_balance", "Open AR Balance (INR)"),
        ("overdue_ar_balance", "Overdue AR (INR)"),
        ("max_overdue_days", "Max Overdue (Days)"),
        ("cleared_doc_count", "Cleared Invoices"),
        ("avg_clearing_delay_days", "Avg Clearing Delay (Days)"),
        ("probability_of_default", "P(Default)"),
        ("credit_risk_tier_name", "Credit Risk Tier"),
        ("credit_risk_category", "Risk Category"),
        ("recommended_credit_action", "Recommended Action")
    ]

    ws2.append([h for _, h in export_cols])
    for c_idx in range(1, len(export_cols) + 1):
        cell = ws2.cell(row=1, column=c_idx)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='center')

    col_keys = [k for k, _ in export_cols]
    df_export = df_scored.reindex(columns=col_keys).copy()

    # Clean text columns: replace NaN with blank ""
    for col in ['customer_id', 'customer_name', 'company_code', 'company_name', 'country', 'city', 'account_group_name', 'credit_risk_tier_name', 'credit_risk_category', 'recommended_credit_action']:
        if col in df_export.columns:
            df_export[col] = df_export[col].fillna('').astype(str).str.strip().replace({'nan': '', 'None': ''})
    if 'company_code' in df_export.columns:
        df_export['company_code'] = df_export['company_code'].str.replace(r'\.0$', '', regex=True)

    for row in df_export.itertuples(index=False):
        row_vals = []
        for k, val in zip(col_keys, row):
            if isinstance(val, (float, np.floating)):
                if np.isnan(val):
                    val = 0.0
                else:
                    val = round(val, 2) if k not in ["probability_of_default"] else round(val, 4)
            elif pd.isna(val) or val is None or str(val).lower() == 'nan':
                val = ""
            row_vals.append(val)
        ws2.append(row_vals)

    for col in ws2.columns:
        max_len = max(len(str(cell.value or '')) for cell in col[:100])
        col_letter = get_column_letter(col[0].column)
        ws2.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 45)

    # -------------------------------------------------------------
    # SHEET 3: Default_Watchlist
    # -------------------------------------------------------------
    ws3 = wb.create_sheet(title="Default_Watchlist")
    ws3.views.sheetView[0].showGridLines = True

    df_watchlist = df_scored[
        (df_scored['credit_risk_category'].str.contains("Critical|High")) |
        (df_scored['probability_of_default'] >= 0.50) |
        (df_scored['overdue_ar_balance'] > 50000)
    ].sort_values(by=['probability_of_default', 'overdue_ar_balance'], ascending=[False, False])

    watch_fill = PatternFill(start_color='C00000', end_color='C00000', fill_type='solid')
    ws3.append([h for _, h in export_cols])
    for c_idx in range(1, len(export_cols) + 1):
        cell = ws3.cell(row=1, column=c_idx)
        cell.font = header_font
        cell.fill = watch_fill
        cell.alignment = Alignment(horizontal='center')

    df_watch_export = df_watchlist.reindex(columns=col_keys).copy()
    for col in ['customer_id', 'customer_name', 'company_code', 'company_name', 'country', 'city', 'account_group_name', 'credit_risk_tier_name', 'credit_risk_category', 'recommended_credit_action']:
        if col in df_watch_export.columns:
            df_watch_export[col] = df_watch_export[col].fillna('').astype(str).str.strip().replace({'nan': '', 'None': ''})
    if 'company_code' in df_watch_export.columns:
        df_watch_export['company_code'] = df_watch_export['company_code'].str.replace(r'\.0$', '', regex=True)

    for row in df_watch_export.itertuples(index=False):
        row_vals = []
        for k, val in zip(col_keys, row):
            if isinstance(val, (float, np.floating)):
                if np.isnan(val):
                    val = 0.0
                else:
                    val = round(val, 2) if k not in ["probability_of_default"] else round(val, 4)
            elif pd.isna(val) or val is None or str(val).lower() == 'nan':
                val = ""
            row_vals.append(val)
        ws3.append(row_vals)

    for col in ws3.columns:
        max_len = max(len(str(cell.value or '')) for cell in col[:100])
        col_letter = get_column_letter(col[0].column)
        ws3.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 45)

    # -------------------------------------------------------------
    # SHEET 4: Feature_Importance
    # -------------------------------------------------------------
    ws4 = wb.create_sheet(title="Feature_Importance")
    ws4.views.sheetView[0].showGridLines = True

    ws4.merge_cells("A1:C1")
    ws4.cell(row=1, column=1, value="ML_05 | Feature_Importance").font = Font(name='Calibri', size=12, bold=True, color='FFFFFF')
    ws4.cell(row=1, column=1).fill = PatternFill(start_color='1F497D', end_color='1F497D', fill_type='solid')
    ws4.cell(row=1, column=1).alignment = Alignment(horizontal='left')

    fi_headers = ["feature", "importance", "Direct Mathematical Influence On Customer Default Probability & Credit Risk Scoring:"]
    for c_idx, h in enumerate(fi_headers, start=1):
        cell = ws4.cell(row=2, column=c_idx, value=h)
        cell.font = header_font
        cell.fill = PatternFill(start_color='2E75B6', end_color='2E75B6', fill_type='solid')
        cell.alignment = Alignment(horizontal='left' if c_idx != 2 else 'right')

    if fi_df is None:
        csv_path = os.path.join(os.path.dirname(os.path.abspath(output_path)), "reports", "feature_importance.csv")
        if os.path.exists(csv_path):
            fi_df = pd.read_csv(csv_path)

    if fi_df is not None:
        for r_idx, (_, fi_row) in enumerate(fi_df.iterrows(), start=3):
            feat_name = str(fi_row['Feature'])
            imp_val = float(fi_row['Importance'])
            desc = FEATURE_DESCRIPTIONS.get(feat_name, f"(Normalized credit risk feature: {feat_name})")

            c1 = ws4.cell(row=r_idx, column=1, value=feat_name)
            c2 = ws4.cell(row=r_idx, column=2, value=imp_val)
            c3 = ws4.cell(row=r_idx, column=3, value=desc)

            c1.font = Font(name='Calibri', size=10)
            c2.font = Font(name='Calibri', size=10)
            c3.font = Font(name='Calibri', size=10, italic=True)
            c1.border = thin_border
            c2.border = thin_border
            c3.border = thin_border

    ws4.column_dimensions['A'].width = 32
    ws4.column_dimensions['B'].width = 16
    ws4.column_dimensions['C'].width = 95

    # -------------------------------------------------------------
    # SHEET 5: Model_Comparison
    # -------------------------------------------------------------
    ws5 = wb.create_sheet(title="Model_Comparison")
    ws5.views.sheetView[0].showGridLines = True

    ws5.merge_cells("A1:J1")
    ws5.cell(row=1, column=1, value="USE CASE 5: 5 UNSUPERVISED ML MODELS RANKING & EVALUATION TABLE").font = Font(name='Calibri', size=12, bold=True, color='FFFFFF')
    ws5.cell(row=1, column=1).fill = PatternFill(start_color='1F497D', end_color='1F497D', fill_type='solid')

    if comp_df is None:
        comp_csv = os.path.join(os.path.dirname(os.path.abspath(output_path)), "reports", "model_comparison_metrics.csv")
        if os.path.exists(comp_csv):
            comp_df = pd.read_csv(comp_csv)

    if comp_df is not None:
        comp_headers = list(comp_df.columns)
        for c_idx, h in enumerate(comp_headers, start=1):
            cell = ws5.cell(row=2, column=c_idx, value=h)
            cell.font = header_font
            cell.fill = PatternFill(start_color='2E75B6', end_color='2E75B6', fill_type='solid')
            cell.alignment = Alignment(horizontal='center')

        for r_idx, (_, row_data) in enumerate(comp_df.iterrows(), start=3):
            for c_idx, h in enumerate(comp_headers, start=1):
                val = row_data[h]
                cell = ws5.cell(row=r_idx, column=c_idx, value=val)
                cell.font = Font(name='Calibri', size=10, bold=(r_idx == 3))  # bold Champion
                cell.border = thin_border
                if r_idx == 3:
                    cell.fill = PatternFill(start_color='E2EFDA', end_color='E2EFDA', fill_type='solid')  # light green champion fill

        for col in ws5.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = get_column_letter(col[0].column)
            ws5.column_dimensions[col_letter].width = max(max_len + 3, 14)

    # -------------------------------------------------------------
    # SHEET 6: Columns Ignored
    # -------------------------------------------------------------
    ws6 = wb.create_sheet(title="Columns Ignored")
    ws6.views.sheetView[0].showGridLines = True

    ws6.merge_cells("A1:C1")
    ws6.cell(row=1, column=1, value="Columns Ignored / Excluded from Model Training:").font = Font(name='Calibri', size=12, bold=True, color='1F497D')

    ign_headers = ["Ignored Column", "SAP Field Name", "Why It Was Ignored"]
    for c_idx, h in enumerate(ign_headers, start=1):
        cell = ws6.cell(row=3, column=c_idx, value=h)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal='left')

    for r_idx, (col_name, sap_field, reason) in enumerate(IGNORED_COLUMNS_INFO, start=4):
        c1 = ws6.cell(row=r_idx, column=1, value=col_name)
        c2 = ws6.cell(row=r_idx, column=2, value=sap_field)
        c3 = ws6.cell(row=r_idx, column=3, value=reason)
        c1.border = thin_border
        c2.border = thin_border
        c3.border = thin_border
        c1.font = Font(name='Calibri', size=10, bold=True)
        c2.font = Font(name='Calibri', size=10)
        c3.font = Font(name='Calibri', size=10)

    ws6.column_dimensions['A'].width = 35
    ws6.column_dimensions['B'].width = 30
    ws6.column_dimensions['C'].width = 75

    saved_paths = []
    try:
        wb.save(output_path)
        saved_paths.append(output_path)
        print(f"[REPORT] Executive Excel Report generated successfully with 6 sheets: {output_path}")
    except PermissionError:
        print(f"[WARNING] Could not overwrite {output_path} because it is currently open in Microsoft Excel.")
        alt_save_path = output_path.replace(".xlsx", "_Updated.xlsx")
        try:
            wb.save(alt_save_path)
            saved_paths.append(alt_save_path)
            print(f"[REPORT] Saved to alternate unlocked file: {alt_save_path}")
        except PermissionError:
            pass

    # Also try saving the standard copy if different
    alt_mirror = output_path.replace("_Clean", "")
    if alt_mirror != output_path:
        try:
            wb.save(alt_mirror)
            saved_paths.append(alt_mirror)
            print(f"[REPORT] Mirror copy saved to: {alt_mirror}")
        except PermissionError:
            print(f"[WARNING] Could not overwrite mirror {alt_mirror} because it is currently open in Microsoft Excel.")


def save_single_model_pipeline(model, scaler, feature_cols: list, output_dir: str = "models"):
    """
    Persists trained model and scaler artifacts using joblib.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "champion_credit_risk_model.joblib")
    scaler_path = os.path.join(output_dir, "credit_risk_scaler.joblib")
    bundle_path = os.path.join(output_dir, "credit_risk_model_bundle.joblib")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    bundle = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'model_name': getattr(model, 'model_name', 'K-Means Clustering'),
        'threshold': getattr(model, 'threshold_', 0.60),
        'rank_map': getattr(model, 'rank_map_', {})
    }
    joblib.dump(bundle, bundle_path)
    print(f"[MODEL] Persisted champion model to: {model_path}")
    print(f"[MODEL] Persisted model bundle to: {bundle_path}")
