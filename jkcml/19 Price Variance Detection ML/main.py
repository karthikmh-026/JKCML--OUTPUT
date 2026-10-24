"""
Main Execution Script - Use Case 19: Price Variance Detection
Runs the focused end-to-end Machine Learning pipeline using ONLY the champion model (Isolation Forest):
1. Connects to PostgreSQL server and extracts SAP procurement records (EKPO, EKKO, EINA, EINE, MARA, MAKT, LFA1, T001W, T001) without row limits.
2. Performs feature engineering, reference price calculation, and variance detection.
3. Trains the dedicated single champion ML model (Isolation Forest).
4. Infers anomaly probabilities, predicted price variances, and risk tiers (Critical Risk, High Risk, Medium Risk, Low Risk).
5. Exports the executive enriched Excel report matching the enterprise SAP reference template with clean missing-value blank cells.
"""
import os
import argparse
import pandas as pd
from src.db_connector import load_config, extract_procurement_data
from src.preprocessing import clean_and_engineer_features, prepare_train_test_data
from src.models import SingleModelPriceVarianceDetector
from src.inference import score_with_single_model, export_enriched_excel_report, save_single_model_pipeline


def run_pipeline(limit: int = 0, contamination: float = 0.05, output_excel: str = "ML_19_Price_Variance_Detection_Enriched_Output_Clean.xlsx"):
    print("=" * 80)
    print("  USE CASE 19: PRICE VARIANCE DETECTION - SINGLE CHAMPION ML MODEL (ISOLATION FOREST)")
    print("=" * 80)

    # 1. Config & Data Extraction directly from PostgreSQL Server
    config = load_config()
    print("\n>>> STEP 1: Connecting to PostgreSQL Server & Extracting SAP Records...")
    print("    Extracting joined data from: EKPO, EKKO, EINA, EINE, MARA, MAKT, LFA1, T001W, T001")
    
    # If limit is 0 or None, extract ALL records
    actual_limit = None if (limit is None or limit <= 0) else limit
    raw_df = extract_procurement_data(limit=actual_limit, config=config)

    # 2. Data Cleaning & Feature Engineering
    print("\n>>> STEP 2: Performing Feature Engineering & Computing Reference Benchmark Prices...")
    featured_df = clean_and_engineer_features(raw_df)

    exception_cnt = featured_df['is_variance_exception'].sum()
    exception_rate = (exception_cnt / len(featured_df)) * 100
    print(f"[SUMMARY] Total Extracted Records: {len(featured_df):,}")
    print(f"[SUMMARY] Ground Truth Variance Exceptions: {exception_cnt:,} ({exception_rate:.2f}%)")

    # 3. Data Preparation & Robust Scaling
    print("\n>>> STEP 3: Preparing Feature Matrix & Robust Scaling...")
    data_dict = prepare_train_test_data(featured_df, test_size=0.20, random_state=42)
    X_train_scaled = data_dict['X_train_scaled']
    feature_cols = data_dict['feature_cols']
    scaler = data_dict['scaler']

    # 4. Train the SINGLE ML Model: Isolation Forest
    print("\n>>> STEP 4: Training Single Dedicated ML Model (Isolation Forest)...")
    detector = SingleModelPriceVarianceDetector(contamination=contamination, random_state=42)
    detector.fit(X_train_scaled)

    # 5. Score ALL Transactions
    print("\n>>> STEP 5: Scoring Procurement Transactions with Anomaly Probabilities & Risk Tiers...")
    scored_df = score_with_single_model(featured_df, detector, scaler, feature_cols)

    crit_cnt = (scored_df['risk_category'] == 'Critical Risk').sum()
    high_cnt = (scored_df['risk_category'] == 'High Risk').sum()
    med_cnt = (scored_df['risk_category'] == 'Medium Risk').sum()
    low_cnt = (scored_df['risk_category'] == 'Low Risk').sum()

    print("\n[EXECUTIVE RISK TRIAGE SUMMARY]:")
    print(f"  - Critical Risk: {crit_cnt:,} transactions (immediate audit action)")
    print(f"  - High Risk:     {high_cnt:,} transactions (investigation recommended)")
    print(f"  - Medium Risk:   {med_cnt:,} transactions (variance advisory)")
    print(f"  - Low Risk:      {low_cnt:,} transactions (within tolerance)")

    # 6. Compute Feature Importance & Explainability
    print("\n>>> STEP 6: Computing Feature Importance for Price Variance Detection...")
    from src.evaluate import compute_feature_importance
    feat_imp = compute_feature_importance(detector.model, feature_cols, X_train_scaled, y_true=data_dict['y_train'])
    print("\n=== FEATURE IMPORTANCE SCORES ===")
    print(feat_imp.to_string(index=False))
    os.makedirs("reports", exist_ok=True)
    feat_imp.to_csv("reports/feature_importance.csv", index=False)

    # 7. Export Enriched Excel File matching Reference Format (including Feature_Importance sheet)
    print("\n>>> STEP 7: Generating Executive Enriched Excel Report with Streaming XlsxWriter...")
    excel_path = export_enriched_excel_report(
        df=scored_df,
        output_path=output_excel,
        insight_id="ML_19",
        use_case="Price Variance Detection",
        description="Flag unusual purchase prices, enriched with purchasing document (EKPO, EKKO), info records (EINA, EINE), material master (MARA, MAKT), vendor master (LFA1), and plant (T001W).",
        feature_importance_df=feat_imp
    )

    # Also save a copy in reports/
    reports_excel = os.path.join("reports", os.path.basename(output_excel))
    if os.path.abspath(excel_path) != os.path.abspath(reports_excel):
        export_enriched_excel_report(
            df=scored_df,
            output_path=reports_excel,
            insight_id="ML_19",
            use_case="Price Variance Detection",
            description="Flag unusual purchase prices, enriched with purchasing document (EKPO, EKKO), info records (EINA, EINE), material master (MARA, MAKT), vendor master (LFA1), and plant (T001W).",
            feature_importance_df=feat_imp
        )

    # 8. Persist Model Artifacts
    print("\n>>> STEP 8: Persisting Model Pipeline Artifacts...")
    save_single_model_pipeline(detector, scaler, feature_cols, output_dir="models")

    print("\n" + "=" * 80)
    print("  PRICE VARIANCE PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  EXCEL REPORT GENERATED: {os.path.abspath(excel_path)}")
    print("=" * 80)
    return excel_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAP Price Variance Detection with Single ML Model & Excel Output")
    parser.add_argument("--limit", type=int, default=0, help="Number of records to extract (default: 0 for ALL records without limit)")
    parser.add_argument("--contamination", type=float, default=0.05, help="Anomaly contamination factor (default: 0.05)")
    parser.add_argument("--output", type=str, default="ML_19_Price_Variance_Detection_Enriched_Output_Clean.xlsx", help="Output Excel filename")
    args = parser.parse_args()

    run_pipeline(limit=args.limit, contamination=args.contamination, output_excel=args.output)
