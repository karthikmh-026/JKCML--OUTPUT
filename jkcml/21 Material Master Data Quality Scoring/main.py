"""
Main Execution Script - Use Case 21: Material Master Data Quality Scoring
Runs the end-to-end Unsupervised Machine Learning pipeline without limits across HANA_P2 and HANA_P3:
1. Connects to PostgreSQL server (192.168.1.44:5432 / JKC_Input / HANA_P2 & HANA_P3) and extracts ALL SAP Material Master records
   from MARA, MAKT, MARC, MBEW, MARM, EINA, T134T, T001W, T001.
2. Performs meticulous missing-value handling and engineers 32 domain-specific features for Material Master Data Quality Scoring.
3. Trains 5 Unsupervised Anomaly Detection ML models (Isolation Forest, Local Outlier Factor, One-Class SVM,
   DBSCAN, Autoencoder) strictly on input features X (no output labels used during training).
4. Checks Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Specificity to rank all 5 models
   and selects the #1 Champion Model based on the ranking.
5. Uses the selected #1 Champion Model (Isolation Forest) to score ALL extracted material records with
   calibrated data quality anomaly probabilities, binary predictions, health scores, and risk categories
   (Critical, High, Medium, Low Risk).
6. Computes 32-feature importance and exports the executive 3-sheet Excel reports and visual evaluation charts.
"""
import os
import argparse
import pandas as pd
from src.db_connector import load_config, extract_material_master_data
from src.preprocessing import clean_and_engineer_features, prepare_train_test_data
from src.models import AnomalyDetectionSuite, SingleModelMaterialQualityDetector
from src.evaluate import (
    evaluate_models,
    plot_model_comparison,
    plot_roc_pr_curves,
    compute_feature_importance
)
from src.inference import (
    score_with_single_model,
    export_enriched_excel_report,
    save_single_model_pipeline
)


def run_pipeline(
    limit: int = None,
    schemas: list = None,
    contamination: float = 0.05,
    output_excel: str = "ML_21_Material_Master_Data_Quality_Scoring_Enriched_Output_Clean.xlsx"
):
    print("=" * 96)
    print("  USE CASE 21: MATERIAL MASTER DATA QUALITY SCORING - 5 UNSUPERVISED ML MODELS -> #1 CHAMPION")
    print("  SERVER: 192.168.1.44:5432 / JKC_Input | SCHEMAS: HANA_P2 & HANA_P3 | LIMIT: ALL RECORDS")
    print("=" * 96)

    # 1. Config & Data Extraction directly from PostgreSQL Server
    config = load_config()
    if schemas is None:
        schemas = config["database"].get("schemas", ["HANA_P2", "HANA_P3"])

    print(f"\n>>> STEP 1: Connecting to PostgreSQL Server & Extracting ALL SAP Materials from Schemas {schemas}...")
    print("    Extracting joined data from: MARA, MAKT, MARC, MBEW, MARM, EINA, T134T, T001W, T001")
    raw_df = extract_material_master_data(limit=limit, schemas=schemas, config=config)

    # 2. Data Cleaning & 32-Feature Engineering with Missing-Value Handling
    print("\n>>> STEP 2: Performing Data Cleaning, Missing-Value Handling & Engineering 32 Quality Features...")
    featured_df = clean_and_engineer_features(raw_df)

    exception_cnt = int(featured_df['is_quality_exception'].sum())
    exception_rate = (exception_cnt / len(featured_df)) * 100.0
    print(f"[SUMMARY] Total Extracted SAP Material Master Records: {len(featured_df):,}")
    print(f"[SUMMARY] Audit Benchmark Material Quality Exceptions: {exception_cnt:,} ({exception_rate:.2f}%)")

    # 3. Data Preparation & Robust Scaling (Unsupervised X matrix)
    print("\n>>> STEP 3: Preparing 32-Feature Matrix & Applying RobustScaler (Unsupervised)...")
    data_dict = prepare_train_test_data(featured_df, test_size=0.20, random_state=42)
    X_train_scaled = data_dict['X_train_scaled']
    X_test_scaled = data_dict['X_test_scaled']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    feature_cols = data_dict['feature_cols']
    scaler = data_dict['scaler']

    # 4. Train & Compare 5 Unsupervised Anomaly Detection Models
    print("\n>>> STEP 4: Training 5 Unsupervised ML Models (Input-Only Training, No Output Labels)...")
    suite = AnomalyDetectionSuite(contamination=contamination, random_state=42)
    suite.train_all(X_train_scaled)

    print("\n>>> STEP 5: Evaluating Accuracy & Ranking the 5 Unsupervised ML Models on Test Split...")
    test_preds = suite.predict_all(X_test_scaled)
    comparison_df = evaluate_models(test_preds, y_test)

    print("\n" + "=" * 96)
    print("  5 UNSUPERVISED ML MODELS RANKING & ACCURACY TABLE (TEST SET EVALUATION)")
    print("=" * 96)
    print(comparison_df.to_string(index=False))
    print("=" * 96)

    os.makedirs("reports/figures", exist_ok=True)
    comparison_df.to_csv("reports/model_comparison_metrics.csv", index=False)
    plot_model_comparison(comparison_df, output_dir="reports/figures")
    plot_roc_pr_curves(test_preds, y_test, output_dir="reports/figures")

    champion_model_name = comparison_df.iloc[0]['Model']
    champion_acc = comparison_df.iloc[0]['Accuracy'] * 100.0
    champion_prec = comparison_df.iloc[0]['Precision'] * 100.0
    champion_f1 = comparison_df.iloc[0]['F1-Score']
    champion_auc = comparison_df.iloc[0]['ROC-AUC']
    print(f"\n[CHAMPION SELECTED] Rank #1 Model: {champion_model_name}")
    print(f"  -> Accuracy: {champion_acc:.2f}% | Precision: {champion_prec:.2f}% | F1-Score: {champion_f1:.4f} | ROC-AUC: {champion_auc:.4f}")

    # 6. Fit Selected #1 Champion Model (Isolation Forest) & Score ALL Material Records
    print(f"\n>>> STEP 6: Scoring ALL {len(featured_df):,} SAP Material Records with #1 Champion Model ({champion_model_name})...")
    detector = SingleModelMaterialQualityDetector(
        contamination=contamination,
        random_state=42,
        pretrained_model=suite.models['isolation_forest']
    )
    detector.fit(X_train_scaled)
    scored_df = score_with_single_model(featured_df, detector, scaler, feature_cols)

    crit_cnt = int((scored_df['risk_category'] == 'Critical Risk').sum())
    high_cnt = int((scored_df['risk_category'] == 'High Risk').sum())
    med_cnt = int((scored_df['risk_category'] == 'Medium Risk').sum())
    low_cnt = int((scored_df['risk_category'] == 'Low Risk').sum())

    print("\n[EXECUTIVE MATERIAL MASTER DATA QUALITY TRIAGE SUMMARY]:")
    print(f"  - Critical Risk: {crit_cnt:,} materials (severe valuation/stock/governance corruption - immediate MDM remediation)")
    print(f"  - High Risk:     {high_cnt:,} materials (missing mandatory accounting/purchasing/weight attributes)")
    print(f"  - Medium Risk:   {med_cnt:,} materials (duplicate/cryptic descriptions or incomplete secondary views)")
    print(f"  - Low Risk:      {low_cnt:,} materials (compliant & complete material master records)")

    # Save top high-risk watchlist CSV
    scored_df[scored_df['risk_category'].isin(['Critical Risk', 'High Risk', 'Medium Risk'])].head(5000).to_csv(
        "reports/material_quality_audit_watchlist.csv", index=False
    )

    # 7. Compute 32-Feature Importance & Explainability
    print("\n>>> STEP 7: Computing 32-Feature Importance for Material Master Data Quality Scoring...")
    feat_imp = compute_feature_importance(detector.model, feature_cols, X_train_scaled, y_true=y_train)
    print("\n=== TOP 15 FEATURE IMPORTANCE SCORES ===")
    print(feat_imp.head(15).to_string(index=False))
    feat_imp.to_csv("reports/feature_importance.csv", index=False)

    # 8. Export Enriched 3-Sheet Excel Reports
    print("\n>>> STEP 8: Generating Executive 3-Sheet Enriched Excel Report for ALL records...")
    desc_text = (
        "Detect incomplete or incorrect material records across HANA_P2 and HANA_P3, enriched with General Material Data (MARA), "
        "Material Descriptions (MAKT), Plant/MRP/Purchasing Views (MARC), Accounting/Valuation Data (MBEW), "
        "Units of Measure (MARM), Purchasing Info Records (EINA), Plant (T001W), and Company Code (T001)."
    )
    excel_path = export_enriched_excel_report(
        df=scored_df,
        output_path=output_excel,
        insight_id="ML_21",
        use_case="Material Master Data Quality Scoring",
        description=desc_text,
        feature_importance_df=feat_imp
    )

    # Also save standard named files in root and reports/
    for extra_path in [
        "ML_21_Material_Master_Data_Quality_Scoring_Enriched_Output.xlsx",
        os.path.join("reports", "ML_21_Material_Master_Data_Quality_Scoring_Enriched_Output.xlsx"),
        os.path.join("reports", os.path.basename(output_excel))
    ]:
        if os.path.abspath(extra_path) != os.path.abspath(excel_path):
            export_enriched_excel_report(
                df=scored_df,
                output_path=extra_path,
                insight_id="ML_21",
                use_case="Material Master Data Quality Scoring",
                description=desc_text,
                feature_importance_df=feat_imp
            )

    # 9. Persist Model Artifacts
    print("\n>>> STEP 9: Persisting Model Pipeline & Benchmark Artifacts...")
    save_single_model_pipeline(
        detector, scaler, feature_cols, suite=suite, comparison_df=comparison_df, output_dir="models"
    )

    print("\n" + "=" * 96)
    print("  MATERIAL MASTER DATA QUALITY SCORING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  TOTAL RECORDS PROCESSED: {len(scored_df):,}")
    print(f"  CHAMPION MODEL: {champion_model_name} (Accuracy: {champion_acc:.2f}%, Precision: {champion_prec:.2f}%)")
    print(f"  EXCEL REPORT GENERATED: {os.path.abspath(excel_path)}")
    print("=" * 96)
    return excel_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAP Material Master Data Quality Scoring with 5 Unsupervised ML Models & Champion Excel Output")
    parser.add_argument("--limit", type=int, default=None, help="Number of records to extract (default: None for all data)")
    parser.add_argument("--schemas", nargs="+", default=["HANA_P2", "HANA_P3"], help="Schemas to extract from (default: HANA_P2 HANA_P3)")
    parser.add_argument("--contamination", type=float, default=0.05, help="Anomaly contamination factor (default: 0.05)")
    parser.add_argument("--output", type=str, default="ML_21_Material_Master_Data_Quality_Scoring_Enriched_Output_Clean.xlsx", help="Output Excel filename")
    args = parser.parse_args()

    run_pipeline(limit=args.limit, schemas=args.schemas, contamination=args.contamination, output_excel=args.output)
