"""
Main Execution Script - Use Case 8: Quality Inspection Failure Prediction
Runs the end-to-end Unsupervised Machine Learning pipeline across ALL DATA without limits:
1. Connects to PostgreSQL database (192.168.1.44:5432 / JKC_Input)
   and extracts all inspection lot records from HANA_P3 and HANA_P2 (QALS, QAVE, MAKT, MARA, MARC, MBEW, LFA1).
2. Cleans and engineers 32 domain features capturing inspection quantities, sampling intensity,
   scrap allocations, lead times, financial exposure, and vendor quality history.
   Ensures missing fields are kept blank rather than populated with placeholder strings.
3. Trains 5 Unsupervised Machine Learning Models strictly on input features (no target labels used in training).
4. Evaluates Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Specificity to rank all 5 models.
5. Selects the #1 Ranked Champion Model based on the ranking.
6. Scores all inspection lots with the selected #1 Champion Model, calculating calibrated
   Failure Probability P(Failure), binary predictions, 4 Quality Risk Tiers, and actionable containment recommendations.
7. Persists champion model artifacts and exports executive 5-sheet Excel reports and diagnostic charts.
"""
import os
import shutil
import argparse
import pandas as pd

from src.db_connector import load_config, extract_quality_inspection_data
from src.preprocessing import clean_and_engineer_features, prepare_train_test_data
from src.models import QualityInspectionFailureDetectionSuite, SingleModelQualityFailureDetector
from src.evaluate import (
    evaluate_models,
    plot_model_comparison,
    plot_roc_pr_curves,
    plot_failure_risk_distribution,
    compute_feature_importance,
    plot_feature_importance
)
from src.inference import (
    score_with_single_model,
    export_enriched_excel_report,
    save_single_model_pipeline
)


def run_pipeline(
    limit: int = None,
    contamination: float = 0.05,
    output_excel: str = "ML_8_Quality_Inspection_Failure_Prediction_Enriched_Output_Clean.xlsx"
):
    print("=" * 96)
    print("  USE CASE 8: QUALITY INSPECTION FAILURE PREDICTION & CLASSIFICATION")
    print("  ALL DATA (HANA_P3 & HANA_P2) -> 5 UNSUPERVISED ML MODELS -> #1 CHAMPION PRODUCTION SCORING")
    print("=" * 96)

    # 1. Config & Data Extraction directly from PostgreSQL Server
    config = load_config()
    print("\n>>> STEP 1: Connecting to PostgreSQL (192.168.1.44:5432 / HANA_P3 & HANA_P2) & Extracting All Inspection Lots...")
    print("    Extracting joined data from: QALS, QAVE, MAKT, MARA, MARC, MBEW, LFA1 without limits...")
    raw_df = extract_quality_inspection_data(limit=limit, config=config, use_cache=True)

    # 2. Data Cleaning & 32-Feature Engineering
    print("\n>>> STEP 2: Performing Data Cleaning & Engineering 32 Quality, Sampling, Scrap & Lead Time Features...")
    print("    Properly handling missing values (blank for missing data, robust imputation for mathematical features)...")
    featured_df = clean_and_engineer_features(raw_df)

    fail_cnt = int(featured_df['is_failure_exception'].sum())
    fail_rate = (fail_cnt / max(1, len(featured_df))) * 100.0
    dynamic_contam = max(0.02, min(0.15, fail_cnt / max(1, len(featured_df))))
    if contamination is not None:
        dynamic_contam = contamination

    print(f"[SUMMARY] Total Extracted Quality Inspection Lots: {len(featured_df):,}")
    print(f"[SUMMARY] Audit Benchmark Failure/Exception Lots: {fail_cnt:,} ({fail_rate:.2f}%)")
    print(f"[SUMMARY] Effective Contamination Factor: {dynamic_contam:.4f}")

    # 3. Data Preparation & Robust Scaling (Unsupervised X matrix)
    print("\n>>> STEP 3: Preparing 32-Feature Inspection Matrix & Applying RobustScaler (Unsupervised)...")
    data_dict = prepare_train_test_data(featured_df, test_size=0.20, random_state=42)
    X_train_scaled = data_dict['X_train_scaled']
    X_test_scaled = data_dict['X_test_scaled']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    feature_cols = data_dict['feature_cols']
    scaler = data_dict['scaler']

    # 4. Train 5 Unsupervised ML Models
    print("\n>>> STEP 4: Training 5 Unsupervised ML Models (Input-Only Training, No Output Labels Used)...")
    suite = QualityInspectionFailureDetectionSuite(contamination=dynamic_contam, random_state=42)
    suite.train_all(X_train_scaled)

    # 5. Evaluate & Rank 5 Models
    print("\n>>> STEP 5: Evaluating Accuracy & Ranking All 5 Unsupervised Models on Held-out Test Split...")
    # Evaluate on held-out test set (sample up to 20,000 for evaluation if large)
    if len(X_test_scaled) > 20000:
        rng = pd.Series(range(len(X_test_scaled))).sample(n=20000, random_state=42).values
        eval_X = X_test_scaled[rng]
        eval_y = y_test[rng]
    else:
        eval_X = X_test_scaled
        eval_y = y_test

    test_preds = suite.predict_all(eval_X)
    comparison_df = evaluate_models(test_preds, eval_y)

    print("\n" + "=" * 96)
    print("  5 UNSUPERVISED ML MODELS RANKING & ACCURACY TABLE (TEST SET EVALUATION)")
    print("=" * 96)
    print(comparison_df.to_string(index=False))
    print("=" * 96)

    os.makedirs("reports/figures", exist_ok=True)
    comparison_df.to_csv("reports/model_comparison_metrics.csv", index=False)
    plot_model_comparison(comparison_df, output_dir="reports/figures")
    plot_roc_pr_curves(test_preds, eval_y, output_dir="reports/figures")

    champion_model_name = comparison_df.iloc[0]['Model']
    champion_acc = comparison_df.iloc[0]['Accuracy'] * 100.0
    champion_prec = comparison_df.iloc[0]['Precision'] * 100.0
    champion_rec = comparison_df.iloc[0]['Recall'] * 100.0
    champion_f1 = comparison_df.iloc[0]['F1-Score']
    champion_auc = comparison_df.iloc[0]['ROC-AUC']
    print(f"\n[CHAMPION SELECTED] Rank #1 Model: {champion_model_name}")
    print(f"  -> Accuracy: {champion_acc:.2f}% | Precision: {champion_prec:.2f}% | Recall: {champion_rec:.2f}% | F1-Score: {champion_f1:.4f} | ROC-AUC: {champion_auc:.4f}")

    # 6. Fit Selected #1 Champion Model & Segment All Inspection Lots
    print(f"\n>>> STEP 6: Deploying Selected #1 Champion Model ({champion_model_name}) for Production Scoring...")
    champion_segmenter = SingleModelQualityFailureDetector(
        model_name=champion_model_name,
        contamination=dynamic_contam,
        random_state=42
    )
    champion_segmenter.fit(X_train_scaled)

    # 7. Score All Inspection Lots & Generate Recommendations
    print(f"\n>>> STEP 7: Scoring All {len(featured_df):,} Inspection Lots & Estimating Probability of Failure P(Failure)...")
    scored_df = score_with_single_model(champion_segmenter, featured_df, scaler, feature_cols)

    crit_cnt = int((scored_df['risk_category'] == 'Critical Failure Risk').sum())
    high_cnt = int((scored_df['risk_category'] == 'High Risk').sum())
    med_cnt = int((scored_df['risk_category'] == 'Medium Risk').sum())
    low_cnt = int((scored_df['risk_category'] == 'Low Risk').sum())

    print("\n[EXECUTIVE QUALITY RISK TRIAGE SUMMARY]:")
    print(f"  - Critical Failure Risk: {crit_cnt:,} lots (severe scrap/quarantine risk - immediate block & containment)")
    print(f"  - High Risk:             {high_cnt:,} lots (anomalous sampling/lead time - mandatory retest & CoA audit)")
    print(f"  - Medium Risk:           {med_cnt:,} lots (moderate delay/rare vendor - physical visual verification)")
    print(f"  - Low Risk:              {low_cnt:,} lots (compliant lots - standard automated release to inventory)")

    # Save high-risk watchlist CSV (top 5,000 lots)
    watchlist = scored_df[scored_df['risk_category'].isin(['Critical Failure Risk', 'High Risk'])].copy()
    watchlist.head(5000).to_csv("reports/quality_inspection_failure_audit_watchlist.csv", index=False)
    print(f"[WATCHLIST] Saved top high-risk inspection lots to reports/quality_inspection_failure_audit_watchlist.csv")

    # 8. Feature Importance
    print("\n>>> STEP 8: Computing 32-Feature Importance and Failure Explainability...")
    fi_df = compute_feature_importance(champion_segmenter.model, feature_cols, X_train_scaled, y_train)
    fi_df.to_csv("reports/feature_importance.csv", index=False)
    plot_feature_importance(fi_df, output_dir="reports/figures", top_n=15)
    print("\nTop 10 Most Influential Quality Failure Drivers:")
    print(fi_df.head(10)[['Rank', 'Feature', 'Importance', 'Relative_Weight_Pct']].to_string(index=False))

    # 9. Generate Charts & Excel Reports
    print(f"\n>>> STEP 9: Generating Distribution Visualizations & Multi-Sheet Executive Excel Reports ({len(scored_df):,} records)...")
    plot_failure_risk_distribution(scored_df, output_dir="reports/figures")
    export_enriched_excel_report(scored_df, output_path=output_excel, fi_df=fi_df, comp_df=comparison_df)

    # Save copy as standard filenames
    alt_excel = "ML_8_Quality_Inspection_Failure_Prediction_Enriched_Output.xlsx"
    shutil.copyfile(output_excel, alt_excel)
    print(f"[REPORT] Copied enriched Excel report to {alt_excel}")

    reports_clean = os.path.join("reports", os.path.basename(output_excel))
    reports_alt = os.path.join("reports", alt_excel)
    shutil.copyfile(output_excel, reports_clean)
    shutil.copyfile(output_excel, reports_alt)
    print(f"[REPORT] Copied reports to {reports_clean} and {reports_alt}")

    # 10. Persist Model Artifacts
    print("\n>>> STEP 10: Persisting Champion Model Pipeline & Scaler Artifacts to models/...")
    save_single_model_pipeline(champion_segmenter, scaler, feature_cols, output_dir="models")

    print("\n" + "=" * 96)
    print("  PIPELINE EXECUTION COMPLETE - ALL DELIVERABLES GENERATED SUCCESSFULLY!")
    print(f"  * #1 Champion Model: {champion_model_name}")
    print(f"  * Total Processed Lots: {len(scored_df):,}")
    print(f"  * Model Comparison Metrics: reports/model_comparison_metrics.csv")
    print(f"  * Feature Importance: reports/feature_importance.csv")
    print(f"  * High-Risk Watchlist: reports/quality_inspection_failure_audit_watchlist.csv")
    print(f"  * Visual Figures: reports/figures/ (model_comparison.png, roc_pr_curves.png, failure_risk_distribution.png, feature_importance.png)")
    print(f"  * Model Artifacts: models/ (champion_quality_failure_model.joblib, quality_failure_model_bundle.joblib, robust_scaler.joblib)")
    print(f"  * Executive Excel Reports: {output_excel}")
    print("=" * 96)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Quality Inspection Failure Prediction Pipeline on All Data")
    parser.add_argument("--limit", type=int, default=None, help="Number of records to extract (default: None for ALL DATA)")
    parser.add_argument("--contamination", type=float, default=0.05, help="Contamination factor (default: 0.05)")
    parser.add_argument("--output_excel", type=str, default="ML_8_Quality_Inspection_Failure_Prediction_Enriched_Output_Clean.xlsx", help="Output Excel filename")
    args = parser.parse_args()

    run_pipeline(limit=args.limit, contamination=args.contamination, output_excel=args.output_excel)
