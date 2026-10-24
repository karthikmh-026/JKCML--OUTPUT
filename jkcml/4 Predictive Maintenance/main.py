"""
Main Execution Script - Use Case 4: Predictive Maintenance Classification
(Predict Equipment Failures Before Breakdown)
Runs the end-to-end 100% Unsupervised Machine Learning pipeline on ALL data across HANA_P2 and HANA_P3:
1. Connects to PostgreSQL server (192.168.1.44:5432 / JKC_Input) and extracts SAP Plant
   Maintenance order master, scheduling, confirmations, spare parts, and work center data from
   AUFK, AFKO, AFRU, AUFM, CRHD, T001W, T001 across both HANA_P2 and HANA_P3 schemas without any limits.
2. Cleans data and engineers 32 domain-specific features capturing machine operational stress,
   duration variances, execution delays, spare parts replacement costs, and NLP degradation signals.
3. Prepares unsupervised train/test splits and scales features with RobustScaler.
4. Trains 5 Unsupervised Machine Learning Models (Isolation Forest, Local Outlier Factor, One-Class SVM,
   Gaussian Mixture Model, K-Means Clustering) strictly on input features X without target labels.
5. Evaluates Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Specificity on the test split,
   ranks all 5 models, and objectively selects the #1 Champion Model.
6. Deploys the selected #1 Champion Model to score all equipment records, categorizing failure risk tiers
   (Critical Failure Imminent, High Breakdown Risk, Moderate Wear, Healthy) and generating prescriptive actions.
7. Computes 32-feature importance and exports executive multi-sheet Excel reports (handling missing values cleanly),
   CSV watchlists, and publication-ready evaluation charts.
"""
import os
import shutil
import argparse
import pandas as pd
from src.db_connector import load_config, extract_predictive_maintenance_data
from src.preprocessing import clean_and_engineer_features, prepare_train_test_data
from src.models import PredictiveMaintenanceUnsupervisedSuite, SingleModelPredictiveMaintenanceClassifier
from src.evaluate import (
    evaluate_models,
    plot_model_comparison,
    plot_roc_pr_curves,
    plot_confusion_matrix,
    compute_feature_importance
)
from src.inference import (
    score_with_single_model,
    export_enriched_excel_report,
    save_single_model_pipeline,
    clean_missing_for_export
)


def run_pipeline(
    limit: int = None,
    contamination: float = 0.25,
    output_excel: str = "ML_4_Predictive_Maintenance_Enriched_Output_Clean.xlsx"
):
    print("=" * 100)
    print("  USE CASE 4: PREDICTIVE MAINTENANCE CLASSIFICATION - PREDICT EQUIPMENT FAILURES BEFORE BREAKDOWN")
    print("  (100% UNSUPERVISED MACHINE LEARNING: 5 ML MODELS -> ACCURACY & RANKING -> #1 CHAMPION MODEL)")
    print("  (DATA SOURCE: ALL DATA FROM POSTGRESQL JKC_INPUT / HANA_P2 & HANA_P3 - NO LIMITS)")
    print("=" * 100)

    # 1. Config & Data Extraction from PostgreSQL Server
    config = load_config()
    print("\n>>> STEP 1: Connecting to PostgreSQL Server (192.168.1.44:5432 / JKC_Input) & Extracting ALL SAP Data...")
    print("    Extracting joined data from: AUFK, AFKO, AFRU, AUFM, CRHD, T001W, T001 across HANA_P2 & HANA_P3...")
    raw_df = extract_predictive_maintenance_data(limit=limit, config=config)
    print(f"    Successfully extracted {len(raw_df):,} SAP Plant Maintenance records (100% full dataset).")

    # 2. Data Cleaning & 32-Feature Engineering
    print("\n>>> STEP 2: Performing Data Cleaning & Engineering 32 Equipment Degradation & Operational Features...")
    featured_df = clean_and_engineer_features(raw_df)

    benchmark_exceptions = int(featured_df['is_equipment_failure_exception'].sum())
    exception_rate = (benchmark_exceptions / len(featured_df)) * 100.0
    actual_contamination = max(0.05, min(0.40, benchmark_exceptions / max(1, len(featured_df))))
    print(f"[SUMMARY] Total Extracted Records: {len(featured_df):,}")
    print(f"[SUMMARY] Breakdown & High-Stress Failure Benchmark Exceptions: {benchmark_exceptions:,} ({exception_rate:.2f}%)")
    print(f"[SUMMARY] Calibrated Model Contamination Parameter: {actual_contamination:.4f}")

    # 3. Data Preparation & Robust Scaling (Unsupervised X matrix)
    print("\n>>> STEP 3: Preparing 32-Feature Matrix & Fitting RobustScaler (Unsupervised)...")
    data_dict = prepare_train_test_data(featured_df, test_size=0.20, random_state=42)
    X_train_scaled = data_dict['X_train_scaled']
    X_test_scaled = data_dict['X_test_scaled']
    X_test_df = data_dict['X_test_df']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    feature_cols = data_dict['feature_cols']
    scaler = data_dict['scaler']

    # 4. Train 5 Unsupervised Machine Learning Models
    print("\n>>> STEP 4: Training 5 Unsupervised ML Models (Input-Only Training, No Output Labels)...")
    suite = PredictiveMaintenanceUnsupervisedSuite(contamination=actual_contamination, random_state=42)
    suite.train_all(X_train_scaled)

    # 5. Evaluate Accuracy & Rank All 5 Models on Test Set
    print("\n>>> STEP 5: Evaluating Accuracy & Ranking the 5 Unsupervised ML Models on Test Split...")
    test_preds = suite.predict_all(X_test_scaled)
    comparison_df = evaluate_models(test_preds, y_test)

    print("\n" + "=" * 100)
    print("  5 UNSUPERVISED ML MODELS ACCURACY & PERFORMANCE RANKING TABLE (TEST SET EVALUATION)")
    print("=" * 100)
    print(comparison_df.to_string(index=False))
    print("=" * 100)

    os.makedirs("reports/figures", exist_ok=True)
    comparison_df.to_csv("reports/model_comparison_metrics.csv", index=False)
    plot_model_comparison(comparison_df, output_dir="reports/figures")
    plot_roc_pr_curves(test_preds, y_test, output_dir="reports/figures")

    # Select #1 Champion Model
    champion_row = comparison_df.iloc[0]
    champion_key = champion_row['Model_Key']
    champion_name = champion_row['Model']
    champion_acc = champion_row['Accuracy'] * 100.0
    champion_prec = champion_row['Precision'] * 100.0
    champion_rec = champion_row['Recall'] * 100.0
    champion_f1 = champion_row['F1-Score']
    champion_auc = champion_row['ROC-AUC']

    print(f"\n[CHAMPION SELECTED] Rank #1 Champion Model: {champion_name} ({champion_key})")
    print(f"  -> Accuracy: {champion_acc:.2f}% | Precision: {champion_prec:.2f}% | Recall: {champion_rec:.2f}% | F1: {champion_f1:.4f} | ROC-AUC: {champion_auc:.4f}")

    # Plot Confusion Matrix for Champion Model
    plot_confusion_matrix(
        y_test, test_preds[champion_key]['preds'],
        champion_name=champion_name,
        output_dir="reports/figures"
    )

    # 6. Fit Selected Champion Classifier & Save Artifacts
    print(f"\n>>> STEP 6: Deploying #1 Champion Model ({champion_name}) for Production Scoring...")
    champion_model_instance = suite.models[champion_key]
    champion_threshold = suite.thresholds[champion_key]
    champion_meta = suite.meta[champion_key]

    champion_classifier = SingleModelPredictiveMaintenanceClassifier(
        model_key=champion_key,
        model_instance=champion_model_instance,
        scaler=scaler,
        threshold=champion_threshold,
        meta=champion_meta,
        feature_cols=feature_cols
    )

    save_single_model_pipeline(
        champion_classifier,
        model_path="models/champion_predictive_maintenance_model.joblib",
        scaler_path="models/predictive_maintenance_scaler.joblib"
    )

    # 7. Compute Feature Importance
    print("\n>>> STEP 7: Computing 32-Feature Contribution & Explainability Ranking...")
    fi_df = compute_feature_importance(champion_classifier, X_test_df, feature_cols)
    fi_df.to_csv("reports/feature_importance.csv", index=False)
    print("\nTop 10 Most Important Features Driving Equipment Failure Predictions:")
    print(fi_df.head(10).to_string(index=False))

    # 8. Score All Extracted Maintenance Records
    print("\n>>> STEP 8: Scoring All Records with Champion Model & Generating Prescriptive Actions...")
    enriched_df = score_with_single_model(champion_classifier, featured_df)

    predicted_exceptions = int(enriched_df['predicted_failure_exception'].sum())
    pred_rate = (predicted_exceptions / len(enriched_df)) * 100.0
    print(f"[RESULTS] Total Orders Scored: {len(enriched_df):,}")
    print(f"[RESULTS] Predicted Equipment Failure Exceptions: {predicted_exceptions:,} ({pred_rate:.2f}%)")
    print("\nPredicted Health Tier Breakdown:")
    print(enriched_df['equipment_health_tier'].value_counts())

    # 9. Save Watchlist and Excel Reports
    print("\n>>> STEP 9: Exporting Executive Multi-Tab Excel Reports & CSV Watchlist...")
    watchlist_df = enriched_df[
        (enriched_df['predicted_failure_exception'] == 1) |
        (enriched_df['equipment_health_tier'].isin(['Critical Failure Imminent', 'High Breakdown Risk']))
    ].sort_values(by='failure_risk_score', ascending=False)
    
    # Clean missing values for CSV export (blanks for missing)
    clean_watchlist_csv = clean_missing_for_export(watchlist_df)
    clean_watchlist_csv.to_csv("reports/predictive_maintenance_watchlist.csv", index=False, na_rep="")
    print(f"[INFO] Watchlist CSV exported to: reports/predictive_maintenance_watchlist.csv ({len(watchlist_df):,} records)")

    # Export clean and styled Excel files
    clean_excel_path = output_excel
    styled_excel_path = "ML_4_Predictive_Maintenance_Enriched_Output.xlsx"

    # Export once with full openpyxl styling and missing value sanitation
    export_enriched_excel_report(enriched_df, comparison_df, fi_df, clean_excel_path)

    # Copy to companion locations instantaneously
    shutil.copyfile(clean_excel_path, styled_excel_path)
    os.makedirs("reports", exist_ok=True)
    shutil.copyfile(clean_excel_path, os.path.join("reports", output_excel))
    shutil.copyfile(clean_excel_path, os.path.join("reports", styled_excel_path))

    print("\n" + "=" * 100)
    print("  PREDICTIVE MAINTENANCE PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"  Primary Output Excel: {os.path.abspath(clean_excel_path)}")
    print(f"  Executive Styled Excel: {os.path.abspath(styled_excel_path)}")
    print(f"  Champion Model Artifact: models/champion_predictive_maintenance_model.joblib")
    print(f"  Evaluation Figures: reports/figures/")
    print("=" * 100)

    return enriched_df, comparison_df, champion_classifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use Case 4: Predictive Maintenance Classification ML Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Number of PM records to extract (default: None for ALL data from HANA_P2 & HANA_P3)")
    parser.add_argument("--contamination", type=float, default=0.25, help="Estimated failure contamination rate")
    parser.add_argument("--output", type=str, default="ML_4_Predictive_Maintenance_Enriched_Output_Clean.xlsx", help="Output Excel filename")
    args = parser.parse_args()

    run_pipeline(limit=args.limit, contamination=args.contamination, output_excel=args.output)
