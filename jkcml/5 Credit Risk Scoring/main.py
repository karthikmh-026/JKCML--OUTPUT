"""
Main Execution Script - Use Case 5: Credit Risk Scoring
Runs the end-to-end Unsupervised Machine Learning Credit Risk pipeline:
1. Connects to PostgreSQL database (192.168.1.44:5432 / JKC_Input / HANA_P2)
   and extracts customer financial transactions from KNA1, KNB1, BSID, BSAD, VBRK, VBAK, T077X, T001.
2. Cleans and engineers 32 domain features for credit risk, delinquency, and clearing delays.
3. Trains 5 Unsupervised Machine Learning Models strictly on input features (no target labels used).
4. Evaluates Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Silhouette Score to rank all 5 models.
5. Selects the #1 Ranked Champion Model based on the ranking.
6. Uses the selected Champion Model to segment all customers into 4 Credit Risk Tiers and calculate
   calibrated Probability of Customer Default P(Default), binary predictions, and risk categories.
7. Persists champion model artifacts and exports executive 3-sheet Excel reports and charts.
"""
import os
import argparse
import pandas as pd
from src.db_connector import load_config, extract_credit_risk_data
from src.preprocessing import clean_and_engineer_features, prepare_train_test_data
from src.models import CreditRiskClusteringSuite, SingleModelCreditRiskSegmenter
from src.evaluate import (
    evaluate_models,
    plot_model_comparison,
    plot_roc_pr_curves,
    plot_credit_risk_distribution,
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
    n_clusters: int = 4,
    output_excel: str = "ML_5_Credit_Risk_Scoring_Enriched_Output_Clean.xlsx"
):
    print("=" * 96)
    print("  USE CASE 5: CREDIT RISK SCORING (UNSUPERVISED DEFAULT PROBABILITY ESTIMATION)")
    print("  5 UNSUPERVISED ML MODELS -> RANKING EVALUATION -> #1 CHAMPION MODEL PRODUCTION SCORING")
    print("  ENTERPRISE SAP DATASET: ALL RECORDS FROM HANA_P2 + HANA_P3 (NO ROW LIMITS)")
    print("=" * 96)

    # 1. Config & Data Extraction directly from PostgreSQL Server
    config = load_config()
    print("\n>>> STEP 1: Connecting to PostgreSQL (192.168.1.44:5432 / HANA_P2 + HANA_P3) & Extracting ALL SAP Customer Data...")
    print("    Extracting joined data from: KNA1, KNB1, BSID, BSAD, VBRK, VBAK, T077X, T001")
    raw_df = extract_credit_risk_data(limit=limit, config=config, use_cache=True)

    # 2. Data Cleaning & 32-Feature Engineering
    print("\n>>> STEP 2: Performing Data Cleaning & Engineering 32 Credit Risk & Delinquency Features...")
    featured_df = clean_and_engineer_features(raw_df)

    exception_cnt = int(featured_df['is_credit_risk_exception'].sum())
    exception_rate = (exception_cnt / max(1, len(featured_df))) * 100.0
    contamination = max(0.05, min(0.40, exception_cnt / max(1, len(featured_df))))
    print(f"[SUMMARY] Total Extracted SAP Customer Records: {len(featured_df):,}")
    print(f"[SUMMARY] Benchmark High-Risk Delinquent / Default Accounts: {exception_cnt:,} ({exception_rate:.2f}%)")

    # 3. Data Preparation & Robust Scaling (Unsupervised X matrix)
    print("\n>>> STEP 3: Preparing 32-Feature Credit Risk Matrix & Applying RobustScaler (Unsupervised)...")
    data_dict = prepare_train_test_data(featured_df, test_size=0.20, random_state=42)
    X_train_scaled = data_dict['X_train_scaled']
    X_test_scaled = data_dict['X_test_scaled']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    feature_cols = data_dict['feature_cols']
    scaler = data_dict['scaler']

    # 4. Train 5 Unsupervised ML Models
    print("\n>>> STEP 4: Training 5 Unsupervised ML Models (Input-Only Training, No Output Labels Used)...")
    suite = CreditRiskClusteringSuite(n_clusters=n_clusters, contamination=contamination, random_state=42)
    suite.train_all(X_train_scaled)

    # 5. Evaluate & Rank 5 Models
    print("\n>>> STEP 5: Evaluating Accuracy & Ranking All 5 Unsupervised Models on Held-out Test Split...")
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
    champion_rec = comparison_df.iloc[0]['Recall'] * 100.0
    champion_f1 = comparison_df.iloc[0]['F1-Score']
    champion_auc = comparison_df.iloc[0]['ROC-AUC']
    champion_sil = comparison_df.iloc[0]['Silhouette']
    print(f"\n[CHAMPION SELECTED] Rank #1 Model: {champion_model_name}")
    print(f"  -> Accuracy: {champion_acc:.2f}% | Precision: {champion_prec:.2f}% | Recall: {champion_rec:.2f}% | F1-Score: {champion_f1:.4f} | ROC-AUC: {champion_auc:.4f} | Silhouette: {champion_sil:.4f}")

    # 6. Fit Selected #1 Champion Model & Segment All Customers
    print(f"\n>>> STEP 6: Deploying Selected #1 Champion Model ({champion_model_name}) for Production Scoring...")
    champion_segmenter = SingleModelCreditRiskSegmenter(
        model_name=champion_model_name,
        n_clusters=n_clusters,
        contamination=contamination,
        random_state=42
    )
    champion_segmenter.fit(X_train_scaled)

    # 7. Score All Customers & Generate Explanations
    print("\n>>> STEP 7: Scoring All Customer Records & Estimating Probability of Default P(Default)...")
    scored_df = score_with_single_model(champion_segmenter, featured_df, scaler, feature_cols)

    # 8. Feature Importance
    print("\n>>> STEP 8: Computing 32-Feature Importance and Credit Risk Explainability...")
    fi_df = compute_feature_importance(X_train_scaled, y_train, feature_cols)
    fi_df.to_csv("reports/feature_importance.csv", index=False)
    plot_feature_importance(fi_df, output_dir="reports/figures", top_n=15)
    print("Top 10 Most Influential Credit Risk Drivers:")
    print(fi_df.head(10)[['Rank', 'Feature', 'Importance', 'Relative_Weight_Pct']].to_string(index=False))

    # 9. Generate Charts & Excel Reports
    print("\n>>> STEP 9: Generating Distribution Visualizations & Multi-Sheet Executive Excel Reports...")
    plot_credit_risk_distribution(scored_df, output_dir="reports/figures")
    export_enriched_excel_report(scored_df, output_path=output_excel, fi_df=fi_df, comp_df=comparison_df)

    # 10. Persist Model Artifacts
    print("\n>>> STEP 10: Persisting Champion Model Pipeline & Scaler Artifacts to models/...")
    save_single_model_pipeline(champion_segmenter, scaler, feature_cols, output_dir="models")

    print("\n" + "=" * 96)
    print("  PIPELINE EXECUTION COMPLETE - ALL DELIVERABLES GENERATED SUCCESSFULLY!")
    print(f"  * #1 Champion Model: {champion_model_name}")
    print(f"  * Model Comparison Metrics: reports/model_comparison_metrics.csv")
    print(f"  * Feature Importance: reports/feature_importance.csv")
    print(f"  * Visual Figures: reports/figures/ (model_comparison.png, roc_pr_curves.png, cluster_distribution.png)")
    print(f"  * Model Artifacts: models/ (champion_credit_risk_model.joblib, credit_risk_model_bundle.joblib)")
    print(f"  * Executive Excel Reports: {output_excel}")
    print("=" * 96)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use Case 5: Credit Risk Scoring (Unsupervised Default Probability Estimation)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of customer records to extract (default: None, use all data)")
    parser.add_argument("--clusters", type=int, default=4, help="Number of credit risk tier clusters")
    parser.add_argument("--output", type=str, default="ML_5_Credit_Risk_Scoring_Enriched_Output_Clean.xlsx", help="Output Excel filename")
    args = parser.parse_args()

    run_pipeline(limit=args.limit, n_clusters=args.clusters, output_excel=args.output)
