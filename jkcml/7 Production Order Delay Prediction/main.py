"""
Main Execution Script - Use Case 7: Production Order Delay Prediction & Classification
Runs the end-to-end Unsupervised Machine Learning pipeline on ALL data across HANA_P2 and HANA_P3 schemas (NO LIMIT):
1. Connects to PostgreSQL database (192.168.1.44:5432 / JKC_Input / HANA_P2 & HANA_P3)
   and extracts all production order records from AFKO, AUFK, AFPO, AFRU, MAKT, MARA, MARC.
2. Cleans and engineers 30 domain features capturing schedule compression, lead times, and shop floor confirmation yield.
   Strictly handles missing values (leaves genuine blanks where data is absent).
3. Trains 5 Unsupervised Machine Learning Models strictly on input features (no target labels used).
4. Evaluates Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Silhouette Score to rank all 5 models.
5. Selects the #1 Ranked Champion Model based on the ranking.
6. Uses the selected Champion Model to segment all production orders into 4 Delay Risk Tiers and calculate
   calibrated Probability of Delay P(Delay), binary predictions, and actionable recommendations.
7. Persists champion model artifacts and exports executive 5-sheet Excel reports and diagnostic charts.
"""
import os
import shutil
import argparse
import pandas as pd
from src.db_connector import load_config, extract_production_order_data
from src.preprocessing import clean_and_engineer_features, prepare_train_test_data
from src.models import ProductionOrderDelayClusteringSuite, SingleModelProductionOrderSegmenter
from src.evaluate import (
    evaluate_models,
    plot_model_comparison,
    plot_roc_pr_curves,
    plot_delay_risk_distribution,
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
    output_excel: str = "ML_7_Production_Order_Delay_Prediction_Enriched_Output_Clean.xlsx",
    use_cache: bool = False
):
    print("=" * 96)
    print("  USE CASE 7: PRODUCTION ORDER DELAY PREDICTION & RISK CLASSIFICATION")
    print("  ALL DATA ACROSS HANA_P2 & HANA_P3 (NO LIMIT) -> 5 UNSUPERVISED ML MODELS -> CHAMPION DEPLOYMENT")
    print("=" * 96)

    # 1. Config & Data Extraction directly from PostgreSQL Server
    config = load_config()
    print("\n>>> STEP 1: Connecting to PostgreSQL (192.168.1.44:5432 / JKC_Input / HANA_P2 & HANA_P3)...")
    print("    Extracting ALL combined data from: AFKO, AUFK, AFPO, AFRU, MAKT, MARA, MARC (NO LIMIT)...")
    raw_df = extract_production_order_data(limit=limit, config=config, use_cache=use_cache)

    # 2. Data Cleaning & 30-Feature Engineering with Strict Missing Value Handling
    print("\n>>> STEP 2: Performing Data Cleaning & Engineering 30 Schedule, Yield & Lead Time Features...")
    featured_df = clean_and_engineer_features(raw_df)

    delay_cnt = int(featured_df['is_delayed_exception'].sum())
    delay_rate = (delay_cnt / max(1, len(featured_df))) * 100.0
    contamination = max(0.10, min(0.60, delay_cnt / max(1, len(featured_df))))
    print(f"[SUMMARY] Total Extracted Production Orders: {len(featured_df):,}")
    print(f"[SUMMARY] Historical Ground-Truth Delayed Orders: {delay_cnt:,} ({delay_rate:.2f}%)")
    print(f"[SUMMARY] Historical Ground-Truth On-Time Orders: {len(featured_df) - delay_cnt:,} ({100.0 - delay_rate:.2f}%)")

    # 3. Data Preparation & Robust Scaling (Unsupervised X matrix)
    print("\n>>> STEP 3: Preparing 30-Feature Production Matrix & Applying RobustScaler (Unsupervised)...")
    data_dict = prepare_train_test_data(featured_df, test_size=0.20, random_state=42)
    X_train_scaled = data_dict['X_train_scaled']
    X_test_scaled = data_dict['X_test_scaled']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    feature_cols = data_dict['feature_cols']
    scaler = data_dict['scaler']

    # 4. Train 5 Unsupervised ML Models
    print("\n>>> STEP 4: Training 5 Unsupervised ML Models (Input-Only Training, No Output Labels Used)...")
    suite = ProductionOrderDelayClusteringSuite(n_clusters=n_clusters, contamination=contamination, random_state=42)
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

    # 6. Fit Selected #1 Champion Model & Segment All Production Orders
    print(f"\n>>> STEP 6: Deploying Selected #1 Champion Model ({champion_model_name}) for Production Scoring...")
    champion_segmenter = SingleModelProductionOrderSegmenter(
        model_name=champion_model_name,
        n_clusters=n_clusters,
        contamination=contamination,
        random_state=42
    )
    champion_segmenter.fit(X_train_scaled)

    # 7. Score All Production Orders & Generate Recommendations
    print("\n>>> STEP 7: Scoring All Production Orders & Estimating Probability of Delay P(Delay)...")
    scored_df = score_with_single_model(champion_segmenter, featured_df, scaler, feature_cols)

    # 8. Feature Importance
    print("\n>>> STEP 8: Computing 30-Feature Importance and Delay Explainability...")
    fi_df = compute_feature_importance(X_train_scaled, y_train, feature_cols)
    fi_df.to_csv("reports/feature_importance.csv", index=False)
    plot_feature_importance(fi_df, output_dir="reports/figures", top_n=15)
    print("Top 10 Most Influential Delay Drivers:")
    print(fi_df.head(10)[['Rank', 'Feature', 'Importance', 'Relative_Weight_Pct']].to_string(index=False))

    # 9. Generate Charts & Excel Reports with Blank Missing Values
    print("\n>>> STEP 9: Generating Distribution Visualizations & Multi-Sheet Executive Excel Reports...")
    plot_delay_risk_distribution(scored_df, output_dir="reports/figures")
    export_enriched_excel_report(scored_df, output_path=output_excel, fi_df=fi_df, comp_df=comparison_df)

    # Also create copy as ML_7_Production_Order_Delay_Prediction_Enriched_Output.xlsx
    alt_excel = "ML_7_Production_Order_Delay_Prediction_Enriched_Output.xlsx"
    try:
        shutil.copyfile(output_excel, alt_excel)
        print(f"[REPORT] Copied enriched Excel report to {alt_excel}")
    except PermissionError:
        print(f"[REPORT] Notice: Could not overwrite {alt_excel} because it is currently open in Excel. Updated file is available at {output_excel}.")

    # 10. Persist Model Artifacts
    print("\n>>> STEP 10: Persisting Champion Model Pipeline & Scaler Artifacts to models/...")
    save_single_model_pipeline(champion_segmenter, scaler, feature_cols, output_dir="models")

    print("\n" + "=" * 96)
    print("  PIPELINE EXECUTION COMPLETE - ALL DELIVERABLES GENERATED SUCCESSFULLY (NO LIMIT)!")
    print(f"  * Total Processed Orders: {len(featured_df):,}")
    print(f"  * #1 Champion Model: {champion_model_name}")
    print(f"  * Model Comparison Metrics: reports/model_comparison_metrics.csv")
    print(f"  * Feature Importance: reports/feature_importance.csv")
    print(f"  * Visual Figures: reports/figures/ (model_comparison.png, roc_pr_curves.png, delay_risk_distribution.png)")
    print(f"  * Model Artifacts: models/ (champion_delay_model.joblib, delay_model_bundle.joblib)")
    print(f"  * Executive Excel Reports: {output_excel}")
    print("=" * 96)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Production Order Delay Prediction Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Number of records to extract (None for all data)")
    parser.add_argument("--n_clusters", type=int, default=4, help="Number of clusters/tiers")
    parser.add_argument("--output_excel", type=str, default="ML_7_Production_Order_Delay_Prediction_Enriched_Output_Clean.xlsx", help="Output Excel filename")
    parser.add_argument("--use_cache", action="store_true", help="Use local cached dataset if available")
    args = parser.parse_args()

    run_pipeline(limit=args.limit, n_clusters=args.n_clusters, output_excel=args.output_excel, use_cache=args.use_cache)
