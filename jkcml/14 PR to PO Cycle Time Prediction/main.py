"""
Main Execution Script - Use Case 14: PR to PO Cycle Time Prediction
Category: Regression
Objective: Estimate procurement processing time.

Runs the end-to-end Regression Machine Learning pipeline on ALL data across HANA_P2 and HANA_P3:
1. Connects to PostgreSQL server (192.168.1.44:5432 / JKC_Input) and extracts SAP Purchase Requisitions
   from both HANA_P2 and HANA_P3 (EBAN, MARA, MAKT, LFA1, T001W) without any row limits (529,000+ records).
2. Performs data cleaning and engineers 32 domain-specific features for PR to PO Cycle Time Regression.
3. Prepares regression train/test split on benchmark historical conversions with RobustScaler.
4. Trains 5 Regression ML models (Extra Trees, Random Forest, Gradient Boosting, Bayesian Ridge, Ridge Regression).
5. Evaluates R² Score, MAE (Days), RMSE (Days), MedAE (Days), and Accuracy (±7 Days %) to rank all 5 models
   and selects the #1 Champion Model based on the ranking.
6. Uses the selected #1 Champion Regression Model to score all 529,001 PR records with estimated procurement processing
   times (days), actual cycle times, forecast variances, individual accuracy %, SLA status, and risk tiers.
7. Computes 32-feature importance and exports executive 3-sheet Excel reports and visual evaluation charts.
   Strict missing value handling: if data is present, writes it cleanly; if missing/null/nan, writes blank.
"""
import os
import shutil
import argparse
import pandas as pd
from src.db_connector import load_config, extract_pr_cycle_data
from src.preprocessing import clean_and_engineer_features, prepare_regression_train_test
from src.models import PRCycleRegressionSuite, SingleModelPRCycleRegressor
from src.evaluate import (
    evaluate_regression_models,
    plot_model_comparison,
    plot_actual_vs_predicted,
    compute_feature_importance
)
from src.inference import (
    score_with_single_model,
    export_enriched_excel_report,
    save_single_model_pipeline
)


def run_pipeline(
    limit: int = 0,
    output_excel: str = "ML_14_PR_to_PO_Cycle_Time_Prediction_Enriched_Output_Clean.xlsx"
):
    print("=" * 105)
    print("  USE CASE 14: PR TO PO CYCLE TIME PREDICTION (REGRESSION: ESTIMATE PROCUREMENT PROCESSING TIME)")
    print("=" * 105)

    # 1. Config & Data Extraction directly from PostgreSQL Server across HANA_P2 & HANA_P3
    config = load_config()
    schemas = config.get("database", {}).get("schemas", ["HANA_P2", "HANA_P3"])
    print(f"\n>>> STEP 1: Connecting to PostgreSQL Server (192.168.1.44:5432 / JKC_Input)...")
    limit_str = "UNLIMITED" if limit <= 0 else f"{limit:,}"
    print(f"    Extracting ALL data from schemas: {schemas} (limit={limit_str})")
    print("    Deduplicated dimension joins: EBAN, MARA, MAKT, LFA1, T001W")
    raw_df = extract_pr_cycle_data(limit=limit, config=config)

    # 2. Data Cleaning & 32-Feature Engineering
    print("\n>>> STEP 2: Performing Data Cleaning & Engineering 32 PR Cycle Time Regression Features...")
    featured_df = clean_and_engineer_features(raw_df)

    n_closed = int(featured_df['actual_cycle_time_days'].notna().sum())
    n_open = int(featured_df['actual_cycle_time_days'].isna().sum())
    print(f"[SUMMARY] Total Extracted SAP PR Records:              {len(featured_df):,}")
    print(f"[SUMMARY] Closed Records (Historical Ground Truth):    {n_closed:,} ({n_closed / len(featured_df) * 100:.1f}%)")
    print(f"[SUMMARY] Open Requisitions (Awaiting PO Conversion):  {n_open:,} ({n_open / len(featured_df) * 100:.1f}%)")

    # 3. Data Preparation & Robust Scaling (Regression train/test split)
    print("\n>>> STEP 3: Preparing 32-Feature Regression Matrix & Applying RobustScaler...")
    data_dict = prepare_regression_train_test(featured_df, test_size=0.20, random_state=42)
    X_train_scaled = data_dict['X_train_scaled']
    X_test_scaled = data_dict['X_test_scaled']
    y_train = data_dict['y_train']
    y_test = data_dict['y_test']
    feature_cols = data_dict['feature_cols']
    scaler = data_dict['scaler']

    print(f"  -> X_train shape: {X_train_scaled.shape} (Historical Training Set)")
    print(f"  -> X_test shape:  {X_test_scaled.shape} (Holdout Test Horizon)")

    # 4. Train 5 Regression ML Models
    print("\n>>> STEP 4: Training 5 Regression ML Models to Estimate Procurement Processing Time...")
    suite = PRCycleRegressionSuite(random_state=42)
    suite.train_all(X_train_scaled, y_train)

    # 5. Evaluate Accuracy & Rank the 5 Regression Models on Test Split
    print("\n>>> STEP 5: Evaluating Regression Accuracy Metrics & Ranking Models on Holdout Test Split...")
    test_preds = suite.predict_all(X_test_scaled)
    comparison_df = evaluate_regression_models(test_preds, y_test)

    print("\n" + "=" * 105)
    print("  5 REGRESSION ML MODELS ACCURACY & PERFORMANCE RANKING (TEST SET EVALUATION)")
    print("=" * 105)
    print(comparison_df.to_string(index=False))
    print("=" * 105)

    os.makedirs("reports/figures", exist_ok=True)
    comparison_df.to_csv("reports/model_comparison_metrics.csv", index=False)
    plot_model_comparison(comparison_df, output_dir="reports/figures")

    champion_model_name = comparison_df.iloc[0]['Model']
    champion_r2 = comparison_df.iloc[0]['R² Score']
    champion_mae = comparison_df.iloc[0]['MAE (Days)']
    champion_rmse = comparison_df.iloc[0]['RMSE (Days)']
    champion_acc = comparison_df.iloc[0]['Accuracy (±7 Days %)']

    print(f"\n[CHAMPION SELECTED] Rank #1 Model: {champion_model_name}")
    print(f"  -> R² Score: {champion_r2:.4f} | MAE: {champion_mae:.2f} Days | RMSE: {champion_rmse:.2f} Days | Accuracy (±7 Days): {champion_acc:.2f}%")

    # Plot Actual vs Predicted for Champion Model
    plot_actual_vs_predicted(
        champion_preds=test_preds[champion_model_name],
        y_true=y_test,
        champion_name=champion_model_name,
        output_dir="reports/figures"
    )

    # 6. Fit Selected #1 Champion Model & Score All PR Records
    print(f"\n>>> STEP 6: Estimating Procurement Processing Time for All {len(featured_df):,} SAP PRs with Champion Model ({champion_model_name})...")
    regressor = SingleModelPRCycleRegressor(
        champion_name=champion_model_name,
        pretrained_model=suite.models[champion_model_name]
    )
    scored_df = score_with_single_model(featured_df, regressor, scaler, feature_cols)

    print("\n[OPERATIONAL SLA STATUS DISTRIBUTION]:")
    for sla_name, cnt in scored_df['sla_status'].value_counts().items():
        pct = (cnt / len(scored_df)) * 100.0
        print(f"  - {sla_name:<34}: {cnt:>7,} requisitions ({pct:.2f}%)")

    crit_cnt = int((scored_df['procurement_risk_tier'] == 'Critical Risk').sum())
    high_cnt = int((scored_df['procurement_risk_tier'] == 'High Risk').sum())
    med_cnt = int((scored_df['procurement_risk_tier'] == 'Medium Risk').sum())
    low_cnt = int((scored_df['procurement_risk_tier'] == 'Low Risk').sum())

    print("\n[EXECUTIVE PROCUREMENT CYCLE TIME RISK TRIAGE SUMMARY]:")
    print(f"  - Critical Risk: {crit_cnt:>7,} requisitions (severe cycle bottleneck > 45 days / dormant / release blocks)")
    print(f"  - High Risk:     {high_cnt:>7,} requisitions (sourcing delay 30-45 days / unassigned vendor)")
    print(f"  - Medium Risk:   {med_cnt:>7,} requisitions (moderate turnaround 15-30 days)")
    print(f"  - Low Risk:      {low_cnt:>7,} requisitions (rapid processing < 15 days)")

    # Save top delayed watchlist CSV
    os.makedirs("reports", exist_ok=True)
    scored_df[scored_df['procurement_risk_tier'].isin(['Critical Risk', 'High Risk'])].head(5000).to_csv(
        "reports/pr_cycle_time_delayed_watchlist.csv", index=False
    )

    # 7. Compute 32-Feature Importance & Explainability
    print("\n>>> STEP 7: Computing 32-Feature Importance for PR to PO Cycle Time Regression...")
    feat_imp = compute_feature_importance(regressor.model, feature_cols, X_train_scaled, y_true=y_train)
    print("\n=== TOP 15 FEATURE IMPORTANCE SCORES ===")
    print(feat_imp.head(15).to_string(index=False))
    feat_imp.to_csv("reports/feature_importance.csv", index=False)

    # 8. Export Enriched 3-Sheet Excel Reports
    print("\n>>> STEP 8: Generating Executive 3-Sheet Enriched Excel Report with XlsxWriter...")
    excel_path = export_enriched_excel_report(
        df=scored_df,
        output_path=output_excel,
        insight_id="ML_14",
        use_case="PR to PO Cycle Time Prediction",
        category="Regression",
        objective="Estimate procurement processing time.",
        feature_importance_df=feat_imp
    )

    # Also mirror standard named files to root and reports/
    for extra_path in [
        "ML_14_PR_to_PO_Cycle_Time_Prediction_Enriched_Output.xlsx",
        os.path.join("reports", "ML_14_PR_to_PO_Cycle_Time_Prediction_Enriched_Output.xlsx"),
        os.path.join("reports", os.path.basename(output_excel))
    ]:
        if os.path.abspath(extra_path) != os.path.abspath(excel_path):
            os.makedirs(os.path.dirname(os.path.abspath(extra_path)), exist_ok=True)
            try:
                shutil.copy2(excel_path, extra_path)
                print(f"[INFO] Mirrored Excel deliverable to: {extra_path}")
            except Exception as e:
                print(f"[WARNING] Could not copy to {extra_path}: {e}")

    # 9. Persist Champion Regression Model Pipeline
    print("\n>>> STEP 9: Persisting Selected #1 Champion Regression Model Pipeline...")
    save_single_model_pipeline(
        model=regressor,
        scaler=scaler,
        feature_cols=feature_cols,
        comparison_df=comparison_df,
        output_dir="models"
    )

    print("\n" + "=" * 105)
    print("  END-TO-END REGRESSION PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"  Executive Excel Report: {excel_path} ({len(scored_df):,} records)")
    print(f"  Model Artifact: models/champion_pr_cycle_regressor.joblib")
    print(f"  Visualizations: reports/figures/model_comparison_metrics.png, actual_vs_predicted_regression.png")
    print("=" * 105)
    return scored_df, comparison_df, feat_imp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use Case 14: PR to PO Cycle Time Prediction (Regression)")
    parser.add_argument("--limit", type=int, default=0, help="Number of SAP PR records to extract from PostgreSQL (0 = all data)")
    parser.add_argument("--output_excel", type=str, default="ML_14_PR_to_PO_Cycle_Time_Prediction_Enriched_Output_Clean.xlsx", help="Output Excel filename")
    args = parser.parse_args()

    run_pipeline(
        limit=args.limit,
        output_excel=args.output_excel
    )
