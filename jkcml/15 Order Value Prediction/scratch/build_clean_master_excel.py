import os
import sys
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
from src.inference import export_enriched_excel_report

print("Loading evaluation metrics and feature importance...")
eval_df = pd.read_csv("reports/model_comparison_metrics.csv")
feat_imp = pd.read_csv("reports/feature_importance.csv")

print("Reading scored dataset...")
csv_path = "ML_15_Order_Value_Prediction_Enriched_Output_Full.csv"
if os.path.exists(csv_path):
    print(f"Reading from {csv_path}...")
    df = pd.read_csv(csv_path, nrows=50000)
else:
    print("Reading from ML_15_Order_Value_Prediction_Enriched_Output_Clean.xlsx...")
    df = pd.read_excel("ML_15_Order_Value_Prediction_Enriched_Output_Clean.xlsx", sheet_name="ML_15_Enriched_Output(Predict)", skiprows=6)

print(f"Loaded {len(df):,} records for Excel workbook generation.")

# Generate Master File 1
export_enriched_excel_report(
    df=df,
    output_path="ML_15_Order_Value_Prediction_Enriched_Output_Clean.xlsx",
    insight_id="ML_15",
    use_case="Order Value Prediction",
    evaluation_df=eval_df,
    feature_importance_df=feat_imp,
    max_excel_rows=50000
)

# Generate Master File 2 (Mirror copy)
export_enriched_excel_report(
    df=df,
    output_path="ML_15_Order_Value_Prediction_Enriched_Output.xlsx",
    insight_id="ML_15",
    use_case="Order Value Prediction",
    evaluation_df=eval_df,
    feature_importance_df=feat_imp,
    max_excel_rows=50000
)

print("Both Master Excel files successfully updated with all sheets in ONE file!")
