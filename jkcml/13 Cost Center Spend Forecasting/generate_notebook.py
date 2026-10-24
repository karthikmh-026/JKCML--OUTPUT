"""
Generates a 100% CODE-ONLY Jupyter Notebook for Use Case 13: Cost Center Spend Forecasting (Time Series).
ZERO Markdown cells. Every single cell is an executable Python code cell across the 21 steps.
"""
import os
import nbformat as nbf


def create_code_only_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # 01. Business Problem
    cells.append(nbf.v4.new_code_cell("""# 01. Business Problem
USE_CASE = "13 - Cost Center Spend Forecasting (Time Series: Forecast Departmental Spending)"
ML_PARADIGM = "Time-Series Machine Learning (2025 Historical Data -> 2026 Expenditure Forecast)"
TARGET = "Forecast Actual Departmental Spending (actual_spend_inr)"

print(f"Business Problem: {USE_CASE}")
print(f"ML Paradigm:      {ML_PARADIGM}")
print(f"Target Variable:  {TARGET}")
"""))

    # 02. Import Libraries
    cells.append(nbf.v4.new_code_cell("""# 02. Import Libraries
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import joblib

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 5)
"""))

    # 03. Connect to Server
    cells.append(nbf.v4.new_code_cell("""# 03. Connect to Server
DB_HOST = "192.168.1.44"
DB_PORT = 5432
DB_NAME = "JKC_Input"
DB_SCHEMA = "HANA_P3"
DB_USER = "postgres"
DB_PASS = "postgres"

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

with engine.connect() as conn:
    print(f"Connected to PostgreSQL: {DB_HOST}:{DB_PORT}/{DB_NAME} (Schema: {DB_SCHEMA})")
"""))

    # 04. Extract Data
    cells.append(nbf.v4.new_code_cell("""# 04. Extract 2025 Historical & 2026 Forecast Spend Data from PostgreSQL Server (HANA_P3)
from src.db_connector import extract_cost_center_spend_data

df_raw = extract_cost_center_spend_data()
print(f"Extracted records: {len(df_raw):,} rows, {df_raw.shape[1]} columns")
"""))

    # 05. Load Data
    cells.append(nbf.v4.new_code_cell("""# 05. Load Data Preview
print(f"Dataset Shape: {df_raw.shape}")
df_raw[['doc_number', 'cost_center', 'cost_center_name', 'gl_account', 'gl_description', 'plant', 'posting_date', 'spend_amount_raw']].head()
"""))

    # 06. Data Understanding / EDA
    cells.append(nbf.v4.new_code_cell("""# 06. Data Understanding / EDA
df_raw[['cost_center', 'cost_center_name', 'gl_account', 'plant', 'debit_credit_ind']].describe().T
"""))

    cells.append(nbf.v4.new_code_cell("""# 06. Data Understanding / EDA (Time-Series Chronological Monthly Spend Distribution)
plt.figure(figsize=(13, 4))
plt.subplot(1, 2, 1)
df_raw['cost_center_name'].value_counts().head(10).plot(kind='barh', color='steelblue')
plt.title('Top 10 Active SAP Cost Centers by Postings')

plt.subplot(1, 2, 2)
pd.to_datetime(df_raw['posting_date']).dt.strftime('%Y-%m').value_counts().sort_index().plot(kind='bar', color='darkorange')
plt.title('Chronological Monthly Postings (2025 History -> 2026 Forecast Horizon)')
plt.tight_layout()
plt.show()
"""))

    # 07. Data Cleaning & 08. Feature Engineering
    cells.append(nbf.v4.new_code_cell("""# 07 & 08. Time-Series Lag/Rolling Feature Engineering & Departmental Spend Categorization
from src.preprocessing import clean_and_engineer_features, get_feature_matrix

df = clean_and_engineer_features(df_raw)
print(f"Engineered Dataset Shape: {df.shape}")
df[['doc_number', 'cost_center', 'cost_center_name', 'gl_account', 'department_spend_category', 'actual_spend_inr', 'lag_1_month_spend_inr', 'rolling_mean_3m_spend_inr', 'ewma_spend_inr', 'cc_mom_spend_growth_pct']].head(8)
"""))

    # 09. Data Preprocessing
    cells.append(nbf.v4.new_code_cell("""# 09. Data Preprocessing (32 Numerical Time-Series Input Features)
X_raw, y_raw, FEATURE_COLS = get_feature_matrix(df)
print(f"Preprocessed Feature Matrix: {X_raw.shape} (32 Numerical Time-Series Features)")
"""))

    # 10. Define Train/Test Horizons (2025 Training Set -> 2026 Forecast Horizon)
    cells.append(nbf.v4.new_code_cell("""# 10. Chronological Partitioning: 2025 Historical Training Horizon -> 2026 Forecast Horizon
from src.preprocessing import prepare_time_series_train_test

data_dict = prepare_time_series_train_test(df, train_year=2025, test_year=2026)
X_train = data_dict['X_train']
X_test = data_dict['X_test']
X_train_scaled = data_dict['X_train_scaled']
X_test_scaled = data_dict['X_test_scaled']
y_train = data_dict['y_train']
y_test = data_dict['y_test']
scaler = data_dict['scaler']

print(f"2025 Training Horizon:  {X_train.shape[0]:,} records")
print(f"2026 Forecast Horizon:  {X_test.shape[0]:,} records")
"""))

    # 11. Model Suite Setup
    cells.append(nbf.v4.new_code_cell("""# 11. Time-Series Forecasting Model Suite Setup
from src.models import CostCenterSpendForecastingSuite
suite = CostCenterSpendForecastingSuite(random_state=42)
print("Initialized 5 Time-Series ML Forecasting Models.")
"""))

    # 12. Model 1 - Random Forest
    cells.append(nbf.v4.new_code_cell("""# 12. Time-Series Model 1/5: Random Forest Time-Series Forecaster
rf_model = suite.train_random_forest(X_train_scaled, y_train)
preds_rf = suite.models['Random Forest Time-Series Forecaster'].predict(X_test_scaled)
print(f"Random Forest 2026 Mean Forecast: INR {preds_rf.mean():,.2f}")
"""))

    # 13. Model 2 - Gradient Boosting
    cells.append(nbf.v4.new_code_cell("""# 13. Time-Series Model 2/5: Gradient Boosting Time-Series Forecaster
gbr_model = suite.train_gradient_boosting(X_train_scaled, y_train)
preds_gbr = suite.models['Gradient Boosting Time-Series Forecaster'].predict(X_test_scaled)
print(f"Gradient Boosting 2026 Mean Forecast: INR {preds_gbr.mean():,.2f}")
"""))

    # 14. Model 3 - Ridge Regressor
    cells.append(nbf.v4.new_code_cell("""# 14. Time-Series Model 3/5: Ridge Time-Series Forecaster
ridge_model = suite.train_ridge(X_train_scaled, y_train)
preds_ridge = suite.models['Ridge Time-Series Forecaster'].predict(X_test_scaled)
print(f"Ridge Regressor 2026 Mean Forecast: INR {preds_ridge.mean():,.2f}")
"""))

    # 15. Model 4 - Holt-Winters Exponential Smoothing
    cells.append(nbf.v4.new_code_cell("""# 15. Time-Series Model 4/5: Holt-Winters Exponential Smoothing Forecaster
hw_model = suite.train_holt_winters(X_train_scaled, y_train)
preds_hw = suite.models['Holt-Winters Exponential Smoothing'].predict(X_test_scaled)
print(f"Holt-Winters 2026 Mean Forecast: INR {preds_hw.mean():,.2f}")
"""))

    # 16. Model 5 - Bayesian AutoRegressive
    cells.append(nbf.v4.new_code_cell("""# 16. Time-Series Model 5/5: Bayesian AutoRegressive Forecaster
bayes_model = suite.train_bayesian_ridge(X_train_scaled, y_train)
preds_bayes = suite.models['Bayesian AutoRegressive Forecaster'].predict(X_test_scaled)
print(f"Bayesian AutoRegressive 2026 Mean Forecast: INR {preds_bayes.mean():,.2f}")
"""))

    # 17. Model Evaluation & Ranking
    cells.append(nbf.v4.new_code_cell("""# 17. Model Evaluation & Accuracy Ranking across all 5 Time-Series Models (2026 Horizon)
from src.evaluate import evaluate_models

all_preds = suite.predict_all(X_test_scaled)
eval_df = evaluate_models(all_preds, y_test)
print(eval_df.to_string(index=False))
eval_df
"""))

    # 18. Model Comparison Visualizations & Champion Selection
    cells.append(nbf.v4.new_code_cell("""# 18. Model Comparison Visualizations & #1 Champion Selection
plt.figure(figsize=(12, 5))
melted = pd.melt(eval_df, id_vars=['Model'], value_vars=['Accuracy (%)', 'R² Score'])
sns.barplot(data=melted, x='Model', y='value', hue='variable', palette='Blues_r')
plt.title('5 Time-Series ML Models Forecasting Accuracy & R² Comparison (2025 History -> 2026 Horizon)')
plt.ylabel('Score / Percentage')
plt.legend(loc='upper right')
plt.xticks(rotation=15)
plt.show()

champion_model = eval_df.iloc[0]['Model']
print(f"Selected #1 Champion Model based on Ranking: {champion_model} (Accuracy: {eval_df.iloc[0]['Accuracy (%)']:.2f}%, R²: {eval_df.iloc[0]['R² Score']:.4f})")
"""))

    # 19. Feature Importance
    cells.append(nbf.v4.new_code_cell("""# 19. Feature Importance (All 32 Time-Series Spend Features)
from src.evaluate import compute_feature_importance

feat_imp = compute_feature_importance(suite.models[champion_model], FEATURE_COLS, X_train_scaled, y_true=y_train)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp.head(15), x='importance', y='feature', palette='mako')
plt.title('Top 15 Time-Series Feature Importances for Departmental Spend Forecasting')
plt.tight_layout()
plt.show()

print(feat_imp.head(15).to_string(index=False))
"""))

    # 20. Final Departmental Spend Forecasting with Champion Model
    cells.append(nbf.v4.new_code_cell("""# 20. Final Departmental Spend Forecasting & Exception Triage using Selected #1 Champion Model
from src.models import SingleModelCostCenterSpendForecaster
from src.inference import score_with_single_model

forecaster = SingleModelCostCenterSpendForecaster(champion_name=champion_model, pretrained_model=suite.models[champion_model])
df_scored = score_with_single_model(df, forecaster, scaler, FEATURE_COLS)

print("=== DEPARTMENTAL SPEND CATEGORY COUNTS ===")
print(df_scored['department_spend_category'].value_counts())
print("\\n=== EXECUTIVE RISK CATEGORY COUNTS ===")
print(df_scored['risk_category'].value_counts())

audit_cols = ['doc_number', 'cost_center', 'cost_center_name', 'gl_account', 'department_spend_category', 'actual_spend_inr', 'forecasted_spend_inr', 'spend_forecast_variance_inr', 'forecast_accuracy_pct', 'risk_category']
print(df_scored[df_scored['risk_category'] == 'Critical Risk'][audit_cols].head(10).to_string(index=False))
"""))

    # 21. Save Model
    cells.append(nbf.v4.new_code_cell("""# 21. Save Champion Model & Scaler
os.makedirs("models", exist_ok=True)

bundle = {
    'champion_model': suite.models[champion_model],
    'champion_name': champion_model,
    'scaler': scaler,
    'feature_cols': FEATURE_COLS
}

joblib.dump(bundle, "models/cost_center_spend_model_bundle.joblib")
joblib.dump(scaler, "models/robust_scaler.joblib")
print("Saved champion Time-Series Cost Center Spend model bundle and scaler to models/")
"""))

    nb.cells = cells

    paths = [
        os.path.join("notebooks", "13_Cost_Center_Spend_Forecasting_ML.ipynb"),
        "13_Cost_Center_Spend_Forecasting_ML.ipynb"
    ]
    for p in paths:
        os.makedirs(os.path.dirname(p) if os.path.dirname(p) else ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"[SUCCESS] Wrote 100% CODE-ONLY notebook to {p} ({len(cells)} code cells, 0 markdown cells).")


if __name__ == "__main__":
    create_code_only_notebook()
