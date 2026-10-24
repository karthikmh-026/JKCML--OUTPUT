"""
Evaluation and Comparison Module - Use Case 14: PR to PO Cycle Time Prediction
Category: Regression
Objective: Estimate procurement processing time.

Calculates regression evaluation metrics (R² Score, MAE, RMSE, MedAE, Accuracy %, MAPE),
ranks the 5 Regression ML models to select the #1 Champion Model,
and computes 32-feature importance and explainability.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.inspection import permutation_importance


def evaluate_regression_models(predictions_dict: dict, y_true: np.ndarray) -> pd.DataFrame:
    """
    Evaluates and ranks all 5 regression models on the test set.
    """
    metrics = []

    for name, preds in predictions_dict.items():
        r2 = r2_score(y_true, preds)
        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        medae = median_absolute_error(y_true, preds)

        # Accuracy within ±7 days tolerance (standard procurement week window)
        acc_7d = np.mean(np.abs(y_true - preds) <= 7.0) * 100.0

        # Mean Absolute Percentage Error (MAPE) handling zero actual cycle days
        denom = np.maximum(y_true, 1.0)
        mape = np.mean(np.abs(y_true - preds) / denom) * 100.0

        metrics.append({
            'Model': name,
            'R² Score': round(float(r2), 4),
            'MAE (Days)': round(float(mae), 2),
            'RMSE (Days)': round(float(rmse), 2),
            'MedAE (Days)': round(float(medae), 2),
            'Accuracy (±7 Days %)': round(float(acc_7d), 2),
            'MAPE (%)': round(float(mape), 2)
        })

    comparison_df = (
        pd.DataFrame(metrics)
        .sort_values(by=['R² Score', 'MAE (Days)'], ascending=[False, True])
        .reset_index(drop=True)
    )
    comparison_df.insert(0, 'Rank', [f"#{i+1}" for i in range(len(comparison_df))])
    return comparison_df


def plot_model_comparison(comparison_df: pd.DataFrame, output_dir: str = None):
    """
    Generates and saves a comparative performance bar chart for the 5 regression models.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Subplot 1: R² Score and Accuracy (%)
    df_plot1 = comparison_df[['Model', 'R² Score']].copy()
    palette = sns.color_palette("Blues_r", len(comparison_df))
    ax1 = sns.barplot(data=df_plot1, x='Model', y='R² Score', hue='Model', palette=palette, legend=False, ax=axes[0])
    axes[0].set_title("5 Regression ML Models R² Score Comparison", fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("R² Score (Coefficient of Determination)", fontsize=11)
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    axes[0].tick_params(axis='x', rotation=15)

    for p in ax1.patches:
        h = p.get_height()
        if h > 0.05:
            ax1.annotate(f"{h:.4f}", (p.get_x() + p.get_width() / 2., h),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 2), textcoords='offset points')

    # Subplot 2: MAE (Days) and MedAE (Days)
    melted_err = pd.melt(
        comparison_df,
        id_vars=['Model'],
        value_vars=['MAE (Days)', 'MedAE (Days)'],
        var_name='Error Metric',
        value_name='Days'
    )
    ax2 = sns.barplot(data=melted_err, x='Model', y='Days', hue='Error Metric', palette="Set2", ax=axes[1])
    axes[1].set_title("Forecast Error Comparison: MAE vs Median Absolute Error (Days)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Error in Days (Lower is Better)", fontsize=11)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].legend(loc='upper right')

    for p in ax2.patches:
        h = p.get_height()
        if h > 1.0:
            ax2.annotate(f"{h:.1f}d", (p.get_x() + p.get_width() / 2., h),
                         ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "model_comparison_metrics.png"), dpi=300)
    plt.close()


def plot_actual_vs_predicted(champion_preds: np.ndarray, y_true: np.ndarray, champion_name: str, output_dir: str = None):
    """
    Plots Actual vs Predicted Cycle Time scatter fit and residual error distribution.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter plot
    sample_size = min(2000, len(y_true))
    idx = np.random.choice(len(y_true), sample_size, replace=False)
    y_s = y_true[idx]
    p_s = champion_preds[idx]

    axes[0].scatter(y_s, p_s, alpha=0.4, color='#1F4E78', edgecolors='none', s=20)
    max_val = max(np.percentile(y_s, 99), np.percentile(p_s, 99), 60)
    axes[0].plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Fit (y = x)')
    axes[0].set_xlim(0, max_val)
    axes[0].set_ylim(0, max_val)
    axes[0].set_title(f"Actual vs Predicted Cycle Time (Days) - Champion: {champion_name}", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Actual Cycle Time (Days)", fontsize=11)
    axes[0].set_ylabel("Predicted Cycle Time (Days)", fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper left')

    # Residuals plot
    residuals = champion_preds - y_true
    sns.histplot(residuals, kde=True, color='#2E75B6', bins=40, ax=axes[1])
    axes[1].axvline(0, color='red', linestyle='--', lw=2, label='Zero Error')
    axes[1].set_xlim(-40, 40)
    axes[1].set_title(f"Forecast Residuals Distribution (Predicted - Actual Days)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Residual Error (Days)", fontsize=11)
    axes[1].set_ylabel("Count of Requisitions", fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "actual_vs_predicted_regression.png"), dpi=300)
    plt.close()


def compute_feature_importance(model, feature_names: list, X_scaled: np.ndarray = None, y_true: np.ndarray = None) -> pd.DataFrame:
    """
    Computes explainability and feature importance scores for all 32 engineered features.
    Handles tree-based importance, linear coefficients, or permutation importance for HistGradientBoosting.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    elif X_scaled is not None and y_true is not None:
        try:
            # Use sample of up to 3000 rows for high-speed permutation importance
            n_samples = min(3000, len(X_scaled))
            perm_idx = np.random.RandomState(42).choice(len(X_scaled), n_samples, replace=False)
            perm_result = permutation_importance(
                model, X_scaled[perm_idx], y_true[perm_idx],
                n_repeats=3, random_state=42, n_jobs=-1
            )
            importances = np.maximum(0.0, perm_result.importances_mean)
        except Exception as e:
            print(f"[WARNING] Permutation importance fallback: {e}")
            importances = np.ones(len(feature_names))
    else:
        importances = np.ones(len(feature_names))

    total = np.sum(importances)
    norm_importances = (importances / total) if total > 0 else (np.ones(len(feature_names)) / len(feature_names))

    business_rationale = {
        'plant_avg_cycle_time': 'Historical operational turnaround velocity specific to individual manufacturing plants.',
        'matkl_avg_cycle_time': 'Historical average procurement cycle time benchmark for the material commodity group.',
        'ekgrp_avg_cycle_time': 'Historical average conversion turnaround speed benchmark for the assigned purchasing group.',
        'planned_delivery_days': 'SAP master planned lead time (PLIFZ) setting baseline supplier contract turnaround expectations.',
        'pr_to_delivery_horizon': 'Tight requisition-to-required delivery windows create procurement expediting pressure.',
        'delivery_urgency_ratio': 'Ratio of available procurement horizon against planned delivery lead time.',
        'pr_item_value': 'High monetary exposure attracts multi-tier approval bottlenecks and stringent compliance audit.',
        'log_pr_item_value': 'Normalizes extreme capital expenditure spend variance across plant operational units.',
        'unit_price': 'Premium high-value spares require detailed commercial price negotiation before PO commitment.',
        'log_unit_price': 'Dampens high-outlier unit prices in emergency maintenance and machinery components.',
        'pr_quantity': 'Bulky bulk-material or specialized volume requirements incur extended vendor lead-time verification.',
        'log_pr_quantity': 'Logarithmic scaling of requisition quantity ensuring stable gradient response across material groups.',
        'release_state_code': 'Indicates PR approval status (2=Fully Released, X=Blocked/Pending, other=In Review).',
        'has_release_block': 'Active release block directly stalls PR conversion into actionable Purchase Order.',
        'processing_status_code': 'PR processing stage (N=New/Unassigned, A=Quotation Requested, B=PO Generated).',
        'has_assigned_vendor': 'Pre-assigned fixed vendor accelerates conversion; unassigned PRs require RFQ vendor sourcing.',
        'deletion_flag': 'Marked for deletion (LOEKZ=X) represents cancelled or rejected procurement demands.',
        'release_strategy_code': 'Multi-level managerial release hierarchies introduce administrative queueing lag.',
        'release_group_code': 'Hierarchical procurement grouping defining departmental sign-off authorization authority.',
        'account_assignment_code': 'Cost center, asset, or project WBS assignment controls budget verification cycle times.',
        'item_category_code': 'Standard stock vs consignment, subcontracting, or service requisitions.',
        'high_value_pr_flag': 'Identifies top 10% high-exposure requisitions subject to executive committee sign-off.',
        'purchasing_group_freq': 'High-frequency buyer groups possess specialized supplier networks and turnaround velocity.',
        'plant_freq': 'Plant operational throughput reflects local receiving capacity and procurement coordination.',
        'material_group_freq': 'Specialized raw material vs MRO commodity groups exhibit distinct lead times.',
        'material_type_freq': 'Finished goods vs raw materials vs operating supplies dictate sourcing workflow.',
        'vendor_freq': 'Established contracted vendors process purchase orders significantly faster than ad-hoc suppliers.',
        'company_code_freq': 'Operating legal entity fiscal governance and procurement delegation rules.',
        'pr_creation_month': 'Seasonal procurement surges during fiscal year-end or monsoon shutdowns.',
        'pr_creation_dayofweek': 'Requisitions created on late weekdays often queue over weekend processing pauses.',
        'is_weekend_pr': 'Weekend requisition creations incur delayed handoff to procurement buyer desks.',
        'is_quarter_end_pr': 'Quarter-end budget consumption drives PR spikes and approval queue bottlenecks.'
    }

    df_imp = pd.DataFrame({
        'Feature Name': feature_names,
        'Importance Score': [round(float(v), 6) for v in norm_importances],
        'Business Impact & Procurement Rationale': [business_rationale.get(f, 'Key operational procurement feature.') for f in feature_names]
    }).sort_values(by='Importance Score', ascending=False).reset_index(drop=True)

    df_imp.insert(0, 'Rank', [f"#{i+1}" for i in range(len(df_imp))])
    return df_imp
