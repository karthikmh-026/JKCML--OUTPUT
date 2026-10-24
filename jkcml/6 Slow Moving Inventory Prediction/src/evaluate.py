"""
Evaluation and Comparison Module - Use Case 6: Slow Moving Inventory Prediction
Calculates performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC,
Silhouette Score, and Specificity), ranks all 5 Unsupervised Clustering ML models
to select the #1 Ranked Champion Model, and generates visual charts and feature importance.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier


def evaluate_models(predictions_dict: dict, y_true: np.ndarray) -> pd.DataFrame:
    """
    Evaluates and ranks all 5 Unsupervised Clustering models on held-out test data
    against post-hoc SAP inventory obsolescence standards (MC46/MC50) plus Silhouette Score.
    """
    metrics = []

    for name, res in predictions_dict.items():
        preds = res['preds']
        scores = res['scores']
        sil = res.get('silhouette', 0.0)

        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        try:
            auc = roc_auc_score(y_true, scores)
        except Exception:
            auc = 0.5

        try:
            pr_auc = average_precision_score(y_true, scores)
        except Exception:
            pr_auc = 0.0

        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(auc, 4),
            'PR-AUC': round(pr_auc, 4),
            'Silhouette': round(sil, 4),
            'Specificity': round(specificity, 4),
            'True Positives': int(tp),
            'False Positives': int(fp),
            'False Negatives': int(fn),
            'True Negatives': int(tn)
        })

    comparison_df = (
        pd.DataFrame(metrics)
        .sort_values(by=['F1-Score', 'ROC-AUC', 'Accuracy'], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    comparison_df.insert(0, 'Rank', [f"#{i+1}" for i in range(len(comparison_df))])
    return comparison_df


def plot_model_comparison(comparison_df: pd.DataFrame, output_dir: str = None):
    """Generates and saves a comparative performance bar chart for the 5 unsupervised models."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(13, 6))
    melted = pd.melt(
        comparison_df,
        id_vars=['Model'],
        value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        var_name='Metric',
        value_name='Score'
    )

    palette = sns.color_palette("Set2", 5)
    ax = sns.barplot(data=melted, x='Model', y='Score', hue='Metric', palette=palette)
    plt.title("5 Unsupervised ML Models Performance & Accuracy Comparison - Slow Moving Inventory Prediction", fontsize=13, fontweight='bold')
    plt.ylim(0, 1.15)
    plt.xlabel("Unsupervised Model", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0.05:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 2),
                        textcoords='offset points')

    plt.tight_layout()
    if output_dir:
        out_path = os.path.join(output_dir, "model_comparison.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[REPORT] Saved model comparison chart to {out_path}")


def plot_roc_pr_curves(predictions_dict: dict, y_true: np.ndarray, output_dir: str = None):
    """Plots and saves ROC and Precision-Recall curves for all 5 models."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ROC Curves
    for name, res in predictions_dict.items():
        try:
            fpr, tpr, _ = roc_curve(y_true, res['scores'])
            auc_val = roc_auc_score(y_true, res['scores'])
            axes[0].plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.3f})")
        except Exception:
            pass

    axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    axes[0].set_title("ROC Curves - 5 Unsupervised Models", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='lower right', fontsize=9)

    # PR Curves
    for name, res in predictions_dict.items():
        try:
            prec, rec, _ = precision_recall_curve(y_true, res['scores'])
            pr_auc = average_precision_score(y_true, res['scores'])
            axes[1].plot(rec, prec, lw=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")
        except Exception:
            pass

    axes[1].set_title("Precision-Recall Curves - 5 Unsupervised Models", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='lower left', fontsize=9)

    plt.tight_layout()
    if output_dir:
        out_path = os.path.join(output_dir, "roc_pr_curves.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[REPORT] Saved ROC & PR curves to {out_path}")


def plot_inventory_distribution(df_scored: pd.DataFrame, output_dir: str = None):
    """Plots distribution of materials across the 4 Inventory Velocity Tiers and Risk Bands."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Inventory Tier Distribution
    tier_order = [
        "Fast Moving / High Velocity Stock",
        "Normal / Steady Moving Stock",
        "Slow Moving / Stagnant Stock",
        "Dormant / Obsolete Stock"
    ]
    tier_counts = df_scored['inventory_tier_name'].value_counts()
    ordered_counts = [tier_counts.get(t, 0) for t in tier_order]
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']

    bars = axes[0].bar(range(len(tier_order)), ordered_counts, color=colors, alpha=0.85)
    axes[0].set_xticks(range(len(tier_order)))
    axes[0].set_xticklabels(['Fast Moving (Tier 0)', 'Steady Moving (Tier 1)', 'Slow Moving (Tier 2)', 'Dormant/Obsolete (Tier 3)'], fontsize=9)
    axes[0].set_title("Inventory Velocity Segmentation (Champion Model)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Number of Materials")
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        axes[0].annotate(f"{h:,}\n({h/len(df_scored)*100:.1f}%)",
                         (bar.get_x() + bar.get_width() / 2., h),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    # 2. Risk Band vs Tied-Up Stock Value
    risk_summary = df_scored.groupby('inventory_risk_category')['current_stock_val'].sum() / 1e7
    risk_cats = ["Low Obsolescence Risk", "Moderate Slow Risk", "High Slow Risk", "Critical Obsolete Risk"]
    risk_vals = [risk_summary.get(c, 0.0) for c in risk_cats]
    risk_colors = ['#2ca02c', '#ffbb78', '#ff7f0e', '#d62728']

    bars2 = axes[1].bar(range(len(risk_cats)), risk_vals, color=risk_colors, alpha=0.85)
    axes[1].set_xticks(range(len(risk_cats)))
    axes[1].set_xticklabels(['Low Risk', 'Moderate Risk', 'High Risk', 'Critical Obsolete'], fontsize=9)
    axes[1].set_title("Tied-Up Working Capital at Risk (Crores INR)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Total Inventory Value (Cr INR)")
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars2:
        h = bar.get_height()
        axes[1].annotate(f"INR {h:,.1f} Cr",
                         (bar.get_x() + bar.get_width() / 2., h),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    plt.tight_layout()
    if output_dir:
        out_path = os.path.join(output_dir, "cluster_distribution.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[REPORT] Saved inventory distribution chart to {out_path}")


def compute_feature_importance(X_train_scaled: np.ndarray, y_train: np.ndarray, feature_cols: list) -> pd.DataFrame:
    """Computes surrogate tree feature importances for explainability."""
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)

    importances = rf.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': np.round(importances, 4),
        'Relative_Weight_Pct': np.round(importances * 100.0, 2)
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    fi_df.insert(0, 'Rank', [f"#{i+1}" for i in range(len(fi_df))])
    return fi_df


def plot_feature_importance(fi_df: pd.DataFrame, output_dir: str = None, top_n: int = 15):
    """Plots top N most influential inventory features."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    top_df = fi_df.head(top_n).sort_values(by='Importance', ascending=True)
    plt.figure(figsize=(11, 7))
    bars = plt.barh(top_df['Feature'], top_df['Importance'], color='#1f77b4', alpha=0.85)
    plt.title(f"Top {top_n} Most Influential Drivers for Slow Moving & Obsolete Inventory", fontsize=12, fontweight='bold')
    plt.xlabel("Surrogate Tree Gini Importance")
    plt.grid(axis='x', linestyle='--', alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        plt.annotate(f"{w:.3f}",
                     (w, bar.get_y() + bar.get_height() / 2.),
                     ha='left', va='center', fontsize=9, xytext=(4, 0), textcoords='offset points')

    plt.tight_layout()
    if output_dir:
        out_path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[REPORT] Saved feature importance chart to {out_path}")
