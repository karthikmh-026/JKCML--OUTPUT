"""
Evaluation and Comparison Module - Use Case 4: Predictive Maintenance Classification
(Predict Equipment Failures Before Breakdown)
Calculates performance metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Specificity),
ranks the 5 unsupervised models to select the #1 Champion Model,
computes 32-feature importance, and produces visual evaluation charts.
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


MODEL_DISPLAY_NAMES = {
    'isolation_forest': 'Isolation Forest',
    'lof': 'Local Outlier Factor (LOF)',
    'one_class_svm': 'One-Class SVM',
    'gmm': 'Gaussian Mixture Model (GMM)',
    'kmeans': 'K-Means Clustering'
}


def evaluate_models(predictions_dict: dict, y_true: np.ndarray) -> pd.DataFrame:
    """
    Evaluates and ranks all 5 unsupervised models on the test set against post-hoc
    audit benchmark equipment breakdown/failure exception labels.
    """
    metrics = []

    for key, res in predictions_dict.items():
        name = MODEL_DISPLAY_NAMES.get(key, key)
        preds = res['preds']
        scores = res['scores']

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
            'Model_Key': key,
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(auc, 4),
            'PR-AUC': round(pr_auc, 4),
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
    plt.title("5 Unsupervised ML Models Performance & Accuracy Comparison - Predictive Maintenance", fontsize=13, fontweight='bold')
    plt.ylim(0, 1.12)
    plt.xlabel("Unsupervised Predictive Maintenance Model", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')

    for p in ax.patches:
        height = p.get_height()
        if height > 0.04:
            ax.annotate(f"{height:.2f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        fontsize=8, rotation=0, xytext=(0, 2),
                        textcoords='offset points')

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "model_comparison_metrics.png"), dpi=300)
    plt.close()


def plot_roc_pr_curves(predictions_dict: dict, y_true: np.ndarray, output_dir: str = None):
    """Plots comparative ROC and Precision-Recall curves for all 5 models."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for (key, res), color in zip(predictions_dict.items(), colors):
        name = MODEL_DISPLAY_NAMES.get(key, key)
        scores = res['scores']

        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_val = roc_auc_score(y_true, scores)
        axes[0].plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC = {auc_val:.3f})")

        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        axes[1].plot(recall, precision, color=color, lw=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")

    axes[0].plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    axes[0].set_title("ROC Curves - 5 Unsupervised Models", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("False Positive Rate", fontsize=10)
    axes[0].set_ylabel("True Positive Rate", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='lower right', fontsize=9)

    baseline = np.mean(y_true)
    axes[1].axhline(y=baseline, color='k', linestyle='--', lw=1.5, alpha=0.7, label=f"Baseline ({baseline:.2f})")
    axes[1].set_title("Precision-Recall Curves - 5 Unsupervised Models", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Recall", fontsize=10)
    axes[1].set_ylabel("Precision", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='lower left', fontsize=9)

    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "roc_pr_curves.png"), dpi=300)
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, champion_name: str, output_dir: str = None):
    """Plots annotated confusion matrix for the #1 Champion model."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=False,
        xticklabels=['Normal / Healthy', 'Failure Exception'],
        yticklabels=['Normal / Healthy', 'Failure Exception'],
        annot_kws={'size': 13, 'fontweight': 'bold'}
    )
    plt.title(f"Confusion Matrix - #1 Champion: {champion_name}", fontsize=11, fontweight='bold')
    plt.xlabel("Predicted Status", fontsize=10)
    plt.ylabel("Actual Benchmark Status", fontsize=10)
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()


def compute_feature_importance(champion_segmenter, X_test_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Computes 32-feature importance for the Champion model by calculating the correlation
    and variance sensitivity between each operational feature and the predicted failure risk score.
    """
    X_scaled = champion_segmenter.scaler.transform(X_test_df[feature_cols].fillna(0.0))
    scores = champion_segmenter.predict_risk_scores(X_scaled)

    correlations = []
    for col in feature_cols:
        feat_vals = X_test_df[col].values
        if np.std(feat_vals) > 1e-6 and np.std(scores) > 1e-6:
            r = np.corrcoef(feat_vals, scores)[0, 1]
            if np.isnan(r):
                r = 0.0
        else:
            r = 0.0
        correlations.append(abs(r))

    total = sum(correlations) if sum(correlations) > 0 else 1.0
    importances = [c / total for c in correlations]

    fi_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances,
        'Absolute_Correlation': correlations
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    fi_df['Rank'] = [f"#{i+1}" for i in range(len(fi_df))]
    return fi_df
