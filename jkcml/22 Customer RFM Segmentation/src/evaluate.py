"""
Evaluation and Comparison Module - Use Case 22: Customer RFM Segmentation (Clustering)
Calculates evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Specificity,
and Silhouette Score), ranks the 5 Unsupervised Clustering models to select the #1 Champion Model,
and computes 32-feature importance and explainability.
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


def evaluate_models(predictions_dict: dict, y_true: np.ndarray) -> pd.DataFrame:
    """
    Evaluates and ranks all 5 Unsupervised Clustering models on the test set against post-hoc
    benchmark RFM exception & segmentation labels, plus unsupervised Silhouette Score.
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
    """
    Generates and saves a comparative performance bar chart for the 5 unsupervised clustering models.
    """
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
    plt.title("5 Unsupervised Clustering ML Models Performance & Accuracy Comparison - Customer RFM Segmentation", fontsize=13, fontweight='bold')
    plt.ylim(0, 1.12)
    plt.xlabel("Unsupervised Clustering Model", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')

    for p in ax.patches:
        height = p.get_height()
        if height > 0.04:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontsize=8, rotation=0, xytext=(0, 2),
                textcoords='offset points'
            )

    plt.tight_layout()
    if output_dir:
        save_path = os.path.join(output_dir, "model_comparison.png")
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved 5-model clustering comparison chart to: {save_path}")
    plt.close()


def plot_roc_pr_curves(predictions_dict: dict, y_true: np.ndarray, output_dir: str = None):
    """
    Plots ROC and Precision-Recall Curves for all 5 unsupervised clustering models.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for name, res in predictions_dict.items():
        scores = res['scores']
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_val = roc_auc_score(y_true, scores)
        ax1.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", lw=2)

        prec, rec, _ = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        ax2.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.3f})", lw=2)

    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Baseline')
    ax1.set_title("ROC Curves - Customer RFM Segmentation (5 Clustering Models)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Precision-Recall Curves - Customer RFM Segmentation", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        save_path = os.path.join(output_dir, "roc_pr_curves.png")
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved ROC/PR curves plot to: {save_path}")
    plt.close()


FEATURE_EXPLANATIONS = {
    'rfm_composite_score': '(Composite Customer RFM Behavioral Value Score [0-100] combining Recency 30%, Frequency 35%, and Monetary Value 35%)',
    'rfm_churn_risk_index': '(Customer RFM Churn & Dormancy Risk Index [0-100], defined as 100 - rfm_composite_score)',
    'recency_score_norm': '(Normalized Customer Recency Score [0-100] decaying exponentially with days since last commercial activity)',
    'frequency_score_norm': '(Normalized Customer Purchase Frequency Score [0-100] scaled on log billing & order count relative to 99th percentile)',
    'monetary_score_norm': '(Normalized Customer Monetary Value Score [0-100] scaled on log net revenue relative to 99th percentile)',
    'recency_billing_days': '(Days elapsed since customer most recent SAP billing invoice VBRK-FKDAT relative to snapshot date)',
    'recency_order_days': '(Days elapsed since customer most recent SAP sales order VBAK-ERDAT relative to snapshot date)',
    'recency_effective_days': '(Minimum elapsed days since last commercial touchpoint across VBRK billing, VBAK orders, and BSID AR postings)',
    'customer_tenure_days': '(Total customer relationship tenure in days from earliest KNA1-ERDAT / first billing date to snapshot date)',
    'active_span_days': '(Active commercial purchasing lifespan in days between first billing VBRK-FKDAT and last billing VBRK-FKDAT)',
    'recency_to_tenure_ratio': '(Ratio of recency_effective_days to customer_tenure_days measuring dormancy relative to total relationship length)',
    'billing_doc_count': '(Total count of distinct SAP billing documents VBRK-VBELN generated for the customer)',
    'log_billing_frequency': '(Log-stabilized billing document frequency ln(1 + billing_doc_count))',
    'sales_order_count': '(Total count of distinct SAP sales orders VBAK-VBELN placed by the customer)',
    'log_order_frequency': '(Log-stabilized sales order frequency ln(1 + sales_order_count))',
    'distinct_billing_days': '(Number of unique calendar dates on which the customer was billed in VBRK)',
    'active_billing_months': '(Number of distinct calendar year-months with active customer billing transactions)',
    'monthly_billing_velocity': '(Average billing documents per active month billing_doc_count / active_billing_months)',
    'inter_purchase_interval_days': '(Average elapsed days between consecutive customer billing events active_span_days / (billing_doc_count - 1))',
    'order_to_bill_conversion_ratio': '(Order fulfillment conversion ratio of billed invoices VBRK to sales orders VBAK)',
    'total_billed_net_value': '(Total cumulative net billed revenue SUM(VBRK-NETWR) in INR across active invoices)',
    'log_monetary_value': '(Log-stabilized total customer monetary value ln(1 + total_billed_net_value))',
    'avg_bill_value': '(Mean net monetary value per billing document AVG(VBRK-NETWR) in INR)',
    'log_avg_bill_value': '(Log-stabilized average invoice value ln(1 + avg_bill_value))',
    'max_bill_value': '(Maximum single invoice net value MAX(VBRK-NETWR) in INR capturing peak order size)',
    'std_bill_value': '(Standard deviation of invoice net values STDDEV(VBRK-NETWR) measuring order size dispersion)',
    'bill_value_cv': '(Coefficient of variation of invoice values std_bill_value / avg_bill_value measuring spend volatility)',
    'total_billed_tax_value': '(Cumulative tax amount billed SUM(VBRK-MWSBK) in INR across customer invoices)',
    'total_order_net_value': '(Cumulative sales order net value SUM(VBAK-NETWR) in INR across customer orders)',
    'sales_org_monetary_zscore': '(Standard deviations of customer log monetary value relative to peers in the same Sales Organization VKORG)',
    'cancellation_return_rate': '(Proportion of billing documents that were cancelled VBRK-FKSTO=X or credit/return memos FKART in RE/G2/S1)',
    'ar_to_monetary_exposure_ratio': '(Ratio of open Accounts Receivable balance BSID-DMBTR to total billed net revenue measuring credit exposure)'
}


def compute_feature_importance(champion_model, feature_names: list, X_scaled: np.ndarray, y_true: np.ndarray = None):
    """
    Computes natural tree-based feature importance for Customer RFM Segmentation across all 32 features.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if len(X_scaled) > 25000:
        rng = np.random.RandomState(42)
        idx_samp = rng.choice(len(X_scaled), size=25000, replace=False)
        X_fit = X_scaled[idx_samp]
        y_fit = y_true[idx_samp] if y_true is not None else None
    else:
        X_fit = X_scaled
        y_fit = y_true

    if y_fit is not None and len(np.unique(y_fit)) > 1:
        model = RandomForestClassifier(n_estimators=100, max_depth=8, max_features=4, random_state=42, n_jobs=-1)
        model.fit(X_fit, y_fit)
        raw_importances = model.feature_importances_
    else:
        cluster_labels = champion_model.predict(X_fit[:, :5])
        model = RandomForestRegressor(n_estimators=100, max_depth=8, max_features=4, random_state=42, n_jobs=-1)
        model.fit(X_fit, cluster_labels)
        raw_importances = model.feature_importances_

    sort_indices = np.argsort(-raw_importances)
    sorted_features = [feature_names[i] for i in sort_indices]
    sorted_imps = raw_importances[sort_indices]

    explanations = [
        FEATURE_EXPLANATIONS.get(feat, f"(Statistical customer RFM behavioral feature for {feat})")
        for feat in sorted_features
    ]

    importance_df = pd.DataFrame({
        'feature': sorted_features,
        'importance': sorted_imps,
        'Direct Mathematical Influence On Customer RFM Segmentation & Value Risk:': explanations
    })

    return importance_df
