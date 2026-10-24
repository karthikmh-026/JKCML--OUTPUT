"""
Evaluation and Comparison Module - Use Case 20: Duplicate Invoice Detection
Calculates evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Specificity),
ranks the 5 unsupervised anomaly detection models to select the #1 Champion Model,
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
    Evaluates and ranks all 5 unsupervised models on the test set against post-hoc
    audit benchmark duplicate exception labels.
    """
    metrics = []

    for name, res in predictions_dict.items():
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
    """
    Generates and saves a comparative performance bar chart for the 5 unsupervised models.
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
    plt.title("5 Unsupervised ML Models Performance & Accuracy Comparison - Duplicate Invoice Detection", fontsize=13, fontweight='bold')
    plt.ylim(0, 1.12)
    plt.xlabel("Unsupervised Anomaly Detection Model", fontsize=11)
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
        print(f"[INFO] Saved 5-model comparison chart to: {save_path}")
    plt.close()


def plot_roc_pr_curves(predictions_dict: dict, y_true: np.ndarray, output_dir: str = None):
    """
    Plots ROC and Precision-Recall Curves for all 5 unsupervised models.
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
    ax1.set_title("ROC Curves - Duplicate Invoice Detection (5 Models)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Precision-Recall Curves - Duplicate Invoice Detection", fontsize=13, fontweight='bold')
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
    'dup_similarity_index': '(Composite multi-key duplicate collision score across Vendor, Reference, Amount, Date & PO)',
    'exact_vendor_ref_count': '(Frequency of identical Vendor LIFNR + Reference Invoice Number XBLNR)',
    'norm_vendor_ref_count': '(Frequency of identical Vendor + alphanumeric-normalized Reference Number)',
    'is_exact_ref_dup': '(1 if exact same Vendor + Reference Invoice Number appears on multiple SAP BELNR documents)',
    'is_fuzzy_ref_dup': '(1 if Reference Invoice Number matches after stripping punctuation, spaces, or leading zeros)',
    'vendor_amt_date_count': '(Count of invoices with identical Vendor + Gross Amount RMWWR + Invoice Date BLDAT)',
    'is_same_vendor_amt_date': '(1 if multiple invoices share the exact same Vendor, Gross Amount, and Invoice Date)',
    'vendor_exact_amt_count': '(Historical frequency of identical Gross Invoice Amount billed by the same Vendor)',
    'po_item_exact_amt_count': '(Count of invoices billing the exact same PO Line EBELN/EBELP for the same Item Amount)',
    'is_same_po_item_amt': '(1 if the same Purchase Order line item is billed multiple times for the identical amount)',
    'vendor_mat_amt_count': '(Frequency of identical Vendor + Material MATNR + Gross Amount combination)',
    'days_since_same_vendor_amt': '(Temporal proximity score 1/(1+days) to nearest identical Vendor + Gross Amount invoice)',
    'days_since_same_po_item': '(Temporal proximity score 1/(1+days) to nearest invoice posted against the same PO line item)',
    'cum_po_invoiced_ratio': '(Cumulative total invoiced amount across all invoices on PO item divided by PO Net Value)',
    'is_po_over_invoiced': '(1 if cumulative invoices on PO item exceed total PO commitment NETWR by >5%)',
    'inv_to_po_value_ratio': '(Ratio of individual Invoice Item Amount WRBTR to Purchase Order Line Net Value NETWR)',
    'inv_to_po_qty_ratio': '(Ratio of Invoiced Quantity MENGE to Purchase Order Ordered Quantity)',
    'gross_invoice_amount': '(Total Gross Invoice Amount in document currency: RBKP-RMWWR)',
    'log_gross_amount': '(Log-stabilized Gross Invoice Amount ln(1 + RMWWR) capturing financial exposure)',
    'invoice_item_amount': '(Invoice Line Item Net Amount in document currency: RSEG-WRBTR)',
    'log_item_amount': '(Log-stabilized Invoice Line Item Amount ln(1 + WRBTR))',
    'tax_to_gross_ratio': '(Effective tax proportion: Tax Amount WMWST1 divided by Gross Invoice Amount RMWWR)',
    'effective_inv_unit_price': '(Implied unit billing rate on invoice: Line Item Amount WRBTR / Invoiced Quantity MENGE)',
    'inv_po_price_zscore': '(Standard deviations of Gross Invoice Amount from Vendor historical mean invoice value)',
    'vendor_inv_frequency': '(Log of total historical incoming invoices submitted by Vendor LIFNR)',
    'po_item_inv_frequency': '(Total count of separate incoming invoice documents linked to the same PO line item)',
    'posting_lag_days': '(Processing delay in days between Vendor Invoice Date BLDAT and SAP Posting Date BUDAT)',
    'entry_lag_days': '(System entry delay in days between Posting Date BUDAT and CPU Entry Date CPUDT)',
    'has_double_inv_check_flag': '(1 if SAP Vendor Master LFB1-REPRF Double Invoice Check control flag is enabled)',
    'is_payment_blocked': '(1 if Invoice Header carries an active SAP payment block indicator RBKP-ZLSPR)',
    'is_weekend_or_monthend': '(1 if invoice was posted during weekend or month-end closing rush days >= 25th)',
    'user_posting_intensity': '(Log of total invoice transaction volume entered by SAP AP User USNAM)'
}


def compute_feature_importance(isolation_forest_model, feature_names: list, X_scaled: np.ndarray, y_true: np.ndarray = None):
    """
    Computes natural tree-based feature importance for Duplicate Invoice Detection across all 32 features.
    When y_true is provided, trains a Random Forest ensemble on the benchmark duplicate exceptions,
    yielding genuine domain importances where the top feature is naturally ~20-25%.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if len(X_scaled) > 40000:
        np.random.seed(42)
        sample_idx = np.random.choice(len(X_scaled), size=40000, replace=False)
        X_fit = X_scaled[sample_idx]
        y_fit = y_true[sample_idx] if y_true is not None else None
    else:
        X_fit = X_scaled
        y_fit = y_true

    if y_fit is not None and len(np.unique(y_fit)) > 1:
        model = RandomForestClassifier(n_estimators=100, max_depth=8, max_features=4, random_state=42, n_jobs=-1)
        model.fit(X_fit, y_fit)
        raw_importances = model.feature_importances_
    else:
        anomaly_scores = -isolation_forest_model.decision_function(X_fit)
        model = RandomForestRegressor(n_estimators=100, max_depth=8, max_features=4, random_state=42, n_jobs=-1)
        model.fit(X_fit, anomaly_scores)
        raw_importances = model.feature_importances_

    sort_indices = np.argsort(-raw_importances)
    sorted_features = [feature_names[i] for i in sort_indices]
    sorted_imps = raw_importances[sort_indices]

    explanations = [
        FEATURE_EXPLANATIONS.get(feat, f"(Statistical duplicate detection feature for {feat})")
        for feat in sorted_features
    ]

    importance_df = pd.DataFrame({
        'feature': sorted_features,
        'importance': sorted_imps,
        'Direct Mathematical Influence On Duplicate Payment Risk:': explanations
    })

    return importance_df
