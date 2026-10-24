"""
Evaluation and Comparison Module - Use Case 25: Account Auto Coding (NLP)
Calculates evaluation metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Specificity,
and Silhouette Score), ranks the 5 Unsupervised NLP & Clustering models to select the #1 Champion Model,
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
    Evaluates and ranks all 5 Unsupervised NLP & Clustering models on the test set against post-hoc
    benchmark G/L account coding exception labels, plus unsupervised Silhouette Score.
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
        .sort_values(by=['ROC-AUC', 'PR-AUC', 'F1-Score', 'Accuracy'], ascending=[False, False, False, False])
        .reset_index(drop=True)
    )
    comparison_df.insert(0, 'Rank', [f"#{i+1}" for i in range(len(comparison_df))])
    return comparison_df


def plot_model_comparison(comparison_df: pd.DataFrame, output_dir: str = None):
    """
    Generates and saves a comparative performance bar chart for the 5 unsupervised NLP clustering models.
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
    plt.title("5 Unsupervised NLP Clustering ML Models Performance & Accuracy Comparison - Account Auto Coding", fontsize=13, fontweight='bold')
    plt.ylim(0, 1.12)
    plt.xlabel("Unsupervised NLP Clustering Model", fontsize=11)
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
    Plots ROC and Precision-Recall Curves for all 5 unsupervised NLP clustering models.
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
    ax1.set_title("ROC Curves - Account Auto Coding (5 Unsupervised Models)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Precision-Recall Curves - Account Auto Coding", fontsize=13, fontweight='bold')
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
    'nlp_coding_confidence_score': '(Composite NLP G/L Account Auto-Coding Confidence Score [0-100] combining Text Clarity 34%, Semantic GL Confidence 36%, and CoA Alignment 30%)',
    'nlp_miscoding_risk_index': '(Composite G/L Account Miscoding & Ambiguity Risk Index [0-100], defined as 100 - nlp_coding_confidence_score)',
    'text_clarity_score': '(Normalized Transaction Text Clarity & Narration Completeness Score [0-100] from BSEG-SGTXT length, word count, accounting refs, and TF-IDF richness)',
    'semantic_gl_confidence_score': '(Normalized NLP Semantic G/L Prediction Confidence [0-100] measuring dominant accounting domain affinity and category margin)',
    'coa_alignment_score': '(Normalized Chart of Accounts Alignment Score [0-100] measuring agreement between NLP-inferred G/L from BSEG-SGTXT and posted HKONT/KTOKS/KOSTL)',
    'nlp_freight_logistics_affinity': '(Unsupervised NLP lexical affinity [0-100] to Freight, Logistics & Material Handling G/L tokens: FREIGHT, LOADING, UNLOADING, HANDLING, WHARFAGE, RAIL, DUMP)',
    'nlp_plant_maint_rent_affinity': '(Unsupervised NLP lexical affinity [0-100] to Plant Maintenance, Rent & Facility G/L tokens: RENT, GODOWN, REPAIR, MAINTENANCE, CABLE, CRANE, WATER, CANTEEN)',
    'nlp_sales_promo_comm_affinity': '(Unsupervised NLP lexical affinity [0-100] to Sales Promotion, Advertising & Commission G/L tokens: ADVERTISEMENT, PUBLICITY, COMMISSION, INCENTIVE, BRANDING)',
    'nlp_travel_it_prof_affinity': '(Unsupervised NLP lexical affinity [0-100] to Travel, IT, Telecom & Professional Fee G/L tokens: TAXI, TRAVEL, HOTEL, INTERVIEW, MOBILE, INTERNET, CONSULTANCY)',
    'nlp_tax_bank_prov_affinity': '(Unsupervised NLP lexical affinity [0-100] to Statutory Tax, GST/TDS, Banking & Provision G/L tokens: TDS, CGST, SGST, IGST, ITC, BANK, LC, PROVISION, ADVANCE)',
    'sgtxt_char_length': '(Total character length of SAP Accounting Line Item Text BSEG-SGTXT measuring narration completeness)',
    'sgtxt_word_count': '(Count of distinct alphanumeric word tokens in SAP Transaction Text BSEG-SGTXT)',
    'sgtxt_avg_word_len': '(Mean character length per token in BSEG-SGTXT distinguishing descriptive accounting narrations from 2-letter codes)',
    'sgtxt_digit_ratio': '(Proportion of numeric digits in BSEG-SGTXT capturing invoice numbers, dates, and PO/OL reference codes)',
    'sgtxt_alpha_ratio': '(Proportion of alphabetic characters in BSEG-SGTXT capturing natural language accounting narration density)',
    'accounting_ref_token_count': '(Count of structured accounting period and reference tokens in BSEG-SGTXT: Month/Year, PO/OL/Bill numbers, MT/TON units)',
    'tfidf_active_ngrams': '(Number of non-zero TF-IDF unigram and bigram features activated by the transaction text BSEG-SGTXT)',
    'tfidf_semantic_energy': '(L1 norm of sublinear TF-IDF n-gram activations measuring semantic information density of BSEG-SGTXT)',
    'svd_semantic_comp_1': '(1st Latent Semantic Analysis TruncatedSVD component of the TF-IDF transaction text matrix)',
    'svd_semantic_comp_2': '(2nd Latent Semantic Analysis TruncatedSVD component separating freight/handling narrations from plant/overhead postings)',
    'svd_semantic_comp_3': '(3rd Latent Semantic Analysis TruncatedSVD component capturing statutory tax/TDS and vendor service semantics)',
    'svd_semantic_comp_4': '(4th Latent Semantic Analysis TruncatedSVD component capturing sales promotion, commission, and travel narrations)',
    'svd_semantic_comp_5': '(5th Latent Semantic Analysis TruncatedSVD component capturing godown rent, maintenance, and telecom patterns)',
    'text_gl_desc_similarity_score': '(Semantic and lexical agreement score [0-100] between Transaction Text BSEG-SGTXT and G/L Account Description SKAT-TXT50)',
    'dup_sgtxt_count': '(Number of accounting line items in BSEG sharing the exact same normalized transaction text narration)',
    'dup_sgtxt_diff_gl_count': '(Number of distinct G/L Accounts BSEG-HKONT assigned to identical transaction text narrations indicating coding inconsistency)',
    'is_cryptic_or_code_only_sgtxt': '(Binary indicator = 1 when BSEG-SGTXT has <= 5 chars, <= 1 word token, or < 25% alphabetic characters such as FE, EE, 18, 04)',
    'is_generic_provision_or_clearing_gl': '(Binary indicator = 1 when specific operational expense narration is parked in generic provision/clearing G/L 173101/171001)',
    'is_missing_cost_object_on_pl': '(Binary indicator = 1 when a P&L expense G/L account SKA1-GLACCOUNT_TYPE=P is posted without Cost Center KOSTL or Order AUFNR)',
    'is_nlp_gl_domain_conflict': '(Binary indicator = 1 when NLP semantic accounting domain extracted from BSEG-SGTXT conflicts with posted G/L Account HKONT)',
    'log_amount_local': '(Log-stabilized transaction posting amount in local currency ln(1 + |BSEG-DMBTR|))',
    'gl_account_amount_zscore': '(Standardized Z-score of line item posting amount relative to historical postings in the same G/L Account BSEG-HKONT)'
}


def compute_feature_importance(champion_model, feature_names: list, X_scaled: np.ndarray, y_true: np.ndarray = None):
    """
    Computes natural tree-based feature importance for Account Auto Coding across all 32 features.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if y_true is not None and len(np.unique(y_true)) > 1:
        model = RandomForestClassifier(n_estimators=120, max_depth=8, max_features=4, random_state=42, n_jobs=-1)
        model.fit(X_scaled, y_true)
        raw_importances = model.feature_importances_
    else:
        cluster_labels = champion_model.predict(X_scaled[:, :3])
        model = RandomForestRegressor(n_estimators=120, max_depth=8, max_features=4, random_state=42, n_jobs=-1)
        model.fit(X_scaled, cluster_labels)
        raw_importances = model.feature_importances_

    sort_indices = np.argsort(-raw_importances)
    sorted_features = [feature_names[i] for i in sort_indices]
    sorted_imps = raw_importances[sort_indices]

    explanations = [
        FEATURE_EXPLANATIONS.get(feat, f"(NLP & Chart-of-Accounts G/L auto-coding feature for {feat})")
        for feat in sorted_features
    ]

    importance_df = pd.DataFrame({
        'feature': sorted_features,
        'importance': sorted_imps,
        'Direct Mathematical Influence On GL Account Auto Coding & CoA Governance:': explanations
    })

    return importance_df
