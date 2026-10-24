"""
Evaluation and Comparison Module - Use Case 24: Material Auto Classification (NLP)
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
    benchmark material classification exception labels, plus unsupervised Silhouette Score.
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
    plt.title("5 Unsupervised NLP Clustering ML Models Performance & Accuracy Comparison - Material Auto Classification", fontsize=13, fontweight='bold')
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
    ax1.set_title("ROC Curves - Material Auto Classification (5 Unsupervised Models)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Precision-Recall Curves - Material Auto Classification", fontsize=13, fontweight='bold')
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
    'nlp_classification_confidence_score': '(Composite NLP Material Auto-Classification Confidence Score [0-100] combining Text Clarity 34%, Semantic Confidence 36%, and Taxonomy Alignment 30%)',
    'nlp_misclassification_risk_index': '(Composite Material Misclassification & Ambiguity Risk Index [0-100], defined as 100 - nlp_classification_confidence_score)',
    'text_clarity_score': '(Normalized Material Description Clarity & Specification Completeness Score [0-100] from MAKT-MAKTX length, word count, dimensional specs, and TF-IDF richness)',
    'semantic_confidence_score': '(Normalized NLP Semantic Domain Classification Confidence [0-100] measuring dominant taxonomy affinity and category margin)',
    'taxonomy_alignment_score': '(Normalized Master Taxonomy Alignment Score [0-100] measuring agreement between NLP-inferred class from MAKTX and SAP MTART/MATKL/EKPO-MATKL)',
    'nlp_spare_machinery_affinity': '(Unsupervised NLP lexical affinity [0-100] to Mechanical & Plant Machinery Spare Parts tokens: BEARING, SEAL, PUMP, VALVE, KILN, CRUSHER, MILL, GEARBOX)',
    'nlp_elec_inst_affinity': '(Unsupervised NLP lexical affinity [0-100] to Electrical, Instrumentation & Automation tokens: CABLE, MOTOR, SENSOR, SWITCH, PLC, BREAKER, RELAY, VFD)',
    'nlp_consumable_fastener_affinity': '(Unsupervised NLP lexical affinity [0-100] to Fasteners, Hardware, Welding & Consumable tokens: BOLT, NUT, WASHER, ELECTRODE, PIPE, BELT, OIL, SAFETY)',
    'nlp_finished_paint_cement_affinity': '(Unsupervised NLP lexical affinity [0-100] to Finished Cement, Concrete, Paint & Coating tokens: CEMENT, CONCRET, RMC, PUTTY, PRIMER, EMULSION, ENAMEL, GROUT)',
    'nlp_packaging_raw_affinity': '(Unsupervised NLP lexical affinity [0-100] to Packaging & Raw/Chemical Material tokens: BAG, BOX, CARTON, DRUM, BOPP, GYPSUM, CLINKER, COAL, SOLVENT, PIGMENT)',
    'desc_char_length': '(Total character length of SAP Material Description MAKT-MAKTX measuring descriptive completeness)',
    'desc_word_count': '(Count of distinct alphanumeric word tokens in SAP Material Description MAKT-MAKTX)',
    'desc_avg_word_len': '(Mean character length per token in MAKT-MAKTX distinguishing technical nomenclature from cryptic codes)',
    'desc_digit_ratio': '(Proportion of numeric digits in MAKT-MAKTX capturing model numbers, grades, and part dimensions)',
    'desc_alpha_ratio': '(Proportion of alphabetic characters in MAKT-MAKTX capturing linguistic noun-phrase richness)',
    'desc_spec_token_count': '(Count of structured engineering/dimensional specification tokens in MAKTX/GROES: MM, KG, LTR, KW, V, RPM, M25, NxM)',
    'tfidf_active_ngrams': '(Number of non-zero TF-IDF unigram and bigram features activated by the material description)',
    'tfidf_semantic_energy': '(L1 norm of sublinear TF-IDF n-gram activations measuring semantic information density of MAKTX)',
    'svd_semantic_comp_1': '(1st Latent Semantic Analysis TruncatedSVD component of the TF-IDF description matrix)',
    'svd_semantic_comp_2': '(2nd Latent Semantic Analysis TruncatedSVD component separating mechanical spares from commercial/paint products)',
    'svd_semantic_comp_3': '(3rd Latent Semantic Analysis TruncatedSVD component capturing electrical/instrumentation and fastener semantics)',
    'svd_semantic_comp_4': '(4th Latent Semantic Analysis TruncatedSVD component capturing packaging, raw material, and bulk cement n-grams)',
    'svd_semantic_comp_5': '(5th Latent Semantic Analysis TruncatedSVD component capturing dimensional and grade specification patterns)',
    'po_desc_consistency_score': '(Lexical consistency score [0-100] between Material Master Description MAKT-MAKTX and Purchase Order Short Text EKPO-TXZ01)',
    'dup_desc_count': '(Number of distinct Material IDs MARA-MATNR sharing the exact same normalized material description)',
    'dup_desc_diff_matkl_count': '(Number of distinct SAP Material Groups MARA-MATKL assigned to identical material descriptions indicating taxonomy inconsistency)',
    'is_cryptic_or_short_desc': '(Binary indicator = 1 when MAKT-MAKTX has < 9 chars, <= 1 word token, or < 25% alphabetic characters)',
    'is_unstandardized_matkl': '(Binary indicator = 1 when MARA-MATKL is blank, purely numeric code, or generic placeholder)',
    'is_po_matkl_mismatch': '(Binary indicator = 1 when Purchase Order Material Group EKPO-MATKL conflicts with Master Material Group MARA-MATKL)',
    'is_nlp_mtart_conflict': '(Binary indicator = 1 when NLP semantic domain extracted from MAKTX conflicts with assigned SAP Material Type MARA-MTART)',
    'log_effective_unit_price': '(Log-stabilized effective unit price ln(1 + effective_unit_price) from MBEW-STPRS/VERPR and EKPO-NETPR)',
    'matkl_price_zscore': '(Standardized Z-score of material unit price relative to peers in the same SAP Material Group MARA-MATKL)'
}


def compute_feature_importance(champion_model, feature_names: list, X_scaled: np.ndarray, y_true: np.ndarray = None):
    """
    Computes natural tree-based feature importance for Material Auto Classification across all 32 features.
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
        FEATURE_EXPLANATIONS.get(feat, f"(NLP & Material Taxonomy classification feature for {feat})")
        for feat in sorted_features
    ]

    importance_df = pd.DataFrame({
        'feature': sorted_features,
        'importance': sorted_imps,
        'Direct Mathematical Influence On Material Auto Classification & Taxonomy:': explanations
    })

    return importance_df
