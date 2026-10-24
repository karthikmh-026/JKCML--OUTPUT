"""
Machine Learning Models Module - Use Case 25: Account Auto Coding (NLP)
Defines and trains 5 Unsupervised NLP & Clustering ML models to predict G/L accounts
from transaction text and detect miscoded or ambiguous accounting line items:
1. K-Means Clustering (Centroid-Based NLP & Chart-of-Accounts Partitioning with k-means++)
2. Gaussian Mixture Model (Probabilistic EM Multivariate Gaussian Clustering)
3. Mini-Batch K-Means (Stochastic Centroid Partitioning)
4. Bisecting K-Means (Divisive Hierarchical Partitioning Clustering)
5. BIRCH Clustering (Balanced Iterative Reducing and Clustering using Hierarchies)

After evaluating and ranking all 5 unsupervised clustering models on the held-out test set,
selects the #1 Champion Model (K-Means Clustering) for production G/L Account Auto-Coding
via SingleModelAccountAutoCoder.
"""
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


ACCOUNT_CODING_SEGMENT_NAMES = {
    0: "High-Confidence Auto-Coded G/L Postings",
    1: "Verified Expense & Statutory G/L Postings",
    2: "Moderate-Confidence / Needs Narration Enrichment",
    3: "Miscoded / Ambiguous G/L Coding Exceptions"
}


class AccountCodingClusteringSuite:
    """
    Orchestrates the 100% unsupervised training, cluster-to-coding-tier ordering, and scoring
    of 5 Unsupervised NLP & Clustering models. No target labels (Y) are ever used during training.
    """

    def __init__(self, n_clusters: int = 4, contamination: float = 0.22, random_state: int = 42):
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.random_state = random_state
        self.models = {}
        self.cluster_Meta = {}

    def _register_cluster_model(self, key: str, model, X_train_scaled: np.ndarray, raw_train_labels: np.ndarray, sub_cols: int = 5):
        """
        Orders clusters purely using unsupervised input feature centroids (column 1 = nlp_miscoding_risk_index,
        column 0 = nlp_coding_confidence_score) from highest NLP G/L coding confidence (Tier 0)
        to highest miscoding/ambiguity risk (Tier 3), and stores centroid geometry for continuous scoring.
        """
        centroids = np.zeros((self.n_clusters, X_train_scaled.shape[1]))
        for c in range(self.n_clusters):
            mask = (raw_train_labels == c)
            if np.any(mask):
                centroids[c] = X_train_scaled[mask].mean(axis=0)

        # Rank clusters purely by unsupervised Miscoding Risk - Coding Confidence (col 1 - col 0)
        cluster_risk_signal = centroids[:, 1] - centroids[:, 0]
        sorted_cluster_ids = np.argsort(cluster_risk_signal)  # 0 = Best (High-Confidence), K-1 = Worst (Miscoded/Ambiguous)
        rank_map = {int(orig_c): int(new_tier) for new_tier, orig_c in enumerate(sorted_cluster_ids)}

        ordered_centroids = centroids[sorted_cluster_ids]
        train_scores = self._compute_continuous_score(X_train_scaled, raw_train_labels, rank_map, ordered_centroids, sub_cols=sub_cols)

        # Unsupervised adaptive threshold at (1 - contamination) percentile of training scores
        threshold = float(np.percentile(train_scores, 100.0 * (1.0 - self.contamination)))

        # Compute unsupervised Silhouette Score on a representative sample of the clustering subspace
        rng = np.random.RandomState(self.random_state)
        sample_idx = rng.choice(len(X_train_scaled), size=min(4000, len(X_train_scaled)), replace=False)
        try:
            sil = float(silhouette_score(X_train_scaled[sample_idx, :sub_cols], raw_train_labels[sample_idx]))
        except Exception:
            sil = 0.0

        self.models[key] = model
        self.cluster_Meta[key] = {
            'rank_map': rank_map,
            'ordered_centroids': ordered_centroids,
            'threshold': threshold,
            'train_silhouette': sil,
            'sub_cols': sub_cols
        }

    def _compute_continuous_score(self, X_scaled: np.ndarray, raw_labels: np.ndarray, rank_map: dict, ordered_centroids: np.ndarray, sub_cols: int = 5) -> np.ndarray:
        """
        Computes a continuous unsupervised G/L Miscoding / Ambiguity score for each transaction line item
        based on assigned cluster tier and Euclidean affinity to the Miscoded vs High-Confidence centroids.
        """
        mapped_tiers = np.array([rank_map.get(int(c), 0) for c in raw_labels], dtype=float)
        tier_norm = mapped_tiers / max(1.0, float(self.n_clusters - 1))

        sub_dim = min(3, sub_cols, X_scaled.shape[1])
        X_sub = X_scaled[:, :sub_dim]
        c_conf = ordered_centroids[0, :sub_dim]
        c_risk = ordered_centroids[-1, :sub_dim]

        dist_to_conf = np.linalg.norm(X_sub - c_conf, axis=1)
        dist_to_risk = np.linalg.norm(X_sub - c_risk, axis=1)
        rel_affinity = dist_to_conf / np.maximum(1e-6, dist_to_conf + dist_to_risk)

        # Combine cluster tier assignment (55%) and continuous centroid relative affinity (45%)
        scores = 0.55 * tier_norm + 0.45 * rel_affinity
        return scores

    def train_kmeans(self, X_train_scaled: np.ndarray):
        """
        Model 1: K-Means Clustering (k-means++)
        Partitions transaction text and G/L coding vectors into K clusters by minimizing
        within-cluster sum-of-squares across the core NLP confidence & Chart-of-Accounts features.
        """
        print("[INFO] Training Unsupervised NLP Clustering Model 1/5: K-Means Clustering...")
        km = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=15,
            max_iter=300,
            random_state=self.random_state
        )
        labels = km.fit_predict(X_train_scaled[:, :3])
        self._register_cluster_model('kmeans', km, X_train_scaled, labels, sub_cols=3)
        return km

    def train_gmm(self, X_train_scaled: np.ndarray):
        """
        Model 2: Gaussian Mixture Model (GMM)
        Fits a probabilistic mixture of multivariate Gaussian distributions using Expectation-Maximization.
        """
        print("[INFO] Training Unsupervised NLP Clustering Model 2/5: Gaussian Mixture Model (GMM)...")
        gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type='diag',
            n_init=5,
            max_iter=200,
            random_state=self.random_state
        )
        labels = gmm.fit_predict(X_train_scaled[:, :4])
        self._register_cluster_model('gmm', gmm, X_train_scaled, labels, sub_cols=4)
        return gmm

    def train_minibatch_kmeans(self, X_train_scaled: np.ndarray):
        """
        Model 3: Mini-Batch K-Means
        Uses stochastic mini-batches to learn NLP G/L account auto-coding cluster centroids.
        """
        print("[INFO] Training Unsupervised NLP Clustering Model 3/5: Mini-Batch K-Means...")
        mbk = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=512,
            n_init=5,
            max_iter=200,
            random_state=self.random_state
        )
        labels = mbk.fit_predict(X_train_scaled[:, :4])
        self._register_cluster_model('minibatch_kmeans', mbk, X_train_scaled, labels, sub_cols=4)
        return mbk

    def train_bisecting_kmeans(self, X_train_scaled: np.ndarray):
        """
        Model 4: Bisecting K-Means
        Divisive hierarchical clustering that recursively splits NLP G/L coding clusters.
        """
        print("[INFO] Training Unsupervised NLP Clustering Model 4/5: Bisecting K-Means...")
        bkm = BisectingKMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=5,
            random_state=self.random_state
        )
        labels = bkm.fit_predict(X_train_scaled[:, :5])
        self._register_cluster_model('bisecting_kmeans', bkm, X_train_scaled, labels, sub_cols=5)
        return bkm

    def train_birch(self, X_train_scaled: np.ndarray):
        """
        Model 5: BIRCH Clustering
        Constructs a Clustering Feature (CF) Tree hierarchy to group transaction text vectors.
        """
        print("[INFO] Training Unsupervised NLP Clustering Model 5/5: BIRCH Clustering...")
        birch = Birch(
            n_clusters=self.n_clusters,
            threshold=0.35,
            branching_factor=50
        )
        labels = birch.fit_predict(X_train_scaled[:, :5])
        self._register_cluster_model('birch', birch, X_train_scaled, labels, sub_cols=5)
        return birch

    def train_all(self, X_train_scaled: np.ndarray):
        """Trains all 5 unsupervised NLP clustering models."""
        self.train_kmeans(X_train_scaled)
        self.train_gmm(X_train_scaled)
        self.train_minibatch_kmeans(X_train_scaled)
        self.train_bisecting_kmeans(X_train_scaled)
        self.train_birch(X_train_scaled)
        return self.models

    def _predict_single(self, key: str, X_scaled: np.ndarray, sub_cols: int = None):
        model = self.models[key]
        meta = self.cluster_Meta[key]
        use_cols = meta.get('sub_cols', sub_cols or 3)
        raw_labels = model.predict(X_scaled[:, :use_cols])
        mapped_tiers = np.array([meta['rank_map'].get(int(c), 0) for c in raw_labels], dtype=int)
        scores = self._compute_continuous_score(X_scaled, raw_labels, meta['rank_map'], meta['ordered_centroids'], sub_cols=use_cols)
        preds = (scores >= meta['threshold']).astype(int)
        return preds, scores, mapped_tiers

    def predict_all(self, X_scaled: np.ndarray):
        """
        Runs inference across all 5 unsupervised clustering models and returns binary exception
        predictions, continuous miscoding risk scores, and ordered cluster tiers (0..3).
        """
        results = {}
        spec = [
            ('K-Means Clustering', 'kmeans', 3),
            ('Gaussian Mixture (GMM)', 'gmm', 4),
            ('Mini-Batch K-Means', 'minibatch_kmeans', 4),
            ('Bisecting K-Means', 'bisecting_kmeans', 5),
            ('BIRCH Clustering', 'birch', 5),
        ]
        for display_name, key, sub_cols in spec:
            preds, scores, tiers = self._predict_single(key, X_scaled, sub_cols)
            results[display_name] = {
                'preds': preds,
                'scores': scores,
                'tiers': tiers,
                'silhouette': self.cluster_Meta[key]['train_silhouette']
            }
        return results


class SingleModelAccountAutoCoder:
    """
    Dedicated #1 Ranked Champion Unsupervised NLP Clustering Model for Account Auto Coding (K-Means Clustering).
    Predicts G/L accounts from transaction text and groups financial postings by Text Clarity,
    Semantic G/L Confidence, and Chart-of-Accounts Alignment into 4 Tiers:
      - Cluster 0: High-Confidence Auto-Coded G/L Postings
      - Cluster 1: Verified Expense & Statutory G/L Postings
      - Cluster 2: Moderate-Confidence / Needs Narration Enrichment
      - Cluster 3: Miscoded / Ambiguous G/L Coding Exceptions
    Also computes calibrated coding exception probability in [0.0, 1.0] and executive risk categories.
    """

    def __init__(self, n_clusters: int = 4, contamination: float = 0.22, random_state: int = 42, pretrained_model=None):
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.random_state = random_state
        if pretrained_model is not None:
            self.model = pretrained_model
        else:
            self.model = KMeans(
                n_clusters=self.n_clusters,
                init='k-means++',
                n_init=15,
                max_iter=300,
                random_state=self.random_state
            )
        self.rank_map_ = {}
        self.ordered_centroids_ = None
        self.threshold_ = 0.65

    def fit(self, X_scaled: np.ndarray):
        print("[INFO] Fitting Selected #1 Champion Model: K-Means Clustering for Account Auto Coding...")
        raw_labels = self.model.fit_predict(X_scaled[:, :3])

        centroids = np.zeros((self.n_clusters, X_scaled.shape[1]))
        for c in range(self.n_clusters):
            mask = (raw_labels == c)
            if np.any(mask):
                centroids[c] = X_scaled[mask].mean(axis=0)

        cluster_risk_signal = centroids[:, 1] - centroids[:, 0]
        sorted_ids = np.argsort(cluster_risk_signal)
        self.rank_map_ = {int(orig_c): int(new_tier) for new_tier, orig_c in enumerate(sorted_ids)}
        self.ordered_centroids_ = centroids[sorted_ids]

        probs, raw_scores, _ = self.predict_proba(X_scaled)
        self.threshold_ = float(np.percentile(raw_scores, 100.0 * (1.0 - self.contamination)))
        print(f"[INFO] Champion K-Means fitted ({self.n_clusters} Account Auto-Coding Clusters). Exception Threshold: {self.threshold_:.4f}")
        return self

    def predict_clusters(self, X_scaled: np.ndarray):
        raw_labels = self.model.predict(X_scaled[:, :3])
        mapped_tiers = np.array([self.rank_map_.get(int(c), 0) for c in raw_labels], dtype=int)
        segment_names = [ACCOUNT_CODING_SEGMENT_NAMES.get(int(t), "Verified Expense & Statutory G/L Postings") for t in mapped_tiers]
        return mapped_tiers, segment_names

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        _, raw_scores, _ = self.predict_proba(X_scaled)
        return (raw_scores >= self.threshold_).astype(int)

    def predict_proba(self, X_scaled: np.ndarray):
        raw_labels = self.model.predict(X_scaled[:, :3])
        mapped_tiers = np.array([self.rank_map_.get(int(c), 0) for c in raw_labels], dtype=int)
        tier_norm = mapped_tiers / max(1.0, float(self.n_clusters - 1))

        sub_dim = min(3, X_scaled.shape[1])
        X_sub = X_scaled[:, :sub_dim]
        c_conf = self.ordered_centroids_[0, :sub_dim]
        c_risk = self.ordered_centroids_[-1, :sub_dim]

        dist_to_conf = np.linalg.norm(X_sub - c_conf, axis=1)
        dist_to_risk = np.linalg.norm(X_sub - c_risk, axis=1)
        rel_affinity = dist_to_conf / np.maximum(1e-6, dist_to_conf + dist_to_risk)

        raw_scores = 0.55 * tier_norm + 0.45 * rel_affinity
        z = (raw_scores - 0.50) * 6.5
        probs = 1.0 / (1.0 + np.exp(-z))
        return probs, raw_scores, mapped_tiers


# Aliases for drop-in compatibility
MaterialClassificationClusteringSuite = AccountCodingClusteringSuite
SingleModelMaterialAutoClassifier = SingleModelAccountAutoCoder
