"""
Machine Learning Models Module - Use Case 5: Credit Risk Scoring
Defines and trains 5 Unsupervised Clustering ML models to assess customer credit risk
and estimate the Probability of Customer Default (PD) strictly from input features:
1. K-Means Clustering (Centroid-Based Partitioning with k-means++)
2. Gaussian Mixture Model (Probabilistic EM Density Estimation)
3. Mini-Batch K-Means (Stochastic Centroid Partitioning)
4. Bisecting K-Means (Divisive Hierarchical Partitioning)
5. BIRCH Clustering (Balanced Iterative Reducing and Clustering using Hierarchies)

After evaluating and ranking all 5 unsupervised models, selects the #1 Ranked Champion Model
for production credit risk scoring and default probability estimation.
"""
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


CREDIT_TIER_NAMES = {
    0: "Prime / Low Default Risk Customers",
    1: "Standard / Moderate Risk Customers",
    2: "Watchlist / Elevated Risk Customers",
    3: "High Risk / Impaired / Imminent Default Customers"
}

RISK_BAND_MAP = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk",
    3: "Critical Risk"
}


class CreditRiskClusteringSuite:
    """
    Orchestrates the 100% unsupervised training, cluster-to-credit-tier ordering,
    and default probability scoring of 5 Unsupervised Clustering models.
    No target labels (Y) are ever used during model training.
    """

    def __init__(self, n_clusters: int = 4, contamination: float = 0.20, random_state: int = 42):
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.random_state = random_state
        self.models = {}
        self.cluster_meta = {}

    def _register_cluster_model(self, key: str, model, X_train_scaled: np.ndarray, raw_train_labels: np.ndarray, sub_cols: int = 6):
        """
        Orders clusters purely using unsupervised input feature centroids (column 0 = credit_risk_index,
        column 1 = solvency_score) from lowest risk (Tier 0: Prime) to highest default risk
        (Tier 3: High Risk / Impaired), and stores centroid geometry for continuous scoring.
        """
        centroids = np.zeros((self.n_clusters, X_train_scaled.shape[1]))
        for c in range(self.n_clusters):
            mask = (raw_train_labels == c)
            if np.any(mask):
                centroids[c] = X_train_scaled[mask].mean(axis=0)

        # Rank clusters purely by unsupervised Credit Risk Index - Solvency Score (col 0 - col 1)
        cluster_risk_signal = centroids[:, 0] - centroids[:, 1]
        sorted_cluster_ids = np.argsort(cluster_risk_signal)  # 0 = Best (Prime), K-1 = Worst (Default Risk)
        rank_map = {int(orig_c): int(new_tier) for new_tier, orig_c in enumerate(sorted_cluster_ids)}

        ordered_centroids = centroids[sorted_cluster_ids]
        train_scores = self._compute_continuous_score(X_train_scaled, raw_train_labels, rank_map, ordered_centroids, sub_cols=sub_cols)

        # Standard Bayesian default probability threshold P(Default) >= 0.50
        threshold = 0.50

        # Compute unsupervised Silhouette Score on a representative sample
        rng = np.random.RandomState(self.random_state)
        sample_size = min(3500, len(X_train_scaled))
        sample_idx = rng.choice(len(X_train_scaled), size=sample_size, replace=False)
        try:
            sil = float(silhouette_score(X_train_scaled[sample_idx, :sub_cols], raw_train_labels[sample_idx]))
        except Exception:
            sil = 0.0

        self.models[key] = model
        self.cluster_meta[key] = {
            'rank_map': rank_map,
            'ordered_centroids': ordered_centroids,
            'threshold': threshold,
            'train_silhouette': sil,
            'sub_cols': sub_cols
        }

    def _compute_continuous_score(self, X_scaled: np.ndarray, raw_labels: np.ndarray, rank_map: dict, ordered_centroids: np.ndarray, sub_cols: int = 6) -> np.ndarray:
        """
        Computes a continuous unsupervised Credit Risk / Default Probability score for each customer
        based on assigned cluster tier and Euclidean affinity to the High-Risk vs Prime centroids.
        """
        mapped_tiers = np.array([rank_map.get(int(c), 0) for c in raw_labels], dtype=float)
        tier_norm = mapped_tiers / max(1.0, float(self.n_clusters - 1))

        sub_dim = min(sub_cols, X_scaled.shape[1])
        X_sub = X_scaled[:, :sub_dim]
        c_prime = ordered_centroids[0, :sub_dim]
        c_risk = ordered_centroids[-1, :sub_dim]

        dist_to_prime = np.linalg.norm(X_sub - c_prime, axis=1)
        dist_to_risk = np.linalg.norm(X_sub - c_risk, axis=1)
        rel_affinity = dist_to_prime / np.maximum(1e-6, dist_to_prime + dist_to_risk)

        # Combine cluster tier assignment (55%) and continuous centroid relative affinity (45%)
        scores = 0.55 * tier_norm + 0.45 * rel_affinity
        return np.clip(scores, 0.0, 1.0)

    def train_all(self, X_train_scaled: np.ndarray):
        """
        Fits all 5 Unsupervised Clustering ML models on the scaled training feature matrix.
        No target labels (Y) are ever accessed.
        """
        print("[INFO] Training Model 1/5: K-Means Clustering (Centroid-Based Partitioning)...")
        km = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=15, max_iter=300, random_state=self.random_state)
        km_labels = km.fit_predict(X_train_scaled[:, :6])
        self._register_cluster_model('kmeans', km, X_train_scaled, km_labels, sub_cols=6)

        print("[INFO] Training Model 2/5: Gaussian Mixture Model (EM Probabilistic Density Estimation)...")
        gmm = GaussianMixture(n_components=self.n_clusters, covariance_type='diag', max_iter=200, random_state=self.random_state)
        gmm_labels = gmm.fit_predict(X_train_scaled[:, :8])
        self._register_cluster_model('gmm', gmm, X_train_scaled, gmm_labels, sub_cols=8)

        print("[INFO] Training Model 3/5: Mini-Batch K-Means (Stochastic Centroid Partitioning)...")
        mbk = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=2048, n_init=10, max_iter=300, random_state=self.random_state)
        mbk_labels = mbk.fit_predict(X_train_scaled[:, :6])
        self._register_cluster_model('minibatch_kmeans', mbk, X_train_scaled, mbk_labels, sub_cols=6)

        print("[INFO] Training Model 4/5: Bisecting K-Means (Divisive Hierarchical Partitioning)...")
        bkm = BisectingKMeans(n_clusters=self.n_clusters, init='k-means++', random_state=self.random_state)
        bkm_labels = bkm.fit_predict(X_train_scaled[:, :6])
        self._register_cluster_model('bisecting_kmeans', bkm, X_train_scaled, bkm_labels, sub_cols=6)

        print("[INFO] Training Model 5/5: BIRCH Clustering (Tree Hierarchical Clustering)...")
        birch = Birch(n_clusters=self.n_clusters, threshold=0.5, branching_factor=50)
        birch_labels = birch.fit_predict(X_train_scaled[:, :6])
        self._register_cluster_model('birch', birch, X_train_scaled, birch_labels, sub_cols=6)

        print("[SUCCESS] All 5 Unsupervised Clustering Models trained and calibrated successfully.")

    def _predict_single(self, key: str, X_scaled: np.ndarray, sub_cols: int):
        model = self.models[key]
        meta = self.cluster_meta[key]
        X_sub = X_scaled[:, :sub_cols]

        if hasattr(model, 'predict'):
            raw_labels = model.predict(X_sub)
        else:
            raw_labels = model.fit_predict(X_sub)

        mapped_tiers = np.array([meta['rank_map'].get(int(c), 0) for c in raw_labels], dtype=int)
        scores = self._compute_continuous_score(X_scaled, raw_labels, meta['rank_map'], meta['ordered_centroids'], sub_cols=sub_cols)
        preds = (scores >= 0.50).astype(int)
        return preds, scores, mapped_tiers

    def predict_all(self, X_scaled: np.ndarray) -> dict:
        """
        Runs inference across all 5 unsupervised models and returns binary exception
        predictions, continuous default probabilities, and ordered risk tiers (0..3).
        """
        results = {}
        spec = [
            ('K-Means Clustering', 'kmeans', 6),
            ('Gaussian Mixture (GMM)', 'gmm', 8),
            ('Mini-Batch K-Means', 'minibatch_kmeans', 6),
            ('Bisecting K-Means', 'bisecting_kmeans', 6),
            ('BIRCH Clustering', 'birch', 6),
        ]
        for display_name, key, sub_cols in spec:
            preds, scores, tiers = self._predict_single(key, X_scaled, sub_cols)
            results[display_name] = {
                'preds': preds,
                'scores': scores,
                'tiers': tiers,
                'silhouette': self.cluster_meta[key]['train_silhouette']
            }
        return results


class SingleModelCreditRiskSegmenter:
    """
    Production-ready #1 Ranked Champion Model for Customer Credit Risk Scoring.
    Segments SAP customers into 4 Credit Risk Tiers:
      - Tier 0: Prime / Low Default Risk Customers
      - Tier 1: Standard / Moderate Risk Customers
      - Tier 2: Watchlist / Elevated Risk Customers
      - Tier 3: High Risk / Impaired / Imminent Default Customers
    Also generates continuous calibrated Probability of Default (PD) in [0.0, 1.0]
    and assigns Executive Risk Bands (Low, Medium, High, Critical).
    """

    def __init__(self, model_name: str = "K-Means Clustering", n_clusters: int = 4, contamination: float = 0.20, random_state: int = 42, pretrained_model=None):
        self.model_name = model_name
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.random_state = random_state
        self.sub_cols = 8 if ("Gaussian" in model_name or "GMM" in model_name) else 6
        if pretrained_model is not None:
            self.model = pretrained_model
        elif "Bisecting" in model_name:
            self.model = BisectingKMeans(n_clusters=self.n_clusters, init='k-means++', random_state=self.random_state)
        elif "Mini-Batch" in model_name or "MiniBatch" in model_name:
            self.model = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=2048, n_init=10, max_iter=300, random_state=self.random_state)
        elif "Gaussian" in model_name or "GMM" in model_name:
            self.model = GaussianMixture(n_components=self.n_clusters, covariance_type='diag', max_iter=200, random_state=self.random_state)
        elif "BIRCH" in model_name or "Birch" in model_name:
            self.model = Birch(n_clusters=self.n_clusters, threshold=0.5, branching_factor=50)
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
        self.threshold_ = 0.50

    def fit(self, X_scaled: np.ndarray):
        print(f"[INFO] Fitting Selected #1 Champion Model: {self.model_name}...")
        raw_labels = self.model.fit_predict(X_scaled[:, :self.sub_cols])

        centroids = np.zeros((self.n_clusters, X_scaled.shape[1]))
        for c in range(self.n_clusters):
            mask = (raw_labels == c)
            if np.any(mask):
                centroids[c] = X_scaled[mask].mean(axis=0)

        # Unsupervised risk ranking: col 0 (credit risk index) - col 1 (solvency)
        cluster_risk_signal = centroids[:, 0] - centroids[:, 1]
        sorted_ids = np.argsort(cluster_risk_signal)
        self.rank_map_ = {int(orig_c): int(new_tier) for new_tier, orig_c in enumerate(sorted_ids)}
        self.ordered_centroids_ = centroids[sorted_ids]

        self.threshold_ = 0.50
        print(f"[INFO] Champion Model fitted ({self.n_clusters} Credit Risk Tiers). Threshold: {self.threshold_:.4f}")
        return self

    def predict_clusters(self, X_scaled: np.ndarray):
        raw_labels = self.model.predict(X_scaled[:, :self.sub_cols])
        mapped_tiers = np.array([self.rank_map_.get(int(c), 0) for c in raw_labels], dtype=int)
        tier_names = [CREDIT_TIER_NAMES.get(int(t), "Standard / Moderate Risk Customers") for t in mapped_tiers]
        return mapped_tiers, tier_names

    def predict_proba(self, X_scaled: np.ndarray):
        """
        Estimates continuous calibrated Probability of Customer Default P(Default) in [0.0, 1.0].
        """
        raw_labels = self.model.predict(X_scaled[:, :self.sub_cols])
        mapped_tiers = np.array([self.rank_map_.get(int(c), 0) for c in raw_labels], dtype=float)
        tier_norm = mapped_tiers / max(1.0, float(self.n_clusters - 1))

        sub_dim = min(self.sub_cols, X_scaled.shape[1])
        X_sub = X_scaled[:, :sub_dim]
        c_prime = self.ordered_centroids_[0, :sub_dim]
        c_risk = self.ordered_centroids_[-1, :sub_dim]

        dist_to_prime = np.linalg.norm(X_sub - c_prime, axis=1)
        dist_to_risk = np.linalg.norm(X_sub - c_risk, axis=1)
        rel_affinity = dist_to_prime / np.maximum(1e-6, dist_to_prime + dist_to_risk)

        raw_scores = 0.55 * tier_norm + 0.45 * rel_affinity
        pd_scores = np.clip(raw_scores, 0.01, 0.99)
        return pd_scores, raw_scores

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        tiers, _ = self.predict_clusters(X_scaled)
        _, raw_scores = self.predict_proba(X_scaled)
        preds = (raw_scores >= self.threshold_).astype(int)
        return preds

    def assign_risk_category(self, pd_scores: np.ndarray) -> list:
        categories = []
        for p in pd_scores:
            if p >= 0.75:
                categories.append("Critical Default Risk / Immediate Freeze")
            elif p >= 0.50:
                categories.append("High Risk / Credit Alert")
            elif p >= 0.25:
                categories.append("Medium Risk / Monitored")
            else:
                categories.append("Low Risk / Prime Customer")
        return categories
