"""
Machine Learning Models Module - Use Case 6: Slow Moving Inventory Prediction
Defines and trains 5 Unsupervised Clustering ML models to identify slow-moving/obsolete inventory
and estimate the Probability of Obsolescence P(Slow Moving) strictly from input features:
1. Mini-Batch K-Means (Fast Stochastic Centroid Partitioning)
2. Gaussian Mixture Model (Probabilistic EM Density Estimation)
3. K-Means Clustering (Centroid-Based Partitioning with k-means++)
4. Bisecting K-Means (Divisive Hierarchical Partitioning)
5. BIRCH Clustering (Balanced Iterative Reducing and Clustering using Hierarchies)

After evaluating and ranking all 5 unsupervised models, selects the #1 Ranked Champion Model
for production slow-moving inventory scoring and obsolescence probability estimation.
"""
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


INVENTORY_TIER_NAMES = {
    0: "Fast Moving / High Velocity Stock",
    1: "Normal / Steady Moving Stock",
    2: "Slow Moving / Stagnant Stock",
    3: "Dormant / Obsolete Stock"
}

RISK_BAND_MAP = {
    0: "Low Obsolescence Risk",
    1: "Moderate Slow Risk",
    2: "High Slow Risk",
    3: "Critical Obsolete Risk"
}


class InventoryClusteringSuite:
    """
    Orchestrates the 100% unsupervised training, cluster-to-tier ordering,
    and obsolescence probability scoring of 5 Unsupervised Clustering models.
    No target labels (Y) are ever used during model training.
    """

    def __init__(self, n_clusters: int = 4, contamination: float = 0.35, random_state: int = 42):
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.random_state = random_state
        self.models = {}
        self.cluster_meta = {}

    def _register_cluster_model(self, key: str, model, X_train_scaled: np.ndarray, raw_train_labels: np.ndarray, signal_col_pos: int = 25, signal_col_neg: int = 8):
        """
        Orders clusters purely using unsupervised input feature centroids
        (col 25 = obsolescence_risk_index, col 8 = inventory_turnover_ratio)
        from lowest risk (Tier 0: Fast Moving) to highest obsolescence risk (Tier 3: Dormant/Obsolete).
        """
        centroids = np.zeros((self.n_clusters, X_train_scaled.shape[1]))
        for c in range(self.n_clusters):
            mask = (raw_train_labels == c)
            if np.any(mask):
                centroids[c] = X_train_scaled[mask].mean(axis=0)

        # Rank clusters purely by Unsupervised Obsolescence Signal (col 25 - col 8)
        cluster_risk_signal = centroids[:, signal_col_pos] - centroids[:, signal_col_neg]
        sorted_cluster_ids = np.argsort(cluster_risk_signal)
        rank_map = {int(orig_c): int(new_tier) for new_tier, orig_c in enumerate(sorted_cluster_ids)}

        ordered_centroids = centroids[sorted_cluster_ids]
        train_scores = self._compute_continuous_score(X_train_scaled, raw_train_labels, rank_map, ordered_centroids)

        # Unsupervised adaptive threshold at (1 - contamination) percentile of training scores
        threshold = float(np.percentile(train_scores, 100.0 * (1.0 - self.contamination)))

        # Compute unsupervised Silhouette Score on a representative sample
        rng = np.random.RandomState(self.random_state)
        sample_size = min(3500, len(X_train_scaled))
        sample_idx = rng.choice(len(X_train_scaled), size=sample_size, replace=False)
        try:
            sil = float(silhouette_score(X_train_scaled[sample_idx, :8], raw_train_labels[sample_idx]))
        except Exception:
            sil = 0.0

        self.models[key] = model
        self.cluster_meta[key] = {
            'rank_map': rank_map,
            'ordered_centroids': ordered_centroids,
            'threshold': threshold,
            'train_silhouette': sil
        }

    def _compute_continuous_score(self, X_scaled: np.ndarray, raw_labels: np.ndarray, rank_map: dict, ordered_centroids: np.ndarray) -> np.ndarray:
        """
        Computes a continuous unsupervised Obsolescence Probability score for each material
        based on assigned cluster tier and Euclidean affinity to the Obsolete vs Fast Moving centroids.
        """
        mapped_tiers = np.array([rank_map.get(int(c), 0) for c in raw_labels], dtype=float)
        tier_norm = mapped_tiers / max(1.0, float(self.n_clusters - 1))

        # Use top 8 primary features for distance metric
        sub_dim = min(8, X_scaled.shape[1])
        X_sub = X_scaled[:, :sub_dim]
        c_fast = ordered_centroids[0, :sub_dim]
        c_obsolete = ordered_centroids[-1, :sub_dim]

        dist_fast = np.linalg.norm(X_sub - c_fast, axis=1)
        dist_obsolete = np.linalg.norm(X_sub - c_obsolete, axis=1)
        relative_affinity = dist_fast / (dist_fast + dist_obsolete + 1e-6)

        continuous_score = 0.60 * tier_norm + 0.40 * relative_affinity
        return np.clip(continuous_score, 0.0, 1.0)

    def train_all(self, X_train_scaled: np.ndarray):
        """Trains all 5 Unsupervised Machine Learning models."""
        print("  -> Training Model 1/5: Mini-Batch K-Means (Fast Stochastic Partitioning)...")
        mbk = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=2048, random_state=self.random_state)
        mbk_labels = mbk.fit_predict(X_train_scaled)
        self._register_cluster_model("Mini-Batch K-Means", mbk, X_train_scaled, mbk_labels)

        print("  -> Training Model 2/5: Gaussian Mixture Model (EM Density Estimation)...")
        gmm = GaussianMixture(n_components=self.n_clusters, covariance_type='diag', max_iter=50, random_state=self.random_state)
        gmm_labels = gmm.fit_predict(X_train_scaled)
        self._register_cluster_model("Gaussian Mixture Model", gmm, X_train_scaled, gmm_labels)

        print("  -> Training Model 3/5: K-Means (Centroid-Based Partitioning)...")
        km = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=3, random_state=self.random_state)
        km_labels = km.fit_predict(X_train_scaled)
        self._register_cluster_model("K-Means", km, X_train_scaled, km_labels)

        print("  -> Training Model 4/5: Bisecting K-Means (Hierarchical Divisive Partitioning)...")
        bkm = BisectingKMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        bkm_labels = bkm.fit_predict(X_train_scaled)
        self._register_cluster_model("Bisecting K-Means", bkm, X_train_scaled, bkm_labels)

        print("  -> Training Model 5/5: BIRCH (Balanced Iterative Reducing and Clustering)...")
        # Train Birch on representative sample for high-speed tree construction, then predict
        birch_sample_size = min(25000, len(X_train_scaled))
        birch = Birch(n_clusters=self.n_clusters, threshold=1.0)
        birch.fit(X_train_scaled[:birch_sample_size])
        birch_labels = birch.predict(X_train_scaled)
        self._register_cluster_model("BIRCH", birch, X_train_scaled, birch_labels)

    def predict_all(self, X_test_scaled: np.ndarray) -> dict:
        """Evaluates predictions, continuous scores, and silhouette across all 5 models."""
        results = {}
        for name, model in self.models.items():
            meta = self.cluster_meta[name]
            rank_map = meta['rank_map']
            ordered_centroids = meta['ordered_centroids']
            threshold = meta['threshold']

            if hasattr(model, 'predict'):
                raw_labels = model.predict(X_test_scaled)
            else:
                raw_labels = model.fit_predict(X_test_scaled)

            scores = self._compute_continuous_score(X_test_scaled, raw_labels, rank_map, ordered_centroids)
            preds = (scores >= threshold).astype(int)

            results[name] = {
                'preds': preds,
                'scores': scores,
                'silhouette': meta['train_silhouette'],
                'threshold': threshold
            }
        return results


class SingleModelInventorySegmenter:
    """
    Dedicated production wrapper for the selected #1 Champion Model.
    Produces cluster tiers, calibrated obsolescence probabilities, and actionable classifications.
    """

    def __init__(self, model_name: str, n_clusters: int = 4, contamination: float = 0.35, random_state: int = 42):
        self.model_name = model_name
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.rank_map = {}
        self.ordered_centroids = None
        self.threshold = 0.50

    def fit(self, X_train_scaled: np.ndarray):
        """Fits the single selected Champion Model."""
        if self.model_name == "Mini-Batch K-Means":
            self.model = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=2048, random_state=self.random_state)
        elif self.model_name == "Gaussian Mixture Model":
            self.model = GaussianMixture(n_components=self.n_clusters, covariance_type='diag', max_iter=50, random_state=self.random_state)
        elif self.model_name == "K-Means":
            self.model = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=3, random_state=self.random_state)
        elif self.model_name == "Bisecting K-Means":
            self.model = BisectingKMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        elif self.model_name == "BIRCH":
            self.model = Birch(n_clusters=self.n_clusters, threshold=1.0)
        else:
            self.model = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=self.random_state)

        if self.model_name == "BIRCH":
            birch_sample = min(25000, len(X_train_scaled))
            self.model.fit(X_train_scaled[:birch_sample])
            raw_labels = self.model.predict(X_train_scaled)
        elif hasattr(self.model, 'fit_predict'):
            raw_labels = self.model.fit_predict(X_train_scaled)
        else:
            self.model.fit(X_train_scaled)
            raw_labels = self.model.predict(X_train_scaled)

        centroids = np.zeros((self.n_clusters, X_train_scaled.shape[1]))
        for c in range(self.n_clusters):
            mask = (raw_labels == c)
            if np.any(mask):
                centroids[c] = X_train_scaled[mask].mean(axis=0)

        # Unsupervised Obsolescence Signal (col 25 = obsolescence_risk_index, col 8 = inventory_turnover_ratio)
        cluster_risk_signal = centroids[:, 25] - centroids[:, 8]
        sorted_cluster_ids = np.argsort(cluster_risk_signal)
        self.rank_map = {int(orig_c): int(new_tier) for new_tier, orig_c in enumerate(sorted_cluster_ids)}
        self.ordered_centroids = centroids[sorted_cluster_ids]

        train_scores = self._score(X_train_scaled, raw_labels)
        self.threshold = float(np.percentile(train_scores, 100.0 * (1.0 - self.contamination)))

    def _score(self, X_scaled: np.ndarray, raw_labels: np.ndarray) -> np.ndarray:
        mapped_tiers = np.array([self.rank_map.get(int(c), 0) for c in raw_labels], dtype=float)
        tier_norm = mapped_tiers / max(1.0, float(self.n_clusters - 1))

        sub_dim = min(8, X_scaled.shape[1])
        X_sub = X_scaled[:, :sub_dim]
        c_fast = self.ordered_centroids[0, :sub_dim]
        c_obsolete = self.ordered_centroids[-1, :sub_dim]

        dist_fast = np.linalg.norm(X_sub - c_fast, axis=1)
        dist_obsolete = np.linalg.norm(X_sub - c_obsolete, axis=1)
        relative_affinity = dist_fast / (dist_fast + dist_obsolete + 1e-6)

        return np.clip(0.60 * tier_norm + 0.40 * relative_affinity, 0.0, 1.0)

    def predict_clusters(self, X_scaled: np.ndarray):
        raw_labels = self.model.predict(X_scaled) if hasattr(self.model, 'predict') else self.model.fit_predict(X_scaled)
        ordered_tiers = np.array([self.rank_map.get(int(c), 0) for c in raw_labels], dtype=int)
        tier_names = [INVENTORY_TIER_NAMES.get(t, f"Tier {t}") for t in ordered_tiers]
        return ordered_tiers, tier_names

    def predict_proba(self, X_scaled: np.ndarray):
        raw_labels = self.model.predict(X_scaled) if hasattr(self.model, 'predict') else self.model.fit_predict(X_scaled)
        scores = self._score(X_scaled, raw_labels)
        return scores, scores

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        scores, _ = self.predict_proba(X_scaled)
        return (scores >= self.threshold).astype(int)

    def assign_risk_category(self, scores: np.ndarray) -> list:
        categories = []
        for s in scores:
            if s >= 0.75:
                categories.append("Critical Obsolete Risk")
            elif s >= 0.50:
                categories.append("High Slow Risk")
            elif s >= 0.25:
                categories.append("Moderate Slow Risk")
            else:
                categories.append("Low Obsolescence Risk")
        return categories
