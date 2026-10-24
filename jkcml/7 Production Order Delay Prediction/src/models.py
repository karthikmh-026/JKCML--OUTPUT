"""
Machine Learning Models Module - Use Case 7: Production Order Delay Prediction
Defines and trains 5 Unsupervised Machine Learning Models to predict production order delays
and estimate calibrated Probability of Delay P(Delay) strictly from input features:
1. K-Means Clustering (Centroid-Based Partitioning with k-means++)
2. Gaussian Mixture Model (Probabilistic EM Density Estimation)
3. Mini-Batch K-Means (Stochastic Scalable Centroid Partitioning)
4. Bisecting K-Means (Divisive Hierarchical Partitioning)
5. PCA Latent Space Clustering (8-Component PCA Latent Space Partitioning)

After evaluating and ranking all 5 unsupervised models, selects the #1 Ranked Champion Model
for production delay prediction and shop floor operational scoring.
"""
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


DELAY_TIER_NAMES = {
    0: "Tier 0: On-Time / Expedited Schedule",
    1: "Tier 1: Low Delay Risk / Minor Variance",
    2: "Tier 2: Moderate Delay Risk / Bottlenecked",
    3: "Tier 3: Critical Delay Exception / Past Due"
}

DELAY_ACTION_MAP = {
    0: "Standard Monitoring - Order proceeding on planned schedule",
    1: "Proactive Tracking - Monitor operation confirm progress and material staging",
    2: "Expedite Required - Shift capacity reallocation and prioritize work center operations",
    3: "Immediate Intervention - Management escalation, emergency staging, and schedule overhaul"
}


class PCALatentClusterer:
    """Combines PCA dimensionality reduction with K-Means clustering for latent space partitioning."""
    def __init__(self, n_clusters: int = 4, n_components: int = 8, random_state: int = 42):
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=5, random_state=random_state)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X_pca = self.pca.fit_transform(X)
        return self.kmeans.fit_predict(X_pca)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_pca = self.pca.transform(X)
        return self.kmeans.predict(X_pca)


class ProductionOrderDelayClusteringSuite:
    """
    Orchestrates the 100% unsupervised training, cluster-to-tier alignment,
    and delay probability scoring of 5 Unsupervised Clustering ML models.
    No target labels (Y) are ever used during model training.
    """

    def __init__(self, n_clusters: int = 4, contamination: float = 0.40, random_state: int = 42):
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.random_state = random_state
        self.models = {}
        self.cluster_meta = {}

    def _register_cluster_model(self, key: str, model, X_train_scaled: np.ndarray, raw_train_labels: np.ndarray, signal_col: int = 29):
        """
        Orders clusters purely using unsupervised input feature centroids
        (col 29 = composite_delay_signal) from lowest risk (Tier 0: On-Time)
        to highest delay risk (Tier 3: Critical Delay).
        """
        centroids = np.zeros((self.n_clusters, X_train_scaled.shape[1]))
        for c in range(self.n_clusters):
            mask = (raw_train_labels == c)
            if np.any(mask):
                centroids[c] = X_train_scaled[mask].mean(axis=0)

        # Rank clusters purely by unsupervised delay signal
        cluster_risk_signal = centroids[:, signal_col]
        sorted_cluster_ids = np.argsort(cluster_risk_signal)
        rank_map = {int(orig_c): int(new_tier) for new_tier, orig_c in enumerate(sorted_cluster_ids)}

        ordered_centroids = centroids[sorted_cluster_ids]
        threshold = 0.50

        # Compute unsupervised Silhouette Score on a sample
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
        Computes a continuous unsupervised Delay Probability score for each production order
        based on assigned cluster tier and Euclidean affinity to Critical Delay vs On-Time centroids.
        """
        mapped_tiers = np.array([rank_map.get(int(c), 0) for c in raw_labels], dtype=float)
        tier_norm = mapped_tiers / max(1.0, float(self.n_clusters - 1))

        # Use top 8 primary features for distance metric
        sub_dim = min(8, X_scaled.shape[1])
        X_sub = X_scaled[:, :sub_dim]
        c_ontime = ordered_centroids[0, :sub_dim]
        c_delayed = ordered_centroids[-1, :sub_dim]

        dist_ontime = np.linalg.norm(X_sub - c_ontime, axis=1)
        dist_delayed = np.linalg.norm(X_sub - c_delayed, axis=1)
        denom = dist_ontime + dist_delayed + 1e-6
        distance_risk_ratio = dist_ontime / denom  # Close to delayed -> ~1.0; close to ontime -> ~0.0

        continuous_score = (0.60 * tier_norm) + (0.40 * distance_risk_ratio)
        return np.clip(continuous_score, 0.0, 1.0)

    def train_all(self, X_train_scaled: np.ndarray):
        """Trains all 5 Unsupervised Machine Learning Models on scaled input features."""
        # 1. K-Means Clustering
        print("  -> Training Model 1/5: K-Means Clustering (Centroid-Based Partitioning)...")
        km = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=5, random_state=self.random_state)
        km_labels = km.fit_predict(X_train_scaled)
        self._register_cluster_model('K-Means Clustering', km, X_train_scaled, km_labels)

        # 2. Gaussian Mixture Model (GMM)
        print("  -> Training Model 2/5: Gaussian Mixture Model (Probabilistic EM Density Estimation)...")
        gmm = GaussianMixture(n_components=self.n_clusters, covariance_type='diag', random_state=self.random_state, max_iter=30)
        gmm_labels = gmm.fit_predict(X_train_scaled)
        self._register_cluster_model('Gaussian Mixture Model', gmm, X_train_scaled, gmm_labels)

        # 3. Mini-Batch K-Means
        print("  -> Training Model 3/5: Mini-Batch K-Means (Stochastic Scalable Centroid Partitioning)...")
        mbk = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=4096, n_init=5, random_state=self.random_state)
        mbk_labels = mbk.fit_predict(X_train_scaled)
        self._register_cluster_model('Mini-Batch K-Means', mbk, X_train_scaled, mbk_labels)

        # 4. Bisecting K-Means
        print("  -> Training Model 4/5: Bisecting K-Means (Divisive Hierarchical Partitioning)...")
        bkm = BisectingKMeans(n_clusters=self.n_clusters, init='k-means++', random_state=self.random_state)
        bkm_labels = bkm.fit_predict(X_train_scaled)
        self._register_cluster_model('Bisecting K-Means', bkm, X_train_scaled, bkm_labels)

        # 5. PCA Latent Space Clustering
        print("  -> Training Model 5/5: PCA Latent Space Clustering (8-Component Latent Partitioning)...")
        pca_km = PCALatentClusterer(n_clusters=self.n_clusters, n_components=8, random_state=self.random_state)
        pca_km_labels = pca_km.fit_predict(X_train_scaled)
        self._register_cluster_model('PCA Latent Clustering', pca_km, X_train_scaled, pca_km_labels)

    def predict_single(self, key: str, X_scaled: np.ndarray) -> dict:
        """Scores given data using one of the 5 trained unsupervised models."""
        model = self.models[key]
        meta = self.cluster_meta[key]
        rank_map = meta['rank_map']
        ordered_centroids = meta['ordered_centroids']
        threshold = meta['threshold']

        raw_labels = model.predict(X_scaled)
        scores = self._compute_continuous_score(X_scaled, raw_labels, rank_map, ordered_centroids)
        mapped_tiers = np.array([rank_map.get(int(c), 0) for c in raw_labels], dtype=int)
        binary_preds = ((scores >= threshold) | (mapped_tiers >= 2)).astype(int)

        return {
            'raw_labels': raw_labels,
            'mapped_tiers': mapped_tiers,
            'scores': scores,
            'preds': binary_preds,
            'threshold': threshold,
            'silhouette': meta['train_silhouette']
        }

    def predict_all(self, X_test_scaled: np.ndarray) -> dict:
        """Returns predictions, probabilities, and metadata for all 5 models."""
        results = {}
        for key in self.models.keys():
            results[key] = self.predict_single(key, X_test_scaled)
        return results


class SingleModelProductionOrderSegmenter:
    """
    Dedicated wrapper for the selected #1 Champion Model.
    Provides inference, probability estimation, and risk tier segmentation for production deployment.
    """

    def __init__(self, model_name: str, n_clusters: int = 4, contamination: float = 0.40, random_state: int = 42):
        self.model_name = model_name
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.rank_map = None
        self.ordered_centroids = None
        self.threshold = 0.50

    def fit(self, X_train_scaled: np.ndarray, signal_col: int = 29):
        """Fits the single selected champion unsupervised model."""
        if 'Gaussian' in self.model_name:
            self.model = GaussianMixture(n_components=self.n_clusters, covariance_type='diag', random_state=self.random_state, max_iter=30)
        elif 'Mini-Batch' in self.model_name:
            self.model = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=4096, n_init=5, random_state=self.random_state)
        elif 'Bisecting' in self.model_name:
            self.model = BisectingKMeans(n_clusters=self.n_clusters, init='k-means++', random_state=self.random_state)
        elif 'PCA' in self.model_name:
            self.model = PCALatentClusterer(n_clusters=self.n_clusters, n_components=8, random_state=self.random_state)
        else:
            self.model = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=5, random_state=self.random_state)

        raw_train_labels = self.model.fit_predict(X_train_scaled)

        # Compute centroids
        centroids = np.zeros((self.n_clusters, X_train_scaled.shape[1]))
        for c in range(self.n_clusters):
            mask = (raw_train_labels == c)
            if np.any(mask):
                centroids[c] = X_train_scaled[mask].mean(axis=0)

        # Order clusters by unsupervised delay risk
        cluster_risk_signal = centroids[:, signal_col]
        sorted_cluster_ids = np.argsort(cluster_risk_signal)
        self.rank_map = {int(orig_c): int(new_tier) for new_tier, orig_c in enumerate(sorted_cluster_ids)}
        self.ordered_centroids = centroids[sorted_cluster_ids]
        self.threshold = 0.50

    def _score(self, X_scaled: np.ndarray, raw_labels: np.ndarray) -> np.ndarray:
        mapped_tiers = np.array([self.rank_map.get(int(c), 0) for c in raw_labels], dtype=float)
        tier_norm = mapped_tiers / max(1.0, float(self.n_clusters - 1))

        sub_dim = min(8, X_scaled.shape[1])
        X_sub = X_scaled[:, :sub_dim]
        c_ontime = self.ordered_centroids[0, :sub_dim]
        c_delayed = self.ordered_centroids[-1, :sub_dim]

        dist_ontime = np.linalg.norm(X_sub - c_ontime, axis=1)
        dist_delayed = np.linalg.norm(X_sub - c_delayed, axis=1)
        denom = dist_ontime + dist_delayed + 1e-6
        distance_risk_ratio = dist_ontime / denom

        continuous_score = (0.60 * tier_norm) + (0.40 * distance_risk_ratio)
        return np.clip(continuous_score, 0.0, 1.0)

    def predict(self, X_scaled: np.ndarray) -> dict:
        """Inference for new production order records."""
        raw_labels = self.model.predict(X_scaled)
        scores = self._score(X_scaled, raw_labels)
        mapped_tiers = np.array([self.rank_map.get(int(c), 0) for c in raw_labels], dtype=int)
        binary_preds = ((scores >= self.threshold) | (mapped_tiers >= 2)).astype(int)

        return {
            'mapped_tiers': mapped_tiers,
            'scores': scores,
            'preds': binary_preds
        }
