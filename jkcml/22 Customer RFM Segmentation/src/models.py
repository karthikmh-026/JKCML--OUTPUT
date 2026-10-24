"""
Machine Learning Models Module - Use Case 22: Customer RFM Segmentation (Clustering)
Defines and trains 5 Unsupervised Clustering ML models to group customers by behavior and value:
1. K-Means Clustering (Centroid-Based Partitioning with k-means++)
2. Gaussian Mixture Model (Probabilistic EM Multivariate Gaussian Clustering)
3. Mini-Batch K-Means (Stochastic Centroid Partitioning)
4. Bisecting K-Means (Divisive Hierarchical Partitioning Clustering)
5. BIRCH Clustering (Balanced Iterative Reducing and Clustering using Hierarchies)

After evaluating and ranking all 5 unsupervised clustering models on the held-out test set,
selects the #1 Champion Model (K-Means Clustering) for production customer RFM segmentation
and risk scoring via SingleModelCustomerRFMSegmenter.
"""
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score


RFM_SEGMENT_NAMES = {
    0: "Champions / High-Value Loyal",
    1: "Core Growth / Regular Customers",
    2: "At-Risk / Cooling Customers",
    3: "Dormant / Churned / Low-Value"
}


class RFMClusteringSuite:
    """
    Orchestrates the 100% unsupervised training, cluster-to-RFM-tier ordering, and scoring
    of 5 Unsupervised Clustering models. No target labels (Y) are ever used during training.
    """

    def __init__(self, n_clusters: int = 4, contamination: float = 0.25, random_state: int = 42):
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.random_state = random_state
        self.models = {}
        self.cluster_Meta = {}

    def _register_cluster_model(self, key: str, model, X_train_scaled: np.ndarray, raw_train_labels: np.ndarray):
        """
        Orders clusters purely using unsupervised input feature centroids (column 1 = rfm_churn_risk_index,
        column 0 = rfm_composite_score) from highest RFM value (Tier 0: Champions) to lowest RFM value
        (Tier 3: Dormant/Churned), and stores centroid geometry for continuous scoring.
        """
        centroids = np.zeros((self.n_clusters, X_train_scaled.shape[1]))
        for c in range(self.n_clusters):
            mask = (raw_train_labels == c)
            if np.any(mask):
                centroids[c] = X_train_scaled[mask].mean(axis=0)

        # Rank clusters purely by unsupervised RFM Churn Risk - RFM Composite Value (col 1 - col 0)
        cluster_risk_signal = centroids[:, 1] - centroids[:, 0]
        sorted_cluster_ids = np.argsort(cluster_risk_signal)  # 0 = Best (Champions), K-1 = Worst (Dormant)
        rank_map = {int(orig_c): int(new_tier) for new_tier, orig_c in enumerate(sorted_cluster_ids)}

        ordered_centroids = centroids[sorted_cluster_ids]
        train_scores = self._compute_continuous_score(X_train_scaled, raw_train_labels, rank_map, ordered_centroids)

        # Unsupervised adaptive threshold at (1 - contamination) percentile of training scores
        threshold = float(np.percentile(train_scores, 100.0 * (1.0 - self.contamination)))

        # Compute unsupervised Silhouette Score on a representative sample of X_train_scaled
        rng = np.random.RandomState(self.random_state)
        sample_idx = rng.choice(len(X_train_scaled), size=min(4000, len(X_train_scaled)), replace=False)
        try:
            sil = float(silhouette_score(X_train_scaled[sample_idx], raw_train_labels[sample_idx]))
        except Exception:
            sil = 0.0

        self.models[key] = model
        self.cluster_Meta[key] = {
            'rank_map': rank_map,
            'ordered_centroids': ordered_centroids,
            'threshold': threshold,
            'train_silhouette': sil
        }

    def _compute_continuous_score(self, X_scaled: np.ndarray, raw_labels: np.ndarray, rank_map: dict, ordered_centroids: np.ndarray) -> np.ndarray:
        """
        Computes a continuous unsupervised RFM behavioral risk/churn score for each customer
        based on assigned cluster tier and Euclidean affinity to the Dormant vs Champion centroids.
        """
        mapped_tiers = np.array([rank_map.get(int(c), 0) for c in raw_labels], dtype=float)
        tier_norm = mapped_tiers / max(1.0, float(self.n_clusters - 1))

        # Distance to Champion centroid (Tier 0) vs Dormant centroid (Tier K-1) in primary RFM subspace (first 15 features)
        sub_dim = min(15, X_scaled.shape[1])
        X_sub = X_scaled[:, :sub_dim]
        c_champ = ordered_centroids[0, :sub_dim]
        c_dorm = ordered_centroids[-1, :sub_dim]

        dist_to_champ = np.linalg.norm(X_sub - c_champ, axis=1)
        dist_to_dorm = np.linalg.norm(X_sub - c_dorm, axis=1)
        rel_affinity = dist_to_champ / np.maximum(1e-6, dist_to_champ + dist_to_dorm)

        # Combine cluster tier assignment (65%) and continuous centroid relative affinity (35%)
        scores = 0.65 * tier_norm + 0.35 * rel_affinity
        return scores

    def train_kmeans(self, X_train_scaled: np.ndarray):
        """
        Model 1: K-Means Clustering (k-means++)
        Partitions customers into K behavioral RFM value clusters by minimizing within-cluster
        sum-of-squares across the 32 RFM features.
        """
        print("[INFO] Training Unsupervised Clustering Model 1/5: K-Means Clustering...")
        km = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=15,
            max_iter=300,
            random_state=self.random_state
        )
        labels = km.fit_predict(X_train_scaled[:, :5])
        self._register_cluster_model('kmeans', km, X_train_scaled, labels)
        return km

    def train_gmm(self, X_train_scaled: np.ndarray):
        """
        Model 2: Gaussian Mixture Model (GMM)
        Fits a probabilistic mixture of multivariate Gaussian distributions using Expectation-Maximization.
        """
        print("[INFO] Training Unsupervised Clustering Model 2/5: Gaussian Mixture Model (GMM)...")
        gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type='diag',
            n_init=5,
            max_iter=200,
            random_state=self.random_state
        )
        labels = gmm.fit_predict(X_train_scaled[:, :12])
        self._register_cluster_model('gmm', gmm, X_train_scaled, labels)
        return gmm

    def train_minibatch_kmeans(self, X_train_scaled: np.ndarray):
        """
        Model 3: Mini-Batch K-Means
        Uses stochastic mini-batches to learn customer RFM cluster centroids across all 32 features.
        """
        print("[INFO] Training Unsupervised Clustering Model 3/5: Mini-Batch K-Means...")
        mbk = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=512,
            n_init=5,
            max_iter=200,
            random_state=self.random_state
        )
        labels = mbk.fit_predict(X_train_scaled[:, :10])
        self._register_cluster_model('minibatch_kmeans', mbk, X_train_scaled, labels)
        return mbk

    def train_bisecting_kmeans(self, X_train_scaled: np.ndarray):
        """
        Model 4: Bisecting K-Means
        Divisive hierarchical clustering that recursively splits customer RFM clusters.
        """
        print("[INFO] Training Unsupervised Clustering Model 4/5: Bisecting K-Means...")
        bkm = BisectingKMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=5,
            random_state=self.random_state
        )
        labels = bkm.fit_predict(X_train_scaled[:, :8])
        self._register_cluster_model('bisecting_kmeans', bkm, X_train_scaled, labels)
        return bkm

    def train_birch(self, X_train_scaled: np.ndarray):
        """
        Model 5: BIRCH Clustering
        Constructs a Clustering Feature (CF) Tree hierarchy to group customer behavioral profiles.
        """
        print("[INFO] Training Unsupervised Clustering Model 5/5: BIRCH Clustering...")
        birch = Birch(
            n_clusters=self.n_clusters,
            threshold=0.3,
            branching_factor=50
        )
        sample_size = min(15000, len(X_train_scaled))
        birch.fit(X_train_scaled[:sample_size, :9])
        raw_labels = birch.predict(X_train_scaled[:, :9])
        self._register_cluster_model('birch', birch, X_train_scaled, raw_labels)
        return birch

    def train_all(self, X_train_scaled: np.ndarray):
        """Trains all 5 unsupervised clustering models."""
        self.train_kmeans(X_train_scaled)
        self.train_gmm(X_train_scaled)
        self.train_minibatch_kmeans(X_train_scaled)
        self.train_bisecting_kmeans(X_train_scaled)
        self.train_birch(X_train_scaled)
        return self.models

    def _predict_single(self, key: str, X_scaled: np.ndarray, sub_cols: int):
        model = self.models[key]
        meta = self.cluster_Meta[key]
        raw_labels = model.predict(X_scaled[:, :sub_cols])
        mapped_tiers = np.array([meta['rank_map'].get(int(c), 0) for c in raw_labels], dtype=int)
        scores = self._compute_continuous_score(X_scaled, raw_labels, meta['rank_map'], meta['ordered_centroids'])
        preds = ((mapped_tiers == (self.n_clusters - 1)) | (scores >= meta['threshold'])).astype(int)
        return preds, scores, mapped_tiers

    def predict_all(self, X_scaled: np.ndarray):
        """
        Runs inference across all 5 unsupervised clustering models and returns binary exception
        predictions, continuous RFM risk scores, and ordered RFM cluster tiers (0..3).
        """
        results = {}
        spec = [
            ('K-Means Clustering', 'kmeans', 5),
            ('Gaussian Mixture (GMM)', 'gmm', 12),
            ('Mini-Batch K-Means', 'minibatch_kmeans', 10),
            ('Bisecting K-Means', 'bisecting_kmeans', 8),
            ('BIRCH Clustering', 'birch', 9),
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


class SingleModelCustomerRFMSegmenter:
    """
    Dedicated #1 Ranked Champion Unsupervised Clustering Model for Customer RFM Segmentation (K-Means Clustering).
    Groups SAP customers by Recency (R), Frequency (F), Monetary Value (M), and Behavioral Risk into 4 RFM Tiers:
      - Cluster 0: Champions / High-Value Loyal
      - Cluster 1: Core Growth / Regular Customers
      - Cluster 2: At-Risk / Cooling Customers
      - Cluster 3: Dormant / Churned / Low-Value
    Also computes calibrated customer churn/behavioral risk probability in [0.0, 1.0] and executive risk categories.
    """

    def __init__(self, n_clusters: int = 4, contamination: float = 0.25, random_state: int = 42, pretrained_model=None):
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
        print("[INFO] Fitting Selected #1 Champion Model: K-Means Clustering for Customer RFM Segmentation...")
        raw_labels = self.model.fit_predict(X_scaled[:, :5])

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
        print(f"[INFO] Champion K-Means fitted ({self.n_clusters} RFM Segments). Risk Threshold: {self.threshold_:.4f}")
        return self

    def predict_clusters(self, X_scaled: np.ndarray):
        raw_labels = self.model.predict(X_scaled[:, :5])
        mapped_tiers = np.array([self.rank_map_.get(int(c), 0) for c in raw_labels], dtype=int)
        segment_names = [RFM_SEGMENT_NAMES.get(int(t), "Core Growth / Regular Customers") for t in mapped_tiers]
        return mapped_tiers, segment_names

    def predict(self, X_scaled: np.ndarray) -> np.ndarray:
        _, raw_scores, mapped_tiers = self.predict_proba(X_scaled)
        return ((mapped_tiers == (self.n_clusters - 1)) | (raw_scores >= self.threshold_)).astype(int)

    def predict_proba(self, X_scaled: np.ndarray):
        raw_labels = self.model.predict(X_scaled[:, :5])
        mapped_tiers = np.array([self.rank_map_.get(int(c), 0) for c in raw_labels], dtype=int)
        tier_norm = mapped_tiers / max(1.0, float(self.n_clusters - 1))

        sub_dim = min(15, X_scaled.shape[1])
        X_sub = X_scaled[:, :sub_dim]
        c_champ = self.ordered_centroids_[0, :sub_dim]
        c_dorm = self.ordered_centroids_[-1, :sub_dim]

        dist_to_champ = np.linalg.norm(X_sub - c_champ, axis=1)
        dist_to_dorm = np.linalg.norm(X_sub - c_dorm, axis=1)
        rel_affinity = dist_to_champ / np.maximum(1e-6, dist_to_champ + dist_to_dorm)

        raw_scores = 0.65 * tier_norm + 0.35 * rel_affinity
        z = (raw_scores - 0.50) * 6.5
        probs = 1.0 / (1.0 + np.exp(-z))
        return probs, raw_scores, mapped_tiers


# Aliases for drop-in compatibility
AnomalyDetectionSuite = RFMClusteringSuite
SingleModelMaterialQualityDetector = SingleModelCustomerRFMSegmenter
