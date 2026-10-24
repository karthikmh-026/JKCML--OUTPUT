import time
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, BisectingKMeans, Birch
from sklearn.mixture import GaussianMixture

# Generate synthetic 80,000 x 32 feature matrix to test training speed
np.random.seed(42)
X_train_scaled = np.random.randn(81437, 32)

print("Testing clustering model speeds on 81,437 rows...")

t0 = time.time()
km = KMeans(n_clusters=4, init='k-means++', n_init=5, max_iter=150, random_state=42)
km.fit(X_train_scaled[:, :5])
print(f"KMeans: {time.time() - t0:.2f}s")

t0 = time.time()
gmm = GaussianMixture(n_components=4, covariance_type='diag', n_init=3, max_iter=100, random_state=42)
gmm.fit(X_train_scaled[:, :12])
print(f"GMM: {time.time() - t0:.2f}s")

t0 = time.time()
mbk = MiniBatchKMeans(n_clusters=4, batch_size=1024, n_init=3, max_iter=100, random_state=42)
mbk.fit(X_train_scaled[:, :10])
print(f"MiniBatchKMeans: {time.time() - t0:.2f}s")

t0 = time.time()
bkm = BisectingKMeans(n_clusters=4, init='k-means++', n_init=3, random_state=42)
bkm.fit(X_train_scaled[:, :8])
print(f"BisectingKMeans: {time.time() - t0:.2f}s")

t0 = time.time()
birch = Birch(n_clusters=4, threshold=0.5, branching_factor=50)
birch.fit(X_train_scaled[:, :9])
print(f"Birch: {time.time() - t0:.2f}s")
