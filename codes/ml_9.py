
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.datasets import make_moons



# ------------------------------------------------------------
# 1. LOAD OR GENERATE THE DATASET
# ------------------------------------------------------------
CSV = "two_moons_dataset.csv"

if os.path.exists(CSV):
    df = pd.read_csv(CSV)
else:
    # Generate two moons and additional random noise points
    rng = np.random.RandomState(42)

    Xm, ym = make_moons(
        n_samples=1000,
        noise=0.06,
        random_state=42
    )

    Xn = rng.uniform(
        [-1.5, -1.0],
        [2.5, 1.5],
        size=(100, 2)
    )

    df = pd.DataFrame(
        np.vstack([Xm, Xn]),
        columns=["x", "y"]
    )

    df["shape"] = np.concatenate([
        ym,
        -np.ones(100, dtype=int)
    ])

# Extract features and true labels
X = df[["x", "y"]].values
y_true = df["shape"].values

print("=" * 60)
print("        DBSCAN CLUSTERING - TWO MOONS DATASET")
print("=" * 60)

print(f"Total Samples      : {X.shape[0]}")
print(f"Total Dimensions   : {X.shape[1]}")
print(f"True noise points  : {(y_true == -1).sum()}")

# ------------------------------------------------------------
# 2. FEATURE SCALING
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------
# 3. SELECT MinPts AND ESTIMATE EPS USING K-DISTANCE GRAPH
# ------------------------------------------------------------
MIN_PTS = 2 * X.shape[1] + 1
k = MIN_PTS

# Find the distance to each point's k-th nearest neighbour
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X_scaled)

distances, _ = nn.kneighbors(X_scaled)

# Sort the k-th neighbour distances
k_dist = np.sort(distances[:, -1])

# Estimate the elbow using maximum distance from the endpoint line
n = len(k_dist)
pts = np.column_stack([
    np.arange(n),
    k_dist
])

line = pts[-1] - pts[0]
line_norm = np.linalg.norm(line)

if line_norm > 0:
    line = line / line_norm
    vec = pts - pts[0]

    dist_to_line = np.abs(
        vec[:, 0] * line[1]
        - vec[:, 1] * line[0]
    )

    knee_idx = int(np.argmax(dist_to_line))
    EPS = round(float(k_dist[knee_idx]), 3)
else:
    knee_idx = 0
    EPS = round(float(k_dist[0]), 3)

print(f"\nMinPts (min_samples): {MIN_PTS}")
print(f"Elbow index          : {knee_idx}")
print(f"Estimated epsilon    : {EPS}")

# Plot the k-distance graph
plt.figure(figsize=(8, 5))

plt.plot(
    k_dist,
    color="steelblue",
    linewidth=2,
    label=f"{k}-distance"
)

plt.axhline(
    EPS,
    color="red",
    linestyle="--",
    label=f"eps = {EPS}"
)

plt.scatter(
    knee_idx,
    k_dist[knee_idx],
    color="red",
    s=80,
    zorder=5,
    label="Elbow"
)

plt.title("K-Distance Graph (Choosing Epsilon)")
plt.xlabel("Points Sorted by Distance")
plt.ylabel(f"Distance to {k}-th Nearest Neighbour")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig("k_distance_graph.png", dpi=150)
plt.show()

# ------------------------------------------------------------
# 4. APPLY DBSCAN CLUSTERING
# ------------------------------------------------------------
db = DBSCAN(
    eps=EPS,
    min_samples=MIN_PTS
)

db.fit(X_scaled)
labels = db.labels_

# Count clusters and noise points
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = int((labels == -1).sum())

# Identify core, border, and noise points
core_mask = np.zeros_like(labels, dtype=bool)
core_mask[db.core_sample_indices_] = True

border_mask = (~core_mask) & (labels != -1)
noise_mask = labels == -1

# ------------------------------------------------------------
# 5. DISPLAY DBSCAN RESULTS
# ------------------------------------------------------------
print("\n" + "-" * 60)
print("RESULTS")
print("-" * 60)

print(f"Estimated number of clusters : {n_clusters}")
print(f"Core points                  : {core_mask.sum()}")
print(f"Border points                : {border_mask.sum()}")
print(f"Noise points (label = -1)    : {n_noise}")

for c in sorted(set(labels) - {-1}):
    print(
        f"  Cluster {c} size            : "
        f"{(labels == c).sum()}"
    )

# ------------------------------------------------------------
# 6. EVALUATE DBSCAN
# ------------------------------------------------------------
ari_db = adjusted_rand_score(y_true, labels)

# Silhouette score excludes noise points
valid = labels != -1
valid_labels = labels[valid]
valid_clusters = np.unique(valid_labels)

if len(valid_clusters) > 1 and valid.sum() > len(valid_clusters):
    sil_db = silhouette_score(
        X_scaled[valid],
        valid_labels
    )
else:
    sil_db = float("nan")

print(f"\nAdjusted Rand Index (DBSCAN) : {ari_db:.4f}")
print(f"Silhouette Score (excl noise): {sil_db:.4f}")

# ------------------------------------------------------------
# 7. COMPARE WITH K-MEANS (K = 2)
# ------------------------------------------------------------
km = KMeans(
    n_clusters=2,
    n_init=10,
    random_state=42
)

km_labels = km.fit_predict(X_scaled)
ari_km = adjusted_rand_score(y_true, km_labels)

print(f"Adjusted Rand Index (KMeans) : {ari_km:.4f}")

# ------------------------------------------------------------
# 8. EPS PARAMETER SENSITIVITY ANALYSIS
# ------------------------------------------------------------
print(f"\nParameter sensitivity (MinPts = {MIN_PTS}):")
print(f"{'eps':>8} {'clusters':>10} {'noise':>8}")

eps_values = [
    EPS * 0.5,
    EPS * 0.75,
    EPS,
    EPS * 1.5,
    EPS * 2.5
]

for e in eps_values:
    test_labels = DBSCAN(
        eps=e,
        min_samples=MIN_PTS
    ).fit_predict(X_scaled)

    test_clusters = (
        len(set(test_labels))
        - (1 if -1 in test_labels else 0)
    )

    test_noise = int((test_labels == -1).sum())

    print(f"{e:8.3f} {test_clusters:10d} {test_noise:8d}")

# ------------------------------------------------------------
# 9. VISUALIZE GROUND TRUTH, K-MEANS, AND DBSCAN
# ------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))

colors = plt.cm.Set1(
    np.linspace(0, 1, max(n_clusters, 1))
)

# (a) Ground truth
ax[0].scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=y_true,
    cmap="viridis",
    s=12
)

ax[0].set_title("Ground Truth")

# (b) K-Means
ax[1].scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=km_labels,
    cmap="coolwarm",
    s=12
)

ax[1].set_title(f"K-Means (k=2) | ARI = {ari_km:.3f}")

# (c) DBSCAN: core, border, and noise points
for c, col in zip(sorted(set(labels) - {-1}), colors):
    cluster_mask = labels == c

    ax[2].scatter(
        X_scaled[cluster_mask & core_mask, 0],
        X_scaled[cluster_mask & core_mask, 1],
        color=col,
        s=25,
        edgecolor="black",
        linewidth=0.3,
        label=f"Cluster {c} (core)"
    )

    ax[2].scatter(
        X_scaled[cluster_mask & border_mask, 0],
        X_scaled[cluster_mask & border_mask, 1],
        color=col,
        s=10,
        alpha=0.6,
        label=f"Cluster {c} (border)"
    )

ax[2].scatter(
    X_scaled[noise_mask, 0],
    X_scaled[noise_mask, 1],
    color="black",
    marker="x",
    s=25,
    label=f"Noise ({n_noise})"
)

ax[2].set_title(
    f"DBSCAN (eps={EPS}, MinPts={MIN_PTS}) "
    f"| ARI = {ari_db:.3f}"
)

ax[2].legend(fontsize=8, loc="best")

# Format all subplots
for a in ax:
    a.set_xlabel("x (scaled)")
    a.set_ylabel("y (scaled)")
    a.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("dbscan_vs_kmeans.png", dpi=150)
plt.show()

# ------------------------------------------------------------
# 10. SAVE CONFIRMATION
# ------------------------------------------------------------
print("\nPlots saved:")
print("1. k_distance_graph.png")
print("2. dbscan_vs_kmeans.png")
