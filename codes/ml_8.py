

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# STEP 1: Load the Mall Customer dataset
df = pd.read_csv("Mall_Customers.csv")

print(f"Dataset Shape: {df.shape}")
print(df.head())


# STEP 2: Select the two clustering features
features = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# STEP 3: Evaluate cluster counts from 2 to 10
k_range = range(2, 11)

inertias = []
silhouette_scores = []

print("\n--- Cluster Evaluation Metrics across k ---")
print(f"{'k':<5}{'WCSS (Inertia)':<20}{'Silhouette Score':<20}")

for k in k_range:

    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, labels)

    inertias.append(inertia)
    silhouette_scores.append(silhouette)

    print(f"{k:<5}{inertia:<20.2f}{silhouette:<20.4f}")


# STEP 4: Plot the Elbow curve
plt.figure(figsize=(7, 5))

plt.plot(
    list(k_range),
    inertias,
    marker="o"
)

plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS (Inertia)")
plt.xticks(list(k_range))
plt.grid(True)
plt.tight_layout()
plt.show()


# STEP 5: Plot Silhouette scores
plt.figure(figsize=(7, 5))

plt.plot(
    list(k_range),
    silhouette_scores,
    marker="o"
)

plt.title("Silhouette Analysis")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Average Silhouette Score")
plt.xticks(list(k_range))
plt.grid(True)
plt.tight_layout()
plt.show()


# STEP 6: Select optimal k using the highest Silhouette score
optimal_k = list(k_range)[np.argmax(silhouette_scores)]

print(f"\nOptimal k based on Silhouette Score: {optimal_k}")


# STEP 7: Fit the final K-Means model
final_model = KMeans(
    n_clusters=optimal_k,
    init="k-means++",
    random_state=42,
    n_init=10
)

df["Cluster"] = final_model.fit_predict(X_scaled)


# STEP 8: Visualize clusters and their centroids
plt.figure(figsize=(9, 6))

for cluster in sorted(df["Cluster"].unique()):

    points = X_scaled[df["Cluster"].values == cluster]

    plt.scatter(
        points[:, 0],
        points[:, 1],
        label=f"Cluster {cluster}",
        s=45
    )

centroids = final_model.cluster_centers_

plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="X",
    s=250,
    edgecolors="black",
    label="Centroids"
)

plt.title(f"K-Means Customer Segmentation (k={optimal_k})")
plt.xlabel("Standardized Annual Income")
plt.ylabel("Standardized Spending Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# STEP 9: Display the first few assigned clusters
print("\n--- Final Cluster Assignments ---")

print(
    df[
        [
            "CustomerID",
            "Annual Income (k$)",
            "Spending Score (1-100)",
            "Cluster"
        ]
    ].head(10)
)

print("\n--- Number of Customers in Each Cluster ---")
print(df["Cluster"].value_counts().sort_index())
