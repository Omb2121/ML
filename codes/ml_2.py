
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------
df = pd.read_excel("Wine_Dataset.xlsx")

print("First 5 rows of dataset:")
print(df.head())

# --------------------------------------------------
# 2. Separate Features and Target
# --------------------------------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# --------------------------------------------------
# 3. Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 4. Standardization
# --------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================================================
# 5. Random Forest with Original Features
# ==================================================
start = time.time()

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train)

training_time_original = time.time() - start

pred_original = rf.predict(X_test_scaled)

accuracy_original = accuracy_score(
    y_test,
    pred_original
)

print("\nOriginal Accuracy:", accuracy_original)
print("Training Time:", training_time_original)

# ==================================================
# 6. PCA + Random Forest
# ==================================================
pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print(
    "Total Explained Variance:",
    pca.explained_variance_ratio_.sum()
)

start = time.time()

rf_pca = RandomForestClassifier(random_state=42)
rf_pca.fit(X_train_pca, y_train)

training_time_pca = time.time() - start

pred_pca = rf_pca.predict(X_test_pca)

accuracy_pca = accuracy_score(
    y_test,
    pred_pca
)

print("\nPCA Accuracy:", accuracy_pca)
print("Training Time:", training_time_pca)

# ==================================================
# 7. LDA + Random Forest
# ==================================================
lda = LDA(n_components=2)

X_train_lda = lda.fit_transform(
    X_train_scaled,
    y_train
)

X_test_lda = lda.transform(X_test_scaled)

start = time.time()

rf_lda = RandomForestClassifier(random_state=42)
rf_lda.fit(X_train_lda, y_train)

training_time_lda = time.time() - start

pred_lda = rf_lda.predict(X_test_lda)

accuracy_lda = accuracy_score(
    y_test,
    pred_lda
)

print("\nLDA Accuracy:", accuracy_lda)
print("Training Time:", training_time_lda)

# ==================================================
# 8. Final Comparison
# ==================================================
results = pd.DataFrame({
    "Method": [
        "Original Features",
        "PCA (2 Components)",
        "LDA (2 Components)"
    ],
    "No. of Features": [
        X.shape[1],
        2,
        2
    ],
    "Accuracy (%)": [
        round(accuracy_original * 100, 2),
        round(accuracy_pca * 100, 2),
        round(accuracy_lda * 100, 2)
    ],
    "Training Time (sec)": [
        round(training_time_original, 5),
        round(training_time_pca, 5),
        round(training_time_lda, 5)
    ]
})

print("\n========================================")
print("FINAL RESULTS")
print("========================================")
print(results.to_string(index=False))
