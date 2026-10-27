
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------------
# PRACTICAL 2: COMPARE CLASSIFIERS
# Breast Cancer Wisconsin Dataset
# ---------------------------------------------------------

# 1. Load the dataset
data = load_breast_cancer()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)
y = data.target

print("Dataset loaded successfully!")
print("Dataset shape:", X.shape)
print("Target classes:", data.target_names)

print("\nFirst 5 rows:")
print(X.head())

print("\nClass distribution:")
print(
    pd.Series(y)
    .map({0: "Malignant", 1: "Benign"})
    .value_counts()
)

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Training features:", X_train.shape[1])

# 3. Define the classification models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    )
}

# 4. Train models and evaluate their performance
results = []
trained_models = {}

for name, model in models.items():
    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time
    y_pred = model.predict(X_test)

    trained_models[name] = model

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision (Malignant)": precision_score(
            y_test,
            y_pred,
            pos_label=0,
            zero_division=0
        ),
        "Recall (Malignant)": recall_score(
            y_test,
            y_pred,
            pos_label=0,
            zero_division=0
        ),
        "F1-Score (Malignant)": f1_score(
            y_test,
            y_pred,
            pos_label=0,
            zero_division=0
        ),
        "Training Time (seconds)": training_time
    })

# 5. Display the evaluation results
results_df = pd.DataFrame(results)

print("\nModel evaluation:")
print(results_df.round(4).to_string(index=False))

# 6. Plot performance comparison metrics
metrics = [
    "Accuracy",
    "Precision (Malignant)",
    "Recall (Malignant)",
    "F1-Score (Malignant)"
]

results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Performance Comparison of Machine Learning Models")
plt.ylabel("Score")
plt.xlabel("Model")
plt.ylim(0, 1.05)
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# 7. Compare training times
plt.figure(figsize=(8, 5))

plt.bar(
    results_df["Model"],
    results_df["Training Time (seconds)"]
)

plt.title("Training Time Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Training Time (seconds)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 8. Plot confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for ax, (name, model) in zip(axes, trained_models.items()):
    y_pred = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"],
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

plt.tight_layout()
plt.show()

# 9. Print classification reports
for name, model in trained_models.items():
    print("\n", "=" * 55)
    print(name)
    print("=" * 55)

    y_pred = model.predict(X_test)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=data.target_names,
            zero_division=0
        )
    )

# 10. Identify the best model based on malignant recall
best_model = results_df.loc[
    results_df["Recall (Malignant)"].idxmax()
]

print("\nBest model based on malignant-class recall:")
print(best_model["Model"])
print("Accuracy:", round(best_model["Accuracy"], 4))
print("Precision:", round(best_model["Precision (Malignant)"], 4))
print("Recall:", round(best_model["Recall (Malignant)"], 4))
print("F1-Score:", round(best_model["F1-Score (Malignant)"], 4))
print(
    "Training Time:",
    round(best_model["Training Time (seconds)"], 4),
    "seconds"
)

# 11. Random Forest experiment with different tree counts
tree_counts = [10, 50, 100, 200, 300]
rf_results = []

for n in tree_counts:
    rf = RandomForestClassifier(
        n_estimators=n,
        random_state=42
    )

    start_time = time.time()

    rf.fit(X_train, y_train)

    training_time = time.time() - start_time
    y_pred = rf.predict(X_test)

    rf_results.append({
        "Number of Trees": n,
        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),
        "Malignant Recall": recall_score(
            y_test,
            y_pred,
            pos_label=0,
            zero_division=0
        ),
        "Training Time (seconds)": training_time
    })

# 12. Display Random Forest experiment results
rf_results_df = pd.DataFrame(rf_results)

print("\nRandom Forest tree-count experiment:")
print(rf_results_df.round(4).to_string(index=False))

# 13. Plot the effect of the number of trees
rf_results_df.set_index("Number of Trees")[
    ["Accuracy", "Malignant Recall"]
].plot(
    marker="o",
    figsize=(9, 5)
)

plt.title("Effect of Number of Trees on Random Forest")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
