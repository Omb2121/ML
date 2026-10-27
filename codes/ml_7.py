

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# STEP 1: Load the Heart Disease dataset
heart = fetch_ucirepo(id=45)

X = heart.data.features.copy()
y = heart.data.targets.iloc[:, 0].copy()

# Convert features and target to numeric values
X = X.apply(pd.to_numeric, errors="coerce")
y = pd.to_numeric(y, errors="coerce")

# Remove rows with missing target values
valid = y.notna()

X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

# Binary classification: 0 = No Heart Disease, 1 = Heart Disease
y = (y > 0).astype(int)


# STEP 2: Display dataset shape and class distribution
print(f"Dataset Shape: ({len(X)}, {X.shape[1] + 1})")

print("Class Distribution:")

counts = y.value_counts(normalize=True).sort_index()

print(
    f"1 (Heart Disease Present): {int((y == 1).sum())} "
    f"({counts[1] * 100:.2f}%)"
)

print(
    f"0 (No Heart Disease)     : {int((y == 0).sum())} "
    f"({counts[0] * 100:.2f}%)"
)


# STEP 3: Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# STEP 4: Define SVM kernels
kernels = {
    "Linear": SVC(
        kernel="linear",
        C=1.0
    ),

    "RBF (Gaussian)": SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale"
    ),

    "Polynomial (deg=3)": SVC(
        kernel="poly",
        degree=3,
        C=1.0,
        gamma="scale",
        coef0=0.0
    )
}

results = {}
models = {}


# STEP 5: Train and evaluate each SVM model
for name, svm in kernels.items():

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        svm
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    scores = model.decision_function(X_test)

    results[name] = {
        "Support Vectors": int(
            model.named_steps["svc"].n_support_.sum()
        ),

        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "F1-Score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_test,
            scores
        )
    }

    models[name] = model


# STEP 6: Print comparative performance table
print("\n--- Support Vector Machine Kernel Performance Comparison ---")

print(
    f"{'Kernel':<22}"
    f"{'Support Vectors':>16}"
    f"{'Accuracy':>11}"
    f"{'Precision':>11}"
    f"{'Recall':>9}"
    f"{'F1-Score':>11}"
    f"{'ROC-AUC':>10}"
)

for name, metrics in results.items():

    print(
        f"{name:<22}"
        f"{metrics['Support Vectors']:>16}"
        f"{metrics['Accuracy']:>11.4f}"
        f"{metrics['Precision']:>11.4f}"
        f"{metrics['Recall']:>9.4f}"
        f"{metrics['F1-Score']:>11.4f}"
        f"{metrics['ROC-AUC']:>10.4f}"
    )


# STEP 7: Detailed report for the RBF kernel
rbf_model = models["RBF (Gaussian)"]
rbf_pred = rbf_model.predict(X_test)

print("\n--- Detailed Classification Report for RBF Kernel ---")

print(
    classification_report(
        y_test,
        rbf_pred,
        digits=2,
        zero_division=0
    )
)


# STEP 8: Confusion matrix for RBF kernel
cm = confusion_matrix(
    y_test,
    rbf_pred,
    labels=[0, 1]
)

print("--- Confusion Matrix (RBF Kernel) ---")
print(cm)


# STEP 9: Visualize 2D decision surfaces using two features
from sklearn.inspection import DecisionBoundaryDisplay

# Use two standardized features for visualization only
X_2d = X.iloc[:, :2]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X_2d,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, svm) in zip(axes, kernels.items()):

    plot_model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(
            kernel=svm.kernel,
            C=1.0,
            gamma="scale",
            degree=3
        )
    )

    plot_model.fit(X2_train, y2_train)

    DecisionBoundaryDisplay.from_estimator(
        plot_model,
        X2_train,
        response_method="predict",
        plot_method="contourf",
        alpha=0.3,
        ax=ax
    )

    ax.scatter(
        X2_train.iloc[:, 0],
        X2_train.iloc[:, 1],
        c=y2_train,
        edgecolors="black",
        s=20
    )

    ax.set_title(f"{name} Kernel")
    ax.set_xlabel(X2_train.columns[0])
    ax.set_ylabel(X2_train.columns[1])

plt.tight_layout()
plt.show()
