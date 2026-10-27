

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)


# 1. Load the Breast Cancer Wisconsin dataset
data = load_breast_cancer()

X = data.data
y = data.target


# 2. Display dataset shape
print("Dataset Shape:", X.shape)


# 3. Display class distribution
class_distribution = (
    pd.Series(y)
    .value_counts(normalize=True)
    .sort_index()
)

print("Class Distribution:")
print(f"1 (Benign)      {class_distribution[1]:.3f}")
print(f"0 (Malignant)   {class_distribution[0]:.3f}")


# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 5. Create and train the Logistic Regression model
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=10000)
)

model.fit(X_train, y_train)


# 6. Make predictions
y_pred = model.predict(X_test)


# 7. Calculate performance metrics for the benign class (class 1)
precision = precision_score(
    y_test,
    y_pred,
    pos_label=1
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=1
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1
)

print("\n--- Summary Performance Metrics (Benign Class) ---")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")


# 8. Display the comprehensive classification report
print("\n--- Comprehensive Classification Report ---")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["malignant", "benign"],
        digits=2
    )
)


# 9. Display the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("--- Confusion Matrix ---")
print(cm)


# 10. Display confusion matrix breakdown
tn, fp, fn, tp = cm.ravel()

print("\nBreakdown of the Confusion Matrix:")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
