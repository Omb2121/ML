
# ============================================================
# EXPERIMENT 4
# Polynomial Regression
# Bias-Variance Trade-off
# Dataset: Auto MPG
# ============================================================



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# STEP 1: INGESTION
# ============================================================

# Load dataset
df = pd.read_csv("auto-mpg.csv")

print("First 5 records:")
print(df.head())

print("\nOriginal Dataset Shape:", df.shape)


# ============================================================
# DATA CLEANING
# ============================================================

# We only need mpg and weight for this experiment.
# Therefore, remove rows only if mpg or weight is missing.

df = df.dropna(subset=["mpg", "weight"])

print("Dataset Shape after cleaning:", df.shape)


# ============================================================
# STEP 1: SELECT INPUT AND TARGET
# ============================================================

# Independent variable
# X = Weight of the car

X = df[["weight"]]

# Dependent variable
# y = Miles per gallon

y = df["mpg"]


# ============================================================
# STEP 1: TRAIN-TEST SPLIT
# 80% Training
# 20% Testing
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# STEP 2: POLYNOMIAL DEGREES
# ============================================================

degrees = [1, 2, 3, 5, 7, 10]
results = []

for degree in degrees:

    # Polynomial Regression Pipeline
    model = Pipeline([
        # Create polynomial features
        (
            "polynomial",
            PolynomialFeatures(
                degree=degree,
                include_bias=False
            )
        ),

        # Scale features
        (
            "scaler",
            StandardScaler()
        ),

        # Linear Regression
        (
            "regression",
            LinearRegression()
        )
    ])

    # Train Model
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # MSE
    train_mse = mean_squared_error(
        y_train,
        y_train_pred
    )

    test_mse = mean_squared_error(
        y_test,
        y_test_pred
    )

    # R²
    test_r2 = r2_score(
        y_test,
        y_test_pred
    )

    # Store results
    results.append([
        degree,
        train_mse,
        test_mse,
        test_r2
    ])


# ============================================================
# CREATE RESULT TABLE
# ============================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Degree",
        "Train MSE",
        "Test MSE",
        "Test R²"
    ]
)


# ============================================================
# STEP 4: DISPLAY OUTPUT
# ============================================================

print("\n")
print("=" * 75)
print("POLYNOMIAL REGRESSION RESULTS")
print("=" * 75)

print(
    f"{'Degree':<10}"
    f"{'Train MSE':<15}"
    f"{'Test MSE':<15}"
    f"{'Test R²':<15}"
    f"Diagnosis"
)

print("-" * 75)

for _, row in results_df.iterrows():

    degree = int(row["Degree"])

    train_mse = row["Train MSE"]
    test_mse = row["Test MSE"]
    test_r2 = row["Test R²"]

    # Diagnosis
    if degree == 1:
        diagnosis = "Underfitting: High Bias"

    elif degree in [2, 3]:
        diagnosis = "Optimal Trade-Off"

    elif degree >= 7:
        diagnosis = "Overfitting: High Variance"

    else:
        diagnosis = ""

    print(
        f"{degree:<10}"
        f"{train_mse:<15.4f}"
        f"{test_mse:<15.4f}"
        f"{test_r2:<15.4f}"
        f"{diagnosis}"
    )

print("-" * 75)


# ============================================================
# FIND BEST DEGREE
# ============================================================

best_index = results_df["Test MSE"].idxmin()

best_degree = int(
    results_df.loc[best_index, "Degree"]
)

best_test_mse = results_df.loc[
    best_index, "Test MSE"
]

print("\nBest Polynomial Degree:", best_degree)
print("Lowest Test MSE:", round(best_test_mse, 4))


# ============================================================
# STEP 5: BIAS-VARIANCE GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

# Training MSE
plt.plot(
    results_df["Degree"],
    results_df["Train MSE"],
    marker="o",
    linewidth=2,
    color="blue",
    label="Training MSE"
)

# Testing MSE
plt.plot(
    results_df["Degree"],
    results_df["Test MSE"],
    marker="o",
    linewidth=2,
    color="red",
    label="Testing MSE"
)

# Highlight best degree
plt.scatter(
    best_degree,
    best_test_mse,
    color="green",
    s=150,
    zorder=5,
    label=f"Best Degree = {best_degree}"
)

# Labels
plt.xlabel(
    "Polynomial Degree",
    fontsize=12
)

plt.ylabel(
    "Mean Squared Error (MSE)",
    fontsize=12
)

plt.title(
    "Bias-Variance Trade-off in Polynomial Regression",
    fontsize=14,
    fontweight="bold"
)

plt.xticks(degrees)

plt.grid(
    True,
    linestyle="--",
    alpha=0.5
)

plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# STEP 6: INTERPRETATION
# ============================================================

print("\n")
print("=" * 75)
print("INTERPRETATION")
print("=" * 75)



print("=" * 75)
print("Experiment completed successfully!")
print("=" * 75)
