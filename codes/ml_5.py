
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler




# Step 1: Generate dataset with 6 features
np.random.seed(42)

X, y = make_regression(
    n_samples=500,
    n_features=6,
    n_informative=5,
    noise=25,
    random_state=42
)

# Introduce multicollinearity
X[:, 1] = 0.9 * X[:, 0] + 0.1 * np.random.randn(500)

feature_names = ["s1", "s2", "s3", "s4", "s5", "s6"]

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y


# Step 2: Display correlation matrix
print("\nCorrelation among serum features:")
print(df[feature_names].corr().round(2))


# Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_names],
    df["target"],
    test_size=0.2,
    random_state=42
)


# Step 4: Define models
models = {
    "Standard OLS": LinearRegression(),

    "Ridge (L2)": make_pipeline(
        StandardScaler(),
        Ridge(alpha=10)
    ),

    "Lasso (L1)": make_pipeline(
        StandardScaler(),
        Lasso(alpha=1.0, max_iter=10000)
    )
}


# Step 5: Train and evaluate models
results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(
        y_train,
        y_train_pred
    )

    test_mse = mean_squared_error(
        y_test,
        y_test_pred
    )

    test_r2 = r2_score(
        y_test,
        y_test_pred
    )

    if name == "Standard OLS":
        coef = model.coef_
    else:
        key = "ridge" if name == "Ridge (L2)" else "lasso"
        coef = model.named_steps[key].coef_

    non_zero = np.sum(np.abs(coef) > 1e-6)

    results.append([
        name,
        train_mse,
        test_mse,
        test_r2,
        non_zero
    ])


# Step 6: Display model performance
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Train MSE",
        "Test MSE",
        "Test R2",
        "Non-Zero Coefs"
    ]
)

print("\n--- Model Performance Comparison ---")

print(
    results_df.to_string(
        index=False,
        float_format="%.4f"
    )
)


# Step 7: Plot model performance
results_df.set_index("Model")[
    ["Train MSE", "Test MSE"]
].plot(
    kind="bar",
    figsize=(9, 5)
)

plt.title("Model Performance Comparison")
plt.ylabel("Mean Squared Error")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
