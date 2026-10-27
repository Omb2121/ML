
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ==================================================
# 1. Student Information
# ==================================================



# ==================================================
# 2. Load Dataset
# ==================================================

df = pd.read_csv("housing.csv")

print("\nHousing Dataset:")
print(df.head())


# ==================================================
# 3. Check Dataset
# ==================================================

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# ==================================================
# 4. Handle Missing Values
# ==================================================

# total_bedrooms has 207 missing values
# Replace missing values with median

df["total_bedrooms"] = df["total_bedrooms"].fillna(
    df["total_bedrooms"].median()
)

print("\nMissing Values After Handling:")
print(df.isnull().sum())


# ==================================================
# 5. Define X and y
# ==================================================

X = df[
    [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income"
    ]
]

y = df["median_house_value"]


# ==================================================
# 6. Split Dataset
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# ==================================================
# 7. Create Linear Regression Model
# ==================================================

model = LinearRegression()
model.fit(X_train, y_train)


# ==================================================
# 8. Display Intercept
# ==================================================

print("\nIntercept:")
print(model.intercept_)


# ==================================================
# 9. Display Coefficients
# ==================================================

print("\nCoefficients:")

for feature, coefficient in zip(X.columns, model.coef_):
    print(f"{feature}: {coefficient:.2f}")


# ==================================================
# 10. Prediction
# ==================================================

y_pred = model.predict(X_test)


# ==================================================
# 11. Model Evaluation
# ==================================================

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("----------------------------")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")


# ==================================================
# 12. Actual vs Predicted
# ==================================================

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted:")
print(comparison.head(10))


# ==================================================
# 13. Predict a New House
# ==================================================

new_house = pd.DataFrame({
    "longitude": [-122.23],
    "latitude": [37.88],
    "housing_median_age": [20],
    "total_rooms": [2000],
    "total_bedrooms": [400],
    "population": [800],
    "households": [300],
    "median_income": [5.0]
})

predicted_price = model.predict(new_house)

print("\nPredicted Price for New House:")
print(f"{predicted_price[0]:.2f}")


# ==================================================
# 14. Program Completed
# ==================================================

print("\nProgram executed successfully!")
