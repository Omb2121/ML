
# ==========================================
# TITANIC DATASET CLEANING
# ==========================================



# Step 1: Import Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 2: Load Titanic Dataset
df = pd.read_csv(r"C:\Users\neelr\Downloads\titanic.csv")

# Step 3: Display First 5 Rows
print("First 5 Rows")
print(df.head())

# Step 4: Display Dataset Information
print("\nDataset Information")
print(df.info())

# Step 5: Check Missing Values
print("\nMissing Values")
print(df.isnull().sum())


# ==========================================
# DATA PREPROCESSING
# ==========================================

# Step 6: Fill Missing Age with Median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Step 7: Fill Missing Embarked with Mode
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

# Step 8: Drop Cabin Column (Too Many Missing Values)
df.drop("Cabin", axis=1, inplace=True)

# Step 9: Drop Unnecessary Columns
df.drop(
    ["PassengerId", "Name", "Ticket"],
    axis=1,
    inplace=True
)

# Step 10: Convert Categorical Data into Numerical Data
le = LabelEncoder()

df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

# Step 11: Display Cleaned Dataset
print("\nCleaned Dataset")
print(df.head())

# Step 12: Separate Features and Target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Step 13: Split Dataset into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# MACHINE LEARNING MODEL
# ==========================================

# Step 14: Create Logistic Regression Model
model = LogisticRegression(max_iter=1000)

# Step 15: Train Model
model.fit(X_train, y_train)

# Step 16: Predict
y_pred = model.predict(X_test)

# Step 17: Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Step 18: Compare Actual vs Predicted Values
result = pd.DataFrame({
    "Actual": y_test,
    "Predicted": y_pred
})

print("\nActual vs Predicted")
print(result.head(10))

df.to_csv(
    r"C:\Users\neelr\Downloads\titanic_proccessed.csv",
    index=False
)

print("Cleaned dataset saved successfully.")
