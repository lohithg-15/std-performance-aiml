import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pickle
import os

# 1. Load dataset
data = pd.read_csv("data/std-data.csv")

# 2. Split input and output
X = data.drop("final_score", axis=1)
y = data["final_score"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate model
predictions = model.predict(X_test)
error = mean_absolute_error(y_test, predictions)

print("Model trained successfully")
print("Mean Absolute Error:", error)

# 6. Save model
os.makedirs("model", exist_ok=True)
with open("model/student_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved in model/student_model.pkl")
