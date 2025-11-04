import argparse
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="Path to input data")
parser.add_argument("--output", type=str, help="Path to save model")
args = parser.parse_args()

print("📂 Loading dataset...")

# Load dataset (fallback to Iris if not found)
try:
    data = pd.read_csv(args.data)
    print(f"✅ Loaded dataset from {args.data}")
except Exception as e:
    print(f"⚠️ Could not load provided dataset: {e}")
    print("Using built-in Iris dataset instead...")
    from sklearn.datasets import load_iris
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data["target"] = iris.target

# Prepare features and labels
X = data.drop("target", axis=1)
y = data["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("🧠 Training Logistic Regression model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained successfully with accuracy: {accuracy:.4f}")

# Ensure output directory exists
os.makedirs(os.path.dirname(args.output), exist_ok=True)

# Save model
joblib.dump(model, args.output)
print(f"💾 Model saved at: {args.output}")
