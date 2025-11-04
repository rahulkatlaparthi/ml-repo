import argparse
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--training_data", type=str)
parser.add_argument("--regularization", type=float, default=0.01)
parser.add_argument("--model_output", type=str)
args = parser.parse_args()

print("🔹 Loading data from:", args.training_data)
df = pd.read_csv(args.training_data)
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression(C=1/args.regularization)
model.fit(X_train, y_train)

print("✅ Training completed. Saving model...")
joblib.dump(model, f"{args.model_output}/model.joblib")
print("Model saved successfully.")
