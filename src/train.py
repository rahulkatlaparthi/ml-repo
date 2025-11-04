import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, help="Path to input data")
parser.add_argument("--output", type=str, help="Path to save model")
args = parser.parse_args()

# Sample data (if no dataset provided)
try:
    data = pd.read_csv(args.data)
except Exception:
    from sklearn.datasets import load_iris
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data["target"] = iris.target

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

joblib.dump(model, args.output)
print(f"✅ Model trained and saved at {args.output}")
