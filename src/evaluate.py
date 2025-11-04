import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

model = joblib.load("outputs/model.pkl")

# Dummy evaluation dataset
from sklearn.datasets import load_iris
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data["target"] = iris.target

preds = model.predict(data[iris.feature_names])
acc = accuracy_score(data["target"], preds)
print(f"Model accuracy: {acc}")
