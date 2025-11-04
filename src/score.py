import json
import numpy as np
import joblib
import os

def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl")
    model = joblib.load(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)["data"])
    preds = model.predict(data)
    return json.dumps({"predictions": preds.tolist()})
