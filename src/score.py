import json
import joblib
import os

def init():
    global model
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    model_path = os.path.join(model_dir, "model_output", "model.joblib")
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")

def run(raw_data):
    try:
        data = json.loads(raw_data)
        input_list = data["data"]  # Python list
        preds = model.predict(input_list)
        return json.dumps({"predictions": preds.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
