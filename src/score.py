import json
import numpy as np
import joblib
import os

def init():
    global model

    # Root directory where Azure places your model
    model_dir = os.getenv("AZUREML_MODEL_DIR")

    # ✅ Your model is inside model_output/model.joblib (not directly at root)
    model_path = os.path.join(model_dir, "model_output", "model.joblib")

    # Load model
    model = joblib.load(model_path)
    print(f"✅ Model loaded from: {model_path}")

def run(raw_data):
    try:
        # Parse input JSON
        data = json.loads(raw_data)
        input_array = np.array(data["data"])

        # Predict
        preds = model.predict(input_array)

        # Return predictions in JSON
        return json.dumps({"predictions": preds.tolist()})

    except Exception as e:
        # Return readable error message to endpoint clients
        return json.dumps({"error": str(e)})
