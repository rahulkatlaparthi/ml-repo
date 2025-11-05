import json
import joblib
import os

def init():
    global model

    # Root directory where Azure places your model
    model_dir = os.getenv("AZUREML_MODEL_DIR")

    # Model is inside model_output/model.joblib
    model_path = os.path.join(model_dir, "model_output", "model.joblib")

    # Load the model
    model = joblib.load(model_path)
    print(f"✅ Model loaded from: {model_path}")

def run(raw_data):
    try:
        # Parse input JSON
        data = json.loads(raw_data)

        # Ensure data is a list of lists (like [[x1, x2, ...]])
        input_list = data["data"]

        # Predict (scikit-learn models accept lists)
        preds = model.predict(input_list)

        # Return predictions in JSON
        return json.dumps({"predictions": preds.tolist()})

    except Exception as e:
        # Return readable error message
        return json.dumps({"error": str(e)})
