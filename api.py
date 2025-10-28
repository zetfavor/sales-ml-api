import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

# 1. Initialize the FastAPI app
# This is the main "waiter" object
app = FastAPI(
    title="Sales Prediction API",
    description="An API to predict sales potential using an XGBoost model."
)

# 2. Define the "Data Contract" (the ingredients)
# Pydantic ensures the data we receive is valid.
# Our model was trained on 15 features.
class SalesFeatures(BaseModel):
    feature_0: float
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float
    feature_6: float
    feature_7: float
    feature_8: float
    feature_9: float
    feature_10: float
    feature_11: float
    feature_12: float
    feature_13: float
    feature_14: float

    # This is an example of what the user would send:
    # { "feature_0": 0.5, "feature_1": -1.2, ... }


# 3. Load the Chef (The Model)
# We load it *once* at startup, not on every request. This is fast.
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "final_sales_model.pkl"
model = joblib.load(MODEL_PATH)

print(f"Model loaded successfully from {MODEL_PATH}")


# 4. Define the "Endpoint" (The Waiter's Station)
# @app.post() means this endpoint receives data.
# "/predict/" is the URL path.
@app.post("/predict/")
def predict_sales(data: SalesFeatures):
    """
    Takes in 15 features of a sales lead and returns a prediction.
    """

    # 1. Convert the Pydantic data into a dict
    data_dict = data.model_dump()

    # 2. Convert the dict into a 1-row DataFrame
    # The model *expects* a DataFrame, just like in training.
    features_df = pd.DataFrame([data_dict])

    # 3. Get the prediction from the "Chef"
    prediction_raw = model.predict(features_df)

    # 4. Format the prediction for the "Customer"
    # The result is a numpy array (e.g., [1]), so we extract the first item
    prediction_int = int(prediction_raw[0])

    # 5. Return the "dish" as a clean JSON
    return {
        "prediction": prediction_int,
        "model_version": "v1.0.0" # Good practice to version your model
    }


# A simple "health check" endpoint at the root
@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to the Sales Prediction API"}