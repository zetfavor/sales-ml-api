from fastapi.testclient import TestClient
from api import app  # Import our FastAPI "app"

# Create a "fake" client for testing, based on our real app
client = TestClient(app)

def test_read_root():
    """Tests if the root '/' endpoint is working."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Welcome to the Sales Prediction API"}

def test_predict_endpoint_success():
    """Tests if the /predict/ endpoint returns a valid prediction."""

    # This is the 15-feature "Order Form"
    test_data = {
        "feature_0": 0.5, "feature_1": -1.2, "feature_2": 0.0,
        "feature_3": 0.8, "feature_4": 1.1, "feature_5": -0.5,
        "feature_6": 2.3, "feature_7": -0.1, "feature_8": 0.9,
        "feature_9": -1.0, "feature_10": 0.2, "feature_11": 0.7,
        "feature_12": -0.3, "feature_13": 1.4, "feature_14": -0.8
    }

    response = client.post("/predict/", json=test_data)

    # Check 1: Did it succeed?
    assert response.status_code == 200

    # Check 2: Did it return the "dish" in the right format?
    data = response.json()
    assert "prediction" in data
    assert "model_version" in data

    # Check 3: Was the prediction valid?
    assert data["prediction"] == 0 or data["prediction"] == 1