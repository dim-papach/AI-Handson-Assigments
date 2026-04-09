import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import os

# Skip tests if dependencies are missing (though they should be there)
pytest.importorskip("fastapi")
pytest.importorskip("pydantic")

from src.api import app, state, Observation, PredictionResponse

client = TestClient(app)

def test_observation_schema() -> None:
    """Test standard Pydantic validation for the Observation model."""
    valid_data = {
        "T": 5.0, "WF1": 12.5, "WF2": 10.2, "WF3": 5.1, "WF4": 2.3,
        "UT": 18.5, "BT": 13.2, "U": 14.5, "G": 13.8, "R": 13.5,
        "I": 13.3, "Z": 13.2, "logM_HEC": 10.5, "logSFR_HEC": 0.5,
        "METAL": 0.02, "AGN_HEC": "N", "VT": 13.5, "IT": 13.3
    }
    obs = Observation(**valid_data)
    assert obs.T == 5.0
    assert obs.AGN_HEC == "N"

    # Test invalid data
    invalid_data = valid_data.copy()
    del invalid_data["T"]
    with pytest.raises(ValueError):
        Observation(**invalid_data)

def test_prediction_response_schema() -> None:
    """Test standard Pydantic validation for the PredictionResponse model."""
    data = {
        "prediction": 1,
        "label": "Galaxy",
        "probability": 0.95
    }
    resp = PredictionResponse(**data)
    assert resp.status == "success"
    assert resp.label == "Galaxy"

def test_health_endpoint() -> None:
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "Healthy and Alive: The API is working"}

@patch("src.api.state")
def test_predict_endpoint_classical(mock_state: MagicMock) -> None:
    """Test the /predict endpoint with a mocked classical model."""
    # Setup mock state
    mock_state.is_pt = False
    mock_model = MagicMock()
    mock_model.predict.return_value = [0]
    mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
    mock_state.model = mock_model
    
    mock_le = MagicMock()
    mock_le.inverse_transform.return_value = ["TypeA"]
    mock_state.le = mock_le
    
    mock_state.scaler = MagicMock()
    mock_state.scaler.feature_names_in_ = np.array(["T", "logM_HEC"])
    mock_state.iqr_bounds = {"T": (0, 10)}
    mock_state.pipeline = MagicMock()
    
    # Mocking preprocessor behavior is complex because of .pipe()
    # But we can mock the functions called by .pipe() in src.api or just mock the end result of preprocessing
    
    with patch("src.api.apply_iqr_capping", side_effect=lambda df, **kwargs: df):
        with patch("src.api.apply_scaling", side_effect=lambda df, **kwargs: df):
            with patch("src.api.apply_imputation", side_effect=lambda df, **kwargs: df):
                with patch("src.api.compute_colors", side_effect=lambda df, **kwargs: df):
                    
                    payload = {
                        "T": 5.0, "WF1": 12.5, "WF2": 10.2, "WF3": 5.1, "WF4": 2.3,
                        "UT": 18.5, "BT": 13.2, "U": 14.5, "G": 13.8, "R": 13.5,
                        "I": 13.3, "Z": 13.2, "logM_HEC": 10.5, "logSFR_HEC": 0.5,
                        "METAL": 0.02, "AGN_HEC": "N", "VT": 13.5, "IT": 13.3
                    }
                    response = client.post("/predict", json=payload)
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["prediction"] == 0
                    assert data["label"] == "TypeA"
                    assert data["probability"] == 0.9

@patch("src.api.state")
def test_predict_endpoint_error(mock_state: MagicMock) -> None:
    """Test the /predict endpoint error handling."""
    # Force an error in preprocessing
    with patch("src.api.pd.DataFrame", side_effect=Exception("Data Error")):
        payload = {
            "T": 5.0, "WF1": 12.5, "WF2": 10.2, "WF3": 5.1, "WF4": 2.3,
            "UT": 18.5, "BT": 13.2, "U": 14.5, "G": 13.8, "R": 13.5,
            "I": 13.3, "Z": 13.2, "logM_HEC": 10.5, "logSFR_HEC": 0.5,
            "METAL": 0.02, "AGN_HEC": "N"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 500
        assert "Data Error" in response.json()["detail"]
