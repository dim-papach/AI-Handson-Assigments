import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from src.config import PipelineConfig
from src.preprocessing import (
    compute_colors,
    apply_iqr_capping,
    apply_scaling,
    apply_imputation,
)

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    Input features for a single galaxy observation.
    
    Attributes
    ----------
    T : float
        T index.
    WF1 : float
        WISE W1 flux.
    WF2 : float
        WISE W2 flux.
    WF3 : float
        WISE W3 flux.
    WF4 : float
        WISE W4 flux.
    UT : float
        GALEX Ultraviolet flux.
    BT : float
        Blue total magnitude.
    U : float
        SDSS u-band magnitude.
    G : float
        SDSS g-band magnitude.
    R : float
        SDSS r-band magnitude.
    I : float
        SDSS i-band magnitude.
    Z : float
        SDSS z-band magnitude.
    logM_HEC : float
        Log of stellar mass from HECATE.
    logSFR_HEC : float
        Log of star formation rate from HECATE.
    METAL : float
        Metallicity flag/value.
    AGN_HEC : str
        AGN classification ('Y' or 'N').
    """
    # Key variables
    T: float = Field(..., description="T index")
    WF1: float = Field(..., description="WISE W1 flux")
    WF2: float = Field(..., description="WISE W2 flux")
    WF3: float = Field(..., description="WISE W3 flux")
    WF4: float = Field(..., description="WISE W4 flux")
    UT: float = Field(..., description="GALEX Ultraviolet flux")
    BT: float = Field(..., description="Blue total magnitude")
    U: float = Field(..., description="SDSS u-band magnitude")
    G: float = Field(..., description="SDSS g-band magnitude")
    R: float = Field(..., description="SDSS r-band magnitude")
    I: float = Field(..., description="SDSS i-band magnitude")
    Z: float = Field(..., description="SDSS z-band magnitude")
    
    # Metadata/Physical properties
    logM_HEC: float = Field(..., description="Log of stellar mass from HECATE")
    logSFR_HEC: float = Field(..., description="Log of star formation rate from HECATE")
    METAL: float = Field(..., description="Metallicity flag/value")
    AGN_HEC: str = Field(..., description="AGN classification ('Y' or 'N')")

    model_config = {
        "json_schema_extra": {
            "example": {
                "T": 5.0, "WF1": 12.5, "WF2": 10.2, "WF3": 5.1, "WF4": 2.3,
                "UT": 18.5, "BT": 13.2, "U": 14.5, "G": 13.8, "R": 13.5,
                "I": 13.3, "Z": 13.2, "logM_HEC": 10.5, "logSFR_HEC": 0.5,
                "METAL": 0.02, "AGN_HEC": "N"
            }
        }
    }

class PredictionResponse(BaseModel):
    """
    Prediction output from the model.
    
    Attributes
    ----------
    prediction : int
        The predicted class ID.
    label : str
        The human-readable label for the class.
    probability : float
        The confidence score for the prediction.
    status : str
        The status of the request (default: "success").
    """
    prediction: int
    label: str
    probability: float
    status: str = "success"

# ---------------------------------------------------------------------------
# API State & Lifecycle
# ---------------------------------------------------------------------------

class APIState:
    """
    Container for singleton instances of models and artifacts.
    
    Attributes
    ----------
    model : Any
        The loaded best model (sklearn or torch).
    scaler : Any
        The fitted StandardScaler.
    le : Any
        The fitted LabelEncoder for targets.
    iqr_bounds : Dict[str, Tuple[float, float]]
        The IQR boundaries for outlier capping.
    pipeline : Any
        The imputation and encoding pipeline.
    is_pt : bool
        Flag indicating if the model is a PyTorch model.
    config : PipelineConfig
        Unified configuration object.
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.le = None
        self.iqr_bounds = None
        self.pipeline = None
        self.is_pt = False
        self.config = PipelineConfig()

state = APIState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager for loading artifacts on startup.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance.
    """
    models_dir = state.config.models_dir
    
    # Load Label Encoder and Scaler first
    state.le = joblib.load(os.path.join(models_dir, state.config.encoder_filename))
    state.scaler = joblib.load(os.path.join(models_dir, state.config.scaler_filename))
    state.iqr_bounds = joblib.load(os.path.join(models_dir, "iqr_bounds.pkl"))
    state.pipeline = joblib.load(os.path.join(models_dir, "imputation_pipeline.pkl"))

    # Load model
    if os.path.exists(os.path.join(models_dir, "best_model.pkl")):
        state.model = joblib.load(os.path.join(models_dir, "best_model.pkl"))
        state.is_pt = False
    elif os.path.exists(os.path.join(models_dir, "best_model.pt")):
        import torch
        from src.train_neural import SimpleNN
        nc = len(state.le.classes_)
        is_binary = nc <= 2
        od = 1 if is_binary else nc
        
        state.model = SimpleNN(
            input_dim=21, 
            hidden_layers=state.config.nn_hidden_layers,
            output_dim=od,
            dropout=state.config.nn_dropout,
            activation=state.config.nn_activation
        )
        state.model.load_state_dict(torch.load(os.path.join(models_dir, "best_model.pt"), map_location=torch.device('cpu')))
        state.model.eval()
        state.is_pt = True
    else:
        raise RuntimeError("No best_model found in models directory")
    
    yield

app = FastAPI(
    title="Galaxy Classification API",
    description="REST API to predict galaxy types from photometric and physical features.",
    version="1.0.0",
    lifespan=lifespan
)

# ---------------------------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
async def predict(observation: Observation):
    """
    Run inference for a single observation through the full pipeline.

    Takes raw input features, applies IQR capping, scaling, imputation, 
    and feature engineering, then returns the model's prediction.

    Parameters
    ----------
    observation : Observation
        The input features validated by Pydantic.

    Returns
    -------
    PredictionResponse
        The prediction result including label and probability.

    Raises
    ------
    HTTPException
        If any error occurs during processing or inference.
    """
    try:
        # 1. Convert to DataFrame
        input_data = [observation.model_dump()]
        df = pd.DataFrame(input_data)
        
        # 2. Preprocessing
        dummy_target = "CLASS_SP"
        df[dummy_target] = 0
        
        # Add missing columns as NaN for the imputation pipeline
        # These are required by the ColumnTransformer state
        required_cols = ['T', 'WF1', 'WF2', 'WF3', 'WF4', 'UT', 'BT', 'VT', 'IT', 'U', 'R', 'G', 'I', 'Z']
        for var in required_cols:
            if var not in df.columns:
                df[var] = np.nan
            
            err_col = f"E_{var}"
            if err_col not in df.columns:
                df[err_col] = np.nan

        # Get numeric columns used for scaling
        num_cols = state.scaler.feature_names_in_
        
        df_processed = (
            df
            .pipe(apply_iqr_capping, bounds=state.iqr_bounds)
            .pipe(apply_scaling, scaler=state.scaler, num_cols=num_cols, target_col=dummy_target)
            .pipe(apply_imputation, pipeline=state.pipeline, target_col=dummy_target)
            .pipe(compute_colors)
        )
        
        # Ensure column order matches training
        expected_cols = ['T', 'WF1', 'WF2', 'WF3', 'WF4', 'UT', 'BT', 'U', 'R', 'G', 'I', 'Z', 
                         'logM_HEC', 'logSFR_HEC', 'METAL', 'AGN_HEC_N', 'AGN_HEC_Y', 
                         'u-g', 'g-r', 'W3-UT', '(W3+UT)/W1']
        
        # If any are missing (like OHE columns), add them as 0
        for col in expected_cols:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        X = df_processed[expected_cols]
        
        # 3. Inference
        if state.is_pt:
            import torch
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
            with torch.no_grad():
                logits = state.model(X_tensor)
                if X_tensor.shape[0] == 1:
                    # Single sample
                    if getattr(state.model, "output_dim", 1) == 1:
                        prob = torch.sigmoid(logits).item()
                        pred = int(prob >= 0.5)
                    else:
                        probs = torch.softmax(logits, dim=1).numpy()
                        pred = np.argmax(probs, axis=1)[0]
                        prob = float(probs[0][pred])
        else:
            pred = state.model.predict(X)[0]
            try:
                probs = state.model.predict_proba(X)
                prob = float(np.max(probs[0]))
            except (AttributeError, ValueError):
                prob = 1.0
            
        # 4. Decode Label
        label = state.le.inverse_transform([pred])[0]
        
        return PredictionResponse(
            prediction=int(pred),
            label=str(label),
            probability=prob
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Simple health check endpoint."""
    return {"status": "Healthy and Alive: The API is working"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
