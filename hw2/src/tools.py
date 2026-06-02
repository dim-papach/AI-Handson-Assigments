import os
import joblib
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Add hw1 to sys.path so we can import its preprocessing modules
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(REPO_DIR, "hw1"))

from src.config import PipelineConfig
from src.preprocessing import (
    apply_iqr_capping,
    apply_imputation,
    filter_error_ratios,
    compute_colors,
    apply_scaling
)

# Load HW1 Model Artifacts globally
MODELS_DIR = os.path.join(BASE_DIR, "models")
try:
    imputation_pipeline = joblib.load(os.path.join(MODELS_DIR, "imputation_pipeline.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    iqr_bounds = joblib.load(os.path.join(MODELS_DIR, "iqr_bounds.pkl"))
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
except Exception as e:
    print(f"Warning: Could not load model artifacts. Ensure they are copied to hw2/models/. Error: {e}")

cfg = PipelineConfig()

CLASS_NAMES = {
    0: "star-forming",
    1: "Seyfert",
    2: "LINER",
    3: "composite"
}

class GalaxyPredictionInput(BaseModel):
    T: float = Field(description="Numerical Hubble-type")
    WF1: float = Field(description="3.3μm-band apparent magnitude")
    WF2: Optional[float] = Field(default=np.nan, description="4.6μm-band apparent magnitude")
    WF3: Optional[float] = Field(default=np.nan, description="12μm-band apparent magnitude")
    WF4: Optional[float] = Field(default=np.nan, description="22μm-band apparent magnitude")
    UT: float = Field(description="Total U-band apparent magnitude")
    BT: Optional[float] = Field(default=np.nan, description="B-band apparent magnitude")
    VT: Optional[float] = Field(default=np.nan, description="V-band apparent magnitude")
    IT: Optional[float] = Field(default=np.nan, description="I-band apparent magnitude")
    U: float = Field(description="u-band SDSS apparent magnitude")
    R: float = Field(description="r-band SDSS apparent magnitude")
    G: float = Field(description="g-band SDSS apparent magnitude")
    I: float = Field(description="i-band SDSS apparent magnitude")
    Z: float = Field(description="z-band SDSS apparent magnitude")
    logM_HEC: float = Field(description="Logarithm of total stellar mass")
    logSFR_HEC: float = Field(description="Logarithm of star-formation rate")
    METAL: float = Field(description="Metallicity")
    AGN_HEC: str = Field(description="Adopted activity classification (Y/N/?)")

@tool("predict_galaxy_class", args_schema=GalaxyPredictionInput)
def predict_galaxy_class(
    T: float, WF1: float, UT: float, U: float, R: float, G: float, I: float, Z: float,
    logM_HEC: float, logSFR_HEC: float, METAL: float, AGN_HEC: str,
    WF2: float = np.nan, WF3: float = np.nan, WF4: float = np.nan,
    BT: float = np.nan, VT: float = np.nan, IT: float = np.nan
) -> str:
    """
    Predicts the nuclear activity classification of a galaxy (star-forming, Seyfert, LINER, composite).
    Input must be provided with numerical galaxy properties (T, WF1, UT, U, R, G, I, Z, logM_HEC, logSFR_HEC, METAL)
    and the categorical string AGN_HEC ('Y', 'N', '?'). Optional inputs are WF2, WF3, WF4, BT, VT, IT.
    """
    # 1. Create DataFrame
    input_dict = {
        'T': T, 'WF1': WF1, 'WF2': WF2, 'WF3': WF3, 'WF4': WF4,
        'UT': UT, 'BT': BT, 'VT': VT, 'IT': IT, 'U': U, 'R': R, 'G': G, 'I': I, 'Z': Z,
        'logM_HEC': logM_HEC, 'logSFR_HEC': logSFR_HEC, 'METAL': METAL, 'AGN_HEC': AGN_HEC
    }
    df = pd.DataFrame([input_dict])
    
    # 2. Add Dummy Target and Error Columns (Needed by HW1 preprocessing functions)
    df['CLASS_SP'] = 0
    for err_col in cfg.err_vars:
        df[err_col] = 0.0001  # Small error to easily pass the > 3 ratio filter

    # 3. Preprocessing steps matching HW1 exactly
    df = apply_iqr_capping(df, iqr_bounds)
    df = apply_imputation(df, imputation_pipeline, target_col='CLASS_SP')
    df = filter_error_ratios(df, cfg.key_vars, cfg.err_vars)
    
    if len(df) == 0:
        return "Prediction failed: Input features did not pass preprocessing filters."

    df = compute_colors(df)
    
    # Apply standard scaling
    num_cols = scaler.feature_names_in_
    df = apply_scaling(df, scaler, num_cols, target_col='CLASS_SP')
    
    # Remove dummy target before prediction
    df.pop('CLASS_SP')
    
    # Ensure column order matches the trained model
    df = df[model.feature_names_in_]

    # 4. Prediction
    probs = model.predict_proba(df)[0]
    pred_idx = model.predict(df)[0]
    
    class_name = CLASS_NAMES.get(pred_idx, str(pred_idx))
    prob_percent = probs[pred_idx] * 100
    
    return f"Prediction: {class_name} (probability: {prob_percent:.1f}%)"
