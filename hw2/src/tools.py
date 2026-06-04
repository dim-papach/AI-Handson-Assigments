import os
import joblib
import math
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Add hw1 and root to sys.path so we can import modules
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(REPO_DIR, "hw1"))
sys.path.append(REPO_DIR)

from src.config import PipelineConfig
from src.preprocessing import (
    apply_iqr_capping,
    apply_imputation,
    filter_error_ratios,
    compute_colors,
    apply_scaling
)

# Import the RAG retrieval function
from hw2.src.rag import retrieve_context

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
    WF2: Optional[float] = Field(default=None, description="4.6μm-band apparent magnitude")
    WF3: Optional[float] = Field(default=None, description="12μm-band apparent magnitude")
    WF4: Optional[float] = Field(default=None, description="22μm-band apparent magnitude")
    UT: float = Field(description="Total U-band apparent magnitude")
    BT: Optional[float] = Field(default=None, description="B-band apparent magnitude")
    VT: Optional[float] = Field(default=None, description="V-band apparent magnitude")
    IT: Optional[float] = Field(default=None, description="I-band apparent magnitude")
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
    WF2: Optional[float] = None, WF3: Optional[float] = None, WF4: Optional[float] = None,
    BT: Optional[float] = None, VT: Optional[float] = None, IT: Optional[float] = None
) -> str:
    """
    Predicts the nuclear activity classification of a galaxy (star-forming, Seyfert, LINER, composite).
    Input must be provided with numerical galaxy properties (T, WF1, UT, U, R, G, I, Z, logM_HEC, logSFR_HEC, METAL)
    and the categorical string AGN_HEC ('Y', 'N', '?'). Optional inputs are WF2, WF3, WF4, BT, VT, IT.
    """
    # 1. Create DataFrame
    input_dict = {
        'T': T, 'WF1': WF1, 'WF2': np.nan if WF2 is None else WF2, 'WF3': np.nan if WF3 is None else WF3, 'WF4': np.nan if WF4 is None else WF4,
        'UT': UT, 'BT': np.nan if BT is None else BT, 'VT': np.nan if VT is None else VT, 'IT': np.nan if IT is None else IT, 'U': U, 'R': R, 'G': G, 'I': I, 'Z': Z,
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

class RetrievalInput(BaseModel):
    query: str = Field(description="The factual or conceptual question to ask the domain knowledge base")

@tool("retrieve_domain_knowledge", args_schema=RetrievalInput)
def retrieve_domain_knowledge(query: str) -> str:
    """
    Calls the RAG retrieval function to fetch factual or conceptual domain knowledge.
    Use this tool when the user asks a factual or conceptual question about the domain literature, galaxy formation, or astrophysics concepts.
    """
    return retrieve_context(query)

class DatasetStatsInput(BaseModel):
    column: str = Field(description="The name of the column in the dataset to get statistics for (e.g. 'T', 'METAL', 'CLASS_SP', 'logM_HEC')")

@tool("dataset_stats", args_schema=DatasetStatsInput)
def dataset_stats(column: str) -> str:
    """
    Returns summary statistics for any numerical or categorical column in the HECATE galaxy dataset.
    Use this tool when the user asks for dataset statistics, averages, or distributions for specific features.
    Input: column name as a string.
    """
    dataset_path = os.path.join(REPO_DIR, "hw1", "data", "HECATE.csv")
    if not os.path.exists(dataset_path):
        return "Error: Dataset file not found."
        
    try:
        # Load only the requested column to save time/memory
        df = pd.read_csv(dataset_path, usecols=[column])
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            desc = df[column].describe()
            stats_str = (
                f"Statistics for {column}:\n"
                f"Count: {int(desc['count'])}\n"
                f"Mean: {desc['mean']:.4f}\n"
                f"Std: {desc['std']:.4f}\n"
                f"Min: {desc['min']}\n"
                f"25%: {desc['25%']}\n"
                f"50% (Median): {desc['50%']}\n"
                f"75%: {desc['75%']}\n"
                f"Max: {desc['max']}"
            )
            return stats_str
        else:
            # Categorical value counts
            counts = df[column].value_counts().to_dict()
            return f"Value counts for categorical column {column}:\n{counts}"
    except ValueError:
        return f"Error: Column '{column}' not found in the dataset. Please provide a valid column name."
    except Exception as e:
        return f"Error retrieving statistics: {str(e)}"

class CalculatorInput(BaseModel):
    expression: str = Field(description="A mathematical expression to evaluate (e.g. '3.14 * (5**2)', '10 / 2.54'). You can use standard python math functions like log10, exp, sqrt.")

@tool("calculator", args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression or unit conversion calculation.
    Use this tool when you need to perform calculations, convert units, or evaluate numerical formulas.
    """
    try:
        # Create a safe dictionary with math functions
        safe_dict = {k: v for k, v in math.__dict__.items() if not k.startswith('_')}
        result = eval(expression, {"__builtins__": None}, safe_dict)
        return f"Calculation Result: {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

class CSVLookupInput(BaseModel):
    query: str = Field(description="A pandas query string to filter the dataset (e.g. 'T > 5 and AGN_HEC == \"Y\"'). Note: categorical values must be in quotes.")
    max_results: int = Field(default=5, description="Maximum number of rows to return (default 5).")
    sort_by: Optional[str] = Field(default=None, description="Optional column name to sort the results by (e.g., 'D' for distance).")
    ascending: bool = Field(default=True, description="If sorting, whether to sort in ascending order (default True). True finds the minimum/closest, False finds the maximum.")

@tool("csv_lookup", args_schema=CSVLookupInput)
def csv_lookup(query: str, max_results: int = 5, sort_by: Optional[str] = None, ascending: bool = True) -> str:
    """
    Queries the HECATE dataset to find specific rows matching the given criteria, optionally sorting the results.
    Use this when the user asks for examples of galaxies with specific properties, or wants to find the 'closest', 'largest', 'highest', etc.
    """
    dataset_path = os.path.join(REPO_DIR, "hw1", "data", "HECATE.csv")
    if not os.path.exists(dataset_path):
        return "Error: Dataset file not found."
    try:
        df = pd.read_csv(dataset_path)
        filtered_df = df.query(query)
        
        if filtered_df.empty:
            return "No matching records found for the given query."
            
        if sort_by and sort_by in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by=sort_by, ascending=ascending)
        
        # We only return the requested max_results, converting to a list of dicts for readability
        res = filtered_df.head(max_results).to_dict(orient="records")
        return f"Found {len(filtered_df)} matches. Showing top {min(len(res), max_results)}:\n{res}"
    except Exception as e:
        return f"Error executing query: {str(e)}"

