import sys
import os
# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hw2.src.tools import predict_galaxy_class

def test_prediction():
    print("Testing Prediction Tool (HW1 Model Wrapper)...\n")
    
    # We provide a dictionary with mock values for a typical galaxy
    input_args = {
        "T": 6.0, 
        "WF1": 12.5, 
        "UT": 14.2, 
        "U": 15.1, 
        "R": 14.3, 
        "G": 14.8, 
        "I": 14.0, 
        "Z": 13.9,
        "logM_HEC": 10.5, 
        "logSFR_HEC": 0.5, 
        "METAL": 0.02, 
        "AGN_HEC": "Y"
    }
    
    print(f"Input features provided to tool:\n{input_args}\n")
    
    # Call the tool directly
    # Note: Because predict_galaxy_class is decorated with @tool, we need to call .invoke()
    result = predict_galaxy_class.invoke(input_args)
    
    print("=== Tool Output ===")
    print(result)
    print("===================")

if __name__ == "__main__":
    test_prediction()
