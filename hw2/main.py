import os
import sys
import uvicorn

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    print("Starting HECATE API server...")
    uvicorn.run("hw2.src.api:app", host="0.0.0.0", port=8000, reload=True)
