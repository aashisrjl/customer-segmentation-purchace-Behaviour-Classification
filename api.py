"""
FastAPI application for Customer Segmentation Classification Model
Endpoint: GET /predict?gender=...&ever_married=...&age=...&...
"""

import pickle
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="Customer Segmentation API",
    description="Real-time cluster prediction for customer data",
    version="1.0.0"
)

# Load trained model
try:
    with open('./models/classifier_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

# Feature names (must match training data order)
FEATURES = ['Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession', 
            'Work_Experience', 'Spending_Score', 'Family_Size']


def parse_numeric_or_map(value, mapper=None, name=None):
    """Try to parse a float; if fails and mapper provided, map string -> float.
    Returns float or raises ValueError.
    """
    try:
        return float(value)
    except Exception:
        if mapper is None:
            raise ValueError(f"Field {name} expects a numeric value or mapped string")
        key = str(value).strip().lower()
        if key in mapper:
            return float(mapper[key])
        raise ValueError(f"Unrecognized value for {name}: {value}")


# Simple mappings — adjust as needed to match your preprocessing
GENDER_MAP = {"male": 0.9091639709665634, "female": -0.9091639709665634}
YESNO_MAP = {"yes": 1.0, "no": 0.0}
SPENDING_MAP = {"low": -0.7368331046002994, "average": 0.0, "high": 1.0}
PROFESSION_MAP = {
    "engineer": 0.8866068253308166,
    "healthcare": 0.5,
    "doctor": 0.7,
    "artist": -0.2,
    "lawyer": 0.2,
    "homemaker": -0.5,
    "executive": 0.3,
    "marketing": -0.1,
    "entertainment": 0.0,
}

@app.get("/")
def home():
    """Home endpoint with API documentation"""
    return {
        "status": "API Running",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/predict")
def predict(
    gender: str = Query(..., description="Gender (normalized or raw: male/female)"),
    ever_married: str = Query(..., description="Ever_Married (normalized or raw: yes/no)"),
    age: str = Query(..., description="Age (normalized or raw number)"),
    graduated: str = Query(..., description="Graduated (normalized or raw: yes/no)"),
    profession: str = Query(..., description="Profession (normalized or raw string)"),
    work_experience: str = Query(None, alias="work_experience", description="Work_Experience (normalized or raw number)"),
    work_Exprerience: str = Query(None, alias="work_Exprerience", description="Alias accepted for work_experience"),
    spending_score: str = Query(..., description="Spending_Score (normalized or raw: Low/Average/High)"),
    family_size: str = Query(..., description="Family_Size (normalized or raw number)")
):
    """
    Predict customer cluster based on input features.
    
    All features should be normalized (scaled) values.
    Example: /predict?gender=0.9&ever_married=-1.2&age=-1.3&graduated=-1.3&profession=0.9&work_experience=-0.5&spending_score=-0.7&family_size=0.8
    """
    
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Model not loaded"}
        )

    # choose work_experience from either alias if provided
    work_exp_val = work_experience if work_experience is not None else work_Exprerience

    try:
        # Parse / map inputs to numeric floats (backwards compatible)
        g = parse_numeric_or_map(gender, mapper=GENDER_MAP, name='gender')
        em = parse_numeric_or_map(ever_married, mapper=YESNO_MAP, name='ever_married')
        age_val = parse_numeric_or_map(age, mapper=None, name='age')
        grad = parse_numeric_or_map(graduated, mapper=YESNO_MAP, name='graduated')
        prof = parse_numeric_or_map(profession, mapper=PROFESSION_MAP, name='profession')
        we = parse_numeric_or_map(work_exp_val, mapper=None, name='work_experience')
        ss = parse_numeric_or_map(spending_score, mapper=SPENDING_MAP, name='spending_score')
        fs = parse_numeric_or_map(family_size, mapper=None, name='family_size')

        # Create feature vector
        input_data = np.array([g, em, age_val, grad, prof, we, ss, fs], dtype=float).reshape(1, -1)

        # Predict cluster
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Create response
        return {
            "status": "success",
            "predicted_cluster": int(prediction),
            "cluster_probabilities": {
                f"cluster_{i}": float(prob) 
                for i, prob in enumerate(probabilities)
            },
            "input_features": {
                feat: val for feat, val in zip(FEATURES, [
                    gender, ever_married, age, graduated, profession,
                    work_experience, spending_score, family_size
                ])
            }
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Prediction error: {str(e)}"}
        )

@app.get("/example")
def example():
    """Returns example data and how to use the API"""
    return {
        "description": "Example request to /predict endpoint",
        "example_request": "/predict?gender=0.91&ever_married=-1.21&age=-1.29&graduated=-1.29&profession=0.89&work_experience=-0.46&spending_score=-0.74&family_size=0.81",
        "features_info": {
            "Gender": "Normalized binary/categorical value",
            "Ever_Married": "Normalized binary/categorical value",
            "Age": "Normalized continuous value",
            "Graduated": "Normalized binary/categorical value",
            "Profession": "Normalized categorical value",
            "Work_Experience": "Normalized continuous value",
            "Spending_Score": "Normalized categorical value",
            "Family_Size": "Normalized continuous value"
        },
        "response_example": {
            "status": "success",
            "predicted_cluster": 2,
            "cluster_probabilities": {
                "cluster_0": 0.15,
                "cluster_1": 0.25,
                "cluster_2": 0.35,
                "cluster_3": 0.15,
                "cluster_4": 0.10
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
