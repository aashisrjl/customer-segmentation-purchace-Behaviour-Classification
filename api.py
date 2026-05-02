"""
FastAPI application for Customer Segmentation Classification Model
Endpoint: GET /predict?gender=...&ever_married=...&age=...&...
"""

import pickle
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pandas as pd

# Initialize FastAPI app
app = FastAPI(
    title="Customer Segmentation API",
    description="Real-time cluster prediction for customer data",
    version="1.0.0"
)

# Load trained models
kmeans_model = None
laptop_classifier = None
scaler = None
encoders = None

try:
    with open('./models/kmeans_model.pkl', 'rb') as f:
        kmeans_model = pickle.load(f)
    print("✓ KMeans model loaded (for clustering)")
except Exception as e:
    print(f"✗ Error loading KMeans: {e}")

try:
    with open('./models/laptop_classifier.pkl', 'rb') as f:
        laptop_classifier = pickle.load(f)
    print("✓ Laptop Classifier model loaded (for BuyLaptop prediction)")
except Exception as e:
    print(f"✗ Error loading Laptop Classifier: {e}")

try:
    with open('./models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded for preprocessing")
except Exception as e:
    print(f"✗ Error loading scaler: {e}")

try:
    with open('./models/encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    print("✓ Label encoders loaded for preprocessing")
except Exception as e:
    print(f"✗ Error loading encoders: {e}")

# Feature names (must match training data order)
FEATURES = ['Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession', 
            'Work_Experience', 'Spending_Score', 'Family_Size']


def _build_raw_row_from_query(gender, ever_married, age, graduated, profession, work_exp_val, spending_score, family_size):
    # Build a raw-row dict using query values (strings or numbers)
    return {
        'Gender': gender,
        'Ever_Married': ever_married,
        'Age': age,
        'Graduated': graduated,
        'Profession': profession,
        'Work_Experience': work_exp_val,
        'Spending_Score': spending_score,
        'Family_Size': family_size,
    }


def preprocess_exact(raw_row):
    """Use saved scaler and encoders to transform a single-row raw_row dict into model input."""
    if scaler is None or encoders is None:
        raise RuntimeError('Scaler and encoders not available')

    # Feature columns (excluding BuyLaptop target)
    feature_cols = ['Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession', 
                    'Work_Experience', 'Spending_Score', 'Family_Size']

    # Build DataFrame with raw values
    row = {}
    for col in feature_cols:
        val = raw_row.get(col)
        if val is None:
            raise ValueError(f'Missing value for {col}')
        row[col] = val

    raw_df = pd.DataFrame([row])

    # Extract mappings and encoder from config
    binary_map = encoders.get('binary_map', {})
    spending_map = encoders.get('spending_map', {})
    profession_encoder = encoders.get('profession_encoder')

    # Apply transformations matching preprocessing notebook
    
    # 1. Normalize categorical text (strip and capitalize)
    for col in ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score']:
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].astype(str).str.strip().str.capitalize()
    
    # 2. Apply Spending_Score mapping
    if 'Spending_Score' in raw_df.columns:
        raw_df['Spending_Score'] = raw_df['Spending_Score'].map(spending_map)
        if raw_df['Spending_Score'].isnull().any():
            raise ValueError(f"Unknown Spending_Score value. Must be one of: {list(spending_map.keys())}")
    
    # 3. Apply binary mappings for Gender, Ever_Married, Graduated
    for col in ['Gender', 'Ever_Married', 'Graduated']:
        if col in raw_df.columns:
            raw_df[col] = raw_df[col].map(binary_map)
            if raw_df[col].isnull().any():
                raise ValueError(f"Unknown {col} value. Must be one of: {list(binary_map.keys())}")
    
    # 4. Encode Profession using saved encoder
    if 'Profession' in raw_df.columns and profession_encoder is not None:
        raw_df['Profession'] = profession_encoder.transform(raw_df['Profession'].astype(str))
    
    # 5. Convert all to float and apply scaler
    X = raw_df[feature_cols].astype(float)
    X_scaled = scaler.transform(X)
    return X_scaled

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
    gender: str = Query(..., description="Gender (male/female)"),
    ever_married: str = Query(..., description="Ever_Married (yes/no)"),
    age: str = Query(..., description="Age (number)"),
    graduated: str = Query(..., description="Graduated (yes/no)"),
    profession: str = Query(..., description="Profession (string)"),
    work_experience: str = Query(None, alias="work_experience", description="Work_Experience (number)"),
    spending_score: str = Query(..., description="Spending_Score (Low/Average/High)"),
    family_size: str = Query(..., description="Family_Size (number)")
):
    """
    Predict customer cluster AND laptop purchase likelihood.
    
    Returns:
    - predicted_cluster: Customer segment (from KMeans)
    - predicted_buy_laptop: Whether customer will buy laptop (from Laptop Classifier)
    - buy_laptop_probability: Confidence score for "Yes" prediction
    - input_features: Echo of input parameters
    """
    
    if kmeans_model is None or laptop_classifier is None:
        return JSONResponse(
            status_code=500,
            content={"error": "Models not loaded"}
        )

    try:
        # Build raw row from query
        raw_row = _build_raw_row_from_query(gender, ever_married, age, graduated, profession, work_experience, spending_score, family_size)
        
        # Preprocess using saved scaler and encoders
        input_data = preprocess_exact(raw_row)

        # Predict cluster (KMeans) — unsupervised
        cluster_prediction = kmeans_model.predict(input_data)[0]
        
        # Predict laptop purchase (Binary Classification: 0=No, 1=Yes)
        laptop_pred_numeric = laptop_classifier.predict(input_data)[0]
        laptop_probabilities = laptop_classifier.predict_proba(input_data)[0]
        
        # Map 0/1 prediction to "No"/"Yes"
        laptop_pred_str = "Yes" if laptop_pred_numeric == 1 else "No"
        
        # Get probability for "Yes" (class 1)
        # Classes are [0, 1], so index 1 gives probability of "Yes"
        yes_prob = float(laptop_probabilities[1]) if 1 in laptop_classifier.classes_ else 0.0
        
        # Create response
        return {
            "status": "success",
            "predicted_cluster": int(cluster_prediction),
            "predicted_buy_laptop": laptop_pred_str,
            "buy_laptop_probability": yes_prob,
            "input_features": {
                "Gender": gender,
                "Ever_Married": ever_married,
                "Age": age,
                "Graduated": graduated,
                "Profession": profession,
                "Work_Experience": work_experience,
                "Spending_Score": spending_score,
                "Family_Size": family_size
            }
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Prediction error: {str(e)}"}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
