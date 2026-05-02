# Customer Segmentation & Laptop Purchase Prediction

## Overview

This project performs **dual machine learning tasks**:

1. **Customer Segmentation** using K-Means clustering (5 clusters)
2. **Laptop Purchase Prediction** using Random Forest binary classification

The system consists of:
- **Data Pipeline**: Raw data → preprocessing → clustering & classification
- **REST API**: Real-time predictions with user-friendly string inputs
- **Models**: Trained and saved for production use

## Folder Structure

```
customer-segmentation/
├── data/
│   ├── raw/
│   │   └── raw_data.csv              # Original data + BuyLaptop field
│   └── processed/
│       ├── processed_data.csv        # Scaled & encoded features (8 cols)
│       └── clustered_data.csv        # Raw data + Cluster assignments
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb        # Impute, encode, scale
│   ├── 03_clustering.ipynb          # K-Means + elbow curve
│   ├── 04_classification.ipynb      # Binary classifier (BuyLaptop)
│   └── rough.py                      # Refactored ML pipeline functions
├── models/
│   ├── kmeans_model.pkl              # Clustering model (k=5)
│   ├── laptop_classifier.pkl         # Binary classifier (Yes/No)
│   ├── scaler.pkl                    # Feature scaling
│   └── encoders.pkl                  # Categorical encoders
├── plots/
│   ├── elbow_curve.png
│   ├── cluster_visualizations/
│   └── preprocessing_plots/
├── api.py                            # FastAPI application
├── API_USAGE.md                      # API documentation & examples
├── requirements.txt
└── README.md
```

## Data

### Input Features (8 features)

| Feature | Type | Values |
|---------|------|--------|
| Gender | Categorical | Male, Female |
| Ever_Married | Categorical | Yes, No |
| Age | Numeric | 18-80 years |
| Graduated | Categorical | Yes, No |
| Profession | Categorical | Engineer, Doctor, Lawyer, Artist, etc. (10 types) |
| Work_Experience | Numeric | 0-14 years |
| Spending_Score | Categorical | Low, Average, High |
| Family_Size | Numeric | 1-8 members |

### Target Variables

#### 1. Cluster (K-Means)
- **Clusters**: 0, 1, 2, 3, 4
- **Algorithm**: K-Means with k=5
- **Quality Metrics**:
  - Silhouette Score: 0.2013
  - Davies-Bouldin Index: 1.7719
  - Calinski-Harabasz Index: 1549.29

#### 2. BuyLaptop (Binary Classifier)
- **Target**: "Yes" or "No"
- **Distribution**: 20.23% Yes, 79.77% No
- **Algorithm**: Random Forest (100 trees)
- **Performance**: F1-Score 0.9988

### Laptop Purchase Logic

A customer is predicted to buy a laptop if:

1. **Spending_Score = "High"** → Always Yes
2. **Spending_Score = "Average"** AND **Profession ∈ {Engineer, Doctor, Lawyer}** → Yes
3. **All other cases** → No

## REST API

### Quick Start

```bash
# Start the API server
cd /home/aashis-rijal/Desktop/customer-segmentation
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Predict Endpoint

**GET** `/predict?gender=male&ever_married=yes&age=40&graduated=yes&profession=Engineer&work_experience=5&spending_score=Average&family_size=3`

### Response

```json
{
    "status": "success",
    "predicted_cluster": 1,
    "predicted_buy_laptop": "Yes",
    "buy_laptop_probability": 0.96,
    "input_features": { ... }
}
```

### Additional Endpoints

- `/segment-info?cluster=<id>` - Get cluster profile (size, means, top professions)
- `/example` - View example requests and responses
- `/docs` - Swagger UI documentation
- `/redoc` - ReDoc documentation

**See [API_USAGE.md](API_USAGE.md) for complete API documentation.**

## Model Performance

### Clustering (KMeans, k=5)

| Metric | Value |
|--------|-------|
| Silhouette Score | 0.2013 |
| Davies-Bouldin Index | 1.7719 |
| Calinski-Harabasz Score | 1549.29 |

**Interpretation**: Clusters are loosely separated; consider alternative k or different features.

### Classification (Laptop Purchase)

| Metric | Value |
|--------|-------|
| F1-Score (5-fold CV) | 0.9988 ± 0.0009 |
| Training Accuracy | 100% |
| Class Distribution | 20% Yes, 80% No |

**Top Features by Importance**:
1. Spending_Score (53.19%)
2. Profession (21.58%)
3. Ever_Married (8.80%)
4. Age (8.38%)
5. Family_Size (4.52%)

## Pipeline
```
data/raw/raw_data.csv
       ↓
  01 EDA — understand data
       ↓
  02 Preprocessing — clean, encode, scale → processed_data.csv
       ↓
  03 Clustering (K-Means) — segment customers → clustered_data.csv
       ↓
  04 Classification (Random Forest) — train/test split → predict Cluster
```

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook
```
Run notebooks **in order**: `01 → 02 → 03 → 04`

## Dataset
Source: `data/raw/raw_data.csv` — 8,068 customers, 10 features

## Tools
Python, scikit-learn, pandas, matplotlib, seaborn
