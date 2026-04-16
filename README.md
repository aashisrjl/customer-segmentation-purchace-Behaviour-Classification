# Customer Segmentation Project

## Overview
This project performs **customer segmentation using K-Means clustering** and then
**classifies customers using Random Forest** based on the cluster labels.

## Folder Structure
```
customer-segmentation/
├── data/
│   ├── raw/
│   │   └── raw_data.csv          # Raw data — 8,068 customers, 10 features (no labels)
│   └── processed/
│       ├── processed_data.csv    # After preprocessing (scaled, encoded)
│       └── clustered_data.csv    # After clustering (with Cluster labels)
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb   # Impute, encode, scale → processed_data.csv
│   ├── 03_clustering.ipynb      # K-Means + elbow → clustered_data.csv
│   └── 04_classification.ipynb  # Train/test split → Random Forest
├── models/
│   ├── kmeans_model.pkl
│   └── classifier_model.pkl
├── plots/
│   ├── elbow_curve.png
│   └── cluster_scatter.png
├── requirements.txt
└── README.md
```

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
