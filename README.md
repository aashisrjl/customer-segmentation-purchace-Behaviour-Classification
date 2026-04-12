# Customer Segmentation Project

## Overview
This project performs **customer segmentation using K-Means clustering** and then
**classifies customers using Random Forest** based on the cluster labels.

## Folder Structure
```
customer-segmentation/
├── data/
│   ├── raw_data.csv          # Original dataset
│   ├── processed_data.csv    # After preprocessing
│   └── clustered_data.csv    # After clustering (with labels)
├── notebooks/
│   ├── 01_EDA.ipynb          # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_clustering.ipynb
│   └── 04_classification.ipynb
├── models/
│   ├── kmeans_model.pkl
│   └── classifier_model.pkl
├── plots/
│   ├── elbow_curve.png
│   └── cluster_scatter.png
├── requirements.txt
└── README.md
```

## How to Run
```bash
conda activate venv
pip install -r requirements.txt
jupyter notebook
```
Run notebooks in order: 01 → 02 → 03 → 04

## Dataset
Mall Customer Dataset — [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

## Tools
Python, scikit-learn, pandas, matplotlib, seaborn
