# =============================================================
#  rough.py — Clustering & Classification scratch code
#  Run after 02_preprocessing.ipynb has produced:
#    data/processed/processed_data.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path

warnings.filterwarnings('ignore')


# File paths
BASE = Path(__file__).resolve().parents[1]
RAW_CSV = BASE / 'data' / 'raw' / 'raw_data.csv'
PROCESSED_CSV = BASE / 'data' / 'processed' / 'processed_data.csv'
CLUSTERED_CSV = BASE / 'data' / 'processed' / 'clustered_data.csv'
MODELS_DIR = BASE / 'models'
PLOTS_DIR = BASE / 'plots'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def run_elbow(df, k_min=2, k_max=10):
    inertias = []
    K_range = range(k_min, k_max + 1)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(df)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=7)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Curve')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'elbow_curve.png', dpi=150)
    plt.close()
    return inertias


def fit_kmeans(df, k=4, save_path=MODELS_DIR / 'kmeans_model.pkl'):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = km.fit_predict(df)
    with open(save_path, 'wb') as f:
        pickle.dump(km, f)
    return km, clusters


def save_clustered_raw(clusters, raw_path=RAW_CSV, out_path=CLUSTERED_CSV):
    raw = pd.read_csv(raw_path)
    raw['Cluster'] = clusters
    raw.to_csv(out_path, index=False)
    return raw


def preprocess_for_classification(clustered_path=CLUSTERED_CSV):
    clustered = pd.read_csv(clustered_path)
    clustered = clustered.drop(columns=['ID'], errors='ignore')

    X_raw = clustered.drop(columns=['Cluster'])
    y = clustered['Cluster']

    # Impute
    for col in X_raw.select_dtypes(include=[np.number]).columns:
        X_raw[col].fillna(X_raw[col].median(), inplace=True)
    for col in X_raw.select_dtypes(include=['object']).columns:
        X_raw[col].fillna(X_raw[col].mode()[0], inplace=True)

    # Map known categorical values first
    spending_map = {'Low': 0, 'Average': 1, 'High': 2}
    binary_map = {'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0}

    if 'Spending_Score' in X_raw.columns:
        X_raw['Spending_Score'] = X_raw['Spending_Score'].map(spending_map).astype(float)

    for col in ['Gender', 'Ever_Married', 'Graduated']:
        if col in X_raw.columns:
            X_raw[col] = X_raw[col].map(binary_map).astype(float)

    # Label-encode remaining object columns and save encoders
    encoders = {}
    for col in X_raw.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_raw[col] = le.fit_transform(X_raw[col].astype(str))
        encoders[col] = le

    # Scale
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)

    # Save scaler + encoders
    with open(MODELS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(MODELS_DIR / 'encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    return X, y, X_raw.columns.tolist()


def train_classifier(X, y, save_path=MODELS_DIR / 'classifier_model.pkl'):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    print(f'5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

    y_pred = clf.predict(X_test)
    print(f'\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}')

    with open(save_path, 'wb') as f:
        pickle.dump(clf, f)
    return clf


def load_models():
    models = {}
    with open(MODELS_DIR / 'classifier_model.pkl', 'rb') as f:
        models['clf'] = pickle.load(f)
    with open(MODELS_DIR / 'kmeans_model.pkl', 'rb') as f:
        models['kmeans'] = pickle.load(f)
    with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
        models['scaler'] = pickle.load(f)
    with open(MODELS_DIR / 'encoders.pkl', 'rb') as f:
        models['encoders'] = pickle.load(f)
    return models


def predict_from_strings(input_dict):
    """Accepts a dict of user-friendly string inputs and returns
    both classifier prediction and kmeans cluster for the same processed sample.

    Example input_dict keys:
      Gender, Ever_Married, Age, Graduated, Profession, Work_Experience, Spending_Score, Family_Size
    Values can be raw strings (e.g., 'male', 'Yes', 'Engineer') or numeric strings.
    """
    # Load models
    models = load_models()
    clf = models['clf']
    kmeans = models['kmeans']
    scaler = models['scaler']
    encoders = models['encoders']

    # Build single-row DataFrame matching training raw columns
    # We rely on the encoders/scaler saved earlier for exact transforms
    # Start with a copy of processed DataFrame columns order saved earlier
    # For simplicity, read processed CSV columns before scaling
    processed_sample = pd.read_csv(PROCESSED_CSV, nrows=1)
    feature_cols = processed_sample.columns.tolist()

    # Build raw dict
    raw_row = {}
    # Basic maps
    spending_map = {'low': 0, 'average': 1, 'high': 2}
    binary_map = {'male': 1, 'female': 0, 'yes': 1, 'no': 0}

    for col in feature_cols:
        val = input_dict.get(col) or input_dict.get(col.lower()) or input_dict.get(col.replace('_', ' ').lower())
        if val is None:
            raise ValueError(f'Missing value for {col}')
        sval = str(val).strip()
        # Numeric first
        try:
            raw_row[col] = float(sval)
            continue
        except Exception:
            pass

        low = sval.lower()
        if col == 'Spending_Score':
            raw_row[col] = spending_map.get(low, sval)
        elif col in ['Gender', 'Ever_Married', 'Graduated']:
            raw_row[col] = binary_map.get(low, sval)
        else:
            raw_row[col] = sval

    raw_df = pd.DataFrame([raw_row])

    # Apply encoders for object cols
    for col, le in encoders.items():
        raw_df[col] = le.transform(raw_df[col].astype(str))

    # Ensure numeric columns exist and fillna if needed
    for col in raw_df.columns:
        if raw_df[col].dtype == object:
            try:
                raw_df[col] = raw_df[col].astype(float)
            except Exception:
                pass

    X_scaled = pd.DataFrame(scaler.transform(raw_df), columns=raw_df.columns)

    clf_pred = int(clf.predict(X_scaled)[0])
    kmeans_pred = int(kmeans.predict(X_scaled)[0])

    return {'classifier_pred': clf_pred, 'kmeans_pred': kmeans_pred}


def main():
    # Run clustering
    df = pd.read_csv(PROCESSED_CSV)
    print('Processed data shape:', df.shape)
    run_elbow(df)
    k_optimal = 4
    km, clusters = fit_kmeans(df, k=k_optimal)
    print('Cluster distribution:')
    print(pd.Series(clusters).value_counts().sort_index())
    # Visualize PCA
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(df)
    vis = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
    vis['Cluster'] = clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=vis, x='PC1', y='PC2', hue='Cluster', palette='tab10', alpha=0.7, s=30)
    plt.title(f'K-Means Clusters (K={k_optimal}) — PCA 2D')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / 'cluster_scatter.png', dpi=150)
    plt.close()

    raw = save_clustered_raw(clusters)
    print('Saved clustered_data.csv — shape:', raw.shape)

    # Classification pipeline
    X, y, cols = preprocess_for_classification()
    clf = train_classifier(X, y)
    print('\nSaved classifier and scaler/encoders in', MODELS_DIR)


if __name__ == '__main__':
    main()
