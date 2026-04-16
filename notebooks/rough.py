# =============================================================
#  rough.py — Clustering & Classification scratch code
#  Run after 02_preprocessing.ipynb has produced:
#    data/processed/processed_data.csv
# =============================================================

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

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# SECTION 1: CLUSTERING (K-Means)
# ------------------------------------------------------------------

df = pd.read_csv('../data/processed/processed_data.csv')
print('Processed data shape:', df.shape)

# --- Elbow Method ---
inertias = []
K_range  = range(2, 11)

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
plt.savefig('../plots/elbow_curve.png', dpi=150)
plt.show()

# --- Fit final K-Means ---
OPTIMAL_K = 4   # adjust after inspecting elbow curve

kmeans   = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=10)
clusters = kmeans.fit_predict(df)

print('Cluster distribution:')
print(pd.Series(clusters).value_counts().sort_index())

# --- PCA visualisation ---
pca = PCA(n_components=2, random_state=42)
pcs = pca.fit_transform(df)
vis = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
vis['Cluster'] = clusters

plt.figure(figsize=(8, 6))
sns.scatterplot(data=vis, x='PC1', y='PC2', hue='Cluster', palette='tab10', alpha=0.7, s=30)
plt.title(f'K-Means Clusters (K={OPTIMAL_K}) — PCA 2D')
plt.tight_layout()
plt.savefig('../plots/cluster_scatter.png', dpi=150)
plt.show()

# --- Save clustered data + model ---
raw = pd.read_csv('../data/raw/raw_data.csv')
raw['Cluster'] = clusters
raw.to_csv('../data/processed/clustered_data.csv', index=False)
print('Saved clustered_data.csv — shape:', raw.shape)

with open('../models/kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)
print('Saved kmeans_model.pkl')

# --- Cluster profiles ---
print('\nCluster Profiles:')
print(raw.groupby('Cluster')[['Age', 'Work_Experience', 'Family_Size']].mean().round(2))


# ------------------------------------------------------------------
# SECTION 2: CLASSIFICATION (Random Forest)
# ------------------------------------------------------------------

clustered = pd.read_csv('../data/processed/clustered_data.csv')
clustered  = clustered.drop(columns=['ID'], errors='ignore')

X_raw = clustered.drop(columns=['Cluster'])
y     = clustered['Cluster']

# Impute
for col in X_raw.select_dtypes(include='number').columns:
    X_raw[col].fillna(X_raw[col].median(), inplace=True)
for col in X_raw.select_dtypes(include='object').columns:
    X_raw[col].fillna(X_raw[col].mode()[0], inplace=True)

# Encode
spending_map = {'Low': 0, 'Average': 1, 'High': 2}
binary_map   = {'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0}
le = LabelEncoder()

X_raw['Spending_Score'] = X_raw['Spending_Score'].map(spending_map)
for col in ['Gender', 'Ever_Married', 'Graduated']:
    if col in X_raw.columns:
        X_raw[col] = X_raw[col].map(binary_map)
for col in X_raw.select_dtypes(include='object').columns:
    X_raw[col] = le.fit_transform(X_raw[col].astype(str))

# Scale
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)

# Train / Test split (80 / 20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'\nTrain: {X_train.shape}  |  Test: {X_test.shape}')

# Train RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
print(f'5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

# Evaluate
y_pred = clf.predict(X_test)
print(f'\nTest Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Cluster')
plt.ylabel('True Cluster')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Feature importance
importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.plot(kind='bar', color='steelblue', edgecolor='white', figsize=(8, 4))
plt.title('Feature Importances')
plt.tight_layout()
plt.show()
print(importances)

# Save model
with open('../models/classifier_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
print('\nSaved classifier_model.pkl')
