📄 Customer Segmentation using Clustering and Classification
Title Page

Project Title: Customer Segmentation using Clustering and Classification
Student Names:

Muna Kc
Aasish Risal
Aakash Dungana

Course: Bachelor in Information Technology
Institution: __________________________
Date: April 2026

Abstract

This project presents a machine learning-based approach to customer segmentation using clustering and classification techniques. K-Means clustering is applied to identify distinct customer groups based on demographic and behavioral features. The optimal number of clusters is determined using the Elbow Method and evaluated using the Silhouette Score. The generated cluster labels are then used as target variables to train a Random Forest classification model. The dataset is preprocessed through handling missing values, encoding categorical variables, and applying feature scaling. The classification model is evaluated using accuracy, confusion matrix, and feature importance analysis. The results demonstrate that meaningful customer segments can be identified and accurately predicted. This approach enables businesses to implement targeted marketing strategies and improve decision-making processes.

Table of Contents
Introduction
Literature Review
Dataset and Methodology
Exploratory Data Analysis
Data Preprocessing
Clustering (K-Means)
Classification (Random Forest)
Discussion
Conclusion
References
Appendices
1. Introduction
1.1 Background and Motivation

Customer segmentation is a critical task in marketing and business intelligence. It enables organizations to group customers based on similar characteristics, allowing businesses to target specific groups effectively and improve customer satisfaction.

1.2 Problem Statement

Traditional marketing approaches treat all customers equally, leading to inefficient targeting and resource utilization. This project addresses the need for data-driven segmentation using machine learning techniques.

1.3 Objectives
To apply K-Means clustering for customer segmentation
To build a classification model using cluster labels
To analyze customer behavior patterns
1.4 Report Structure

This report includes sections on literature review, dataset and methodology, exploratory data analysis, preprocessing, clustering, classification, discussion, and conclusion.

2. Literature Review
2.1 Customer Segmentation Overview

Customer segmentation is widely used in marketing analytics to divide customers into meaningful groups based on behavior and demographics.

2.2 Clustering Algorithms

K-Means clustering partitions data into K groups by minimizing intra-cluster variance. The Elbow Method is commonly used to determine the optimal number of clusters.

2.3 Classification Algorithms

Random Forest is an ensemble learning technique based on decision trees. It improves prediction accuracy and reduces overfitting.

2.4 Related Work

Previous studies and Kaggle projects demonstrate the effectiveness of clustering and classification techniques for customer analytics.

3. Dataset and Methodology
3.1 Dataset Description

The dataset consists of 8,068 customer records with features such as:

Gender
Age
Ever Married
Profession
Work Experience
Spending Score
Family Size
Other encoded variables
3.2 Tools and Environment
Python
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
Jupyter Notebook
3.3 Pipeline

Data Collection → Preprocessing → EDA → Clustering → Classification → Evaluation

4. Exploratory Data Analysis
4.1 Descriptive Statistics

Statistical measures such as mean, median, and standard deviation were analyzed for all features.

4.2 Data Visualization
Histograms for feature distribution
Correlation heatmaps
Pair plots for relationships
4.3 Observations
Spending behavior varies significantly across age groups
Some features show correlation with spending score
5. Data Preprocessing
5.1 Missing Values

Missing values were identified and handled appropriately.

5.2 Encoding

Categorical variables such as Gender were encoded using Label Encoding.

5.3 Feature Scaling

StandardScaler was applied to normalize data, which is essential for K-Means clustering.

6. Clustering (K-Means)
6.1 Algorithm Explanation

K-Means clustering groups data points into K clusters by minimizing the distance between points and cluster centroids.

6.2 Elbow Method

The Elbow Method was used to determine the optimal number of clusters.

Optimal K = 3

6.3 Cluster Results

Cluster distribution:

Cluster 0: 1463 customers
Cluster 1: 2693 customers
Cluster 2: 3912 customers
6.4 Cluster Interpretation
Cluster 0: Moderate customers with higher work experience
Cluster 1: Low-value customers with low spending behavior
Cluster 2: High-value customers with high spending score
6.5 Evaluation

Silhouette Score was used to evaluate clustering quality.

7. Classification (Random Forest)
7.1 Algorithm Explanation

Random Forest builds multiple decision trees and combines their predictions to improve accuracy.

7.2 Data Preparation

Cluster labels were used as the target variable.

7.3 Model Training
Train-test split: 80/20
n_estimators = 100
random_state = 42
7.4 Evaluation
Accuracy Score
Confusion Matrix
Classification Report

The model achieved high accuracy in predicting customer segments.

7.5 Feature Importance

Feature importance analysis showed that spending score, age, and work experience were key factors.

8. Discussion
8.1 Results Analysis

Clustering successfully identified meaningful customer segments, and classification accurately predicted these segments.

8.2 Limitations
Sensitivity of K-Means to outliers
Limited dataset features
8.3 Future Improvements
Use DBSCAN or hierarchical clustering
Hyperparameter tuning
Include more customer features
9. Conclusion

This project demonstrates the successful application of K-Means clustering and Random Forest classification for customer segmentation. The results show that machine learning techniques can effectively identify customer groups and predict their behavior. This approach can be applied in real-world business scenarios for targeted marketing and improved customer relationship management.

10. References
Breiman, L. (2001). Random Forests. Machine Learning Journal
Jain, A. K. (2010). Data Clustering: 50 Years Beyond K-Means
Scikit-learn Documentation
Kaggle Dataset Resources
Han, J., Kamber, M. (Data Mining Concepts and Techniques)
11. Appendices (Optional)
Code snippets
Elbow curve plot
Cluster visualization
Model outputs