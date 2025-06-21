#!/usr/bin/env python
# coding: utf-8

# In[7]:


# fraud_detection_autoencoder.py
# Author: Swarna Anjani Devershetty
# Description: Fraud detection using AutoEncoder (Unsupervised Learning) with PyOD

# -----------------------------------------------
# Import required libraries
# -----------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from pyod.models.auto_encoder import AutoEncoder

# -----------------------------------------------
# Load the dataset
# -----------------------------------------------
print("Loading dataset...")
df = pd.read_csv("creditcard.csv")

# -----------------------------------------------
# Data preprocessing
# -----------------------------------------------
print("Preprocessing data...")
X = df.drop("Class", axis=1)
y = df["Class"]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------
# Define and train the AutoEncoder model
# -----------------------------------------------
print("Training AutoEncoder model...")

# Use default constructor — no unsupported parameters
model = AutoEncoder(verbose=1)
model.fit(X_scaled)

# -----------------------------------------------
# Make predictions and evaluate
# -----------------------------------------------
print("Making predictions...")
y_pred = model.labels_              # 0 = inlier, 1 = outlier
y_scores = model.decision_scores_   # Raw outlier scores

print("\nEvaluation Metrics:")
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

print("\nClassification Report:")
print(classification_report(y, y_pred, digits=4))

print("\nROC-AUC Score:")
print("ROC-AUC:", round(roc_auc_score(y, y_scores), 4))

# -----------------------------------------------
# Visualization
# -----------------------------------------------
print("\nVisualizing reconstruction error distribution...")
plt.figure(figsize=(10, 6))
plt.hist(y_scores, bins=50, color="skyblue", edgecolor="black")
plt.title("Reconstruction Error Distribution")
plt.xlabel("Reconstruction Error Score")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig("reconstruction_error.png")
plt.show()

print("\nExperiment completed. Output and graph saved.")

