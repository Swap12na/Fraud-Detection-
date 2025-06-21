# 🛡️ Credit Card Fraud Detection using AutoEncoder (PyOD)

This project implements an unsupervised fraud detection system using an AutoEncoder neural network model from the PyOD library. It is based on an anonymized credit card transaction dataset available on Kaggle and aims to detect fraudulent transactions by identifying anomalies in reconstruction error.

---

## 📁 Project Structure

📦 FraudDetection_AutoEncoder
┣ 📄 fraud_detection_autoencoder.py
┣ 📄 manifest.txt
┣ 📄 reconstruction_error.png
┣ 📄 output_screenshot.png
┗ 📄 README.md


---

## 📊 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Description**: Contains 284,807 anonymized credit card transactions over 2 days, with 492 fraud cases (highly imbalanced).
- **Features**: 30 total — including `Time`, `Amount`, and 28 anonymized principal components (`V1` to `V28`)
- **Label**: `Class` → `0` for legitimate, `1` for fraud

---

## 🧠 Model Details

- **Model Used**: `AutoEncoder` from [PyOD](https://pyod.readthedocs.io)
- **Type**: Unsupervised Anomaly Detection
- **Learning Goal**: Reconstruct normal transactions accurately, while fraudulent ones have high reconstruction errors

---

## 🚀 How to Run

### 🔧 Requirements

Install Python packages using pip:

```bash
pip install pandas numpy matplotlib scikit-learn pyod


python fraud_detection_autoencoder.py

This will:

Load and preprocess the dataset

Train the AutoEncoder model

Predict outliers (frauds)

Print evaluation metrics (Confusion Matrix, Classification Report, ROC-AUC)

Generate and save the reconstruction error histogram as reconstruction_error.png



📈 Evaluation Metrics
Confusion Matrix: Shows True Positive, False Positive, etc.

Classification Report: Precision, Recall, F1-Score

ROC-AUC Score: Evaluates model’s ability to distinguish fraud from normal

📘 References
PyOD Documentation

Scikit-learn Metrics

Kaggle Credit Card Dataset

✅ Author
Name: Swarna Anjani Devershetty
Course: Advance Artificial Intelligence

---



