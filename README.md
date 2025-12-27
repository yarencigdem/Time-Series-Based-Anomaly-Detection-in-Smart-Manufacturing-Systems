📊 Time Series–Based Anomaly Detection in Smart Manufacturing Systems
Comparative Analysis of Machine Learning and Deep Learning Models
📌 Project Overview

In the Industry 4.0 era, smart manufacturing systems generate massive amounts of multivariate time-series sensor data. Detecting anomalies in these data streams is critical for predictive maintenance, fault prevention, and production continuity.
This project presents a comparative analysis of classical machine learning and deep learning models for time-series anomaly detection in industrial environments.

The study focuses on evaluating the practical applicability vs. theoretical potential of different anomaly detection approaches using the Skoltech Anomaly Benchmark (SKAB) dataset.

🎯 Objectives

Compare classical and deep learning–based anomaly detection methods

Analyze the trade-off between:

F1-Score / Recall (practical usability)

ROC-AUC (theoretical separation capability)

Investigate the impact of static thresholding on deep learning models

Provide insights for real-time industrial deployment

🗂 Dataset

Skoltech Anomaly Benchmark (SKAB)

Multivariate industrial sensor time-series

Includes:

Vibration (Accelerometer RMS)

Electric current

Pressure

Temperature

Voltage

Flow rate

Anomaly types:

Point anomalies (outliers)

Collective anomalies (changepoints)

🔗 Dataset source: https://github.com/waico/SKAB

🤖 Models Used
Classical Machine Learning

Isolation Forest (IF)

One-Class Support Vector Machine (OCSVM)

Deep Learning

LSTM Autoencoder (AE)

LSTM Predictive Model

Each model represents a different philosophy:

Classical methods → fast, stable, low-cost

Deep learning methods → temporal modeling, high theoretical potential

📐 Methodology

Data preprocessing

Merging time-series files

Handling missing values

Min–Max normalization

Model training

Classical models optimized via contamination ratio

Deep models trained on normal behavior only

Anomaly scoring

Reconstruction error (AE)

Prediction error (LSTM-P)

Decision threshold

Static threshold (95th percentile)

Evaluation

📊 Evaluation Metrics

Precision

Recall

F1-Score (primary metric)

ROC-AUC (threshold-independent performance)

F1-Score is emphasized due to severe class imbalance in anomaly detection tasks.

🔍 Key Findings

Isolation Forest

Best practical performance

F1-Score: 0.4536

Recall: 0.5182

Low computational cost → suitable for real-time systems

LSTM Autoencoder

Highest ROC-AUC: 0.6633

Very low Recall due to static thresholding

Demonstrates high theoretical potential but weak practical performance

Static thresholding caused a large number of false negatives in deep learning models

💡 Conclusions

Classical models remain highly competitive for industrial deployment

Deep learning models require:

Adaptive thresholding

Advanced optimization strategies

A hybrid anomaly detection architecture is recommended:

Classical models for stability

Deep models for complex temporal patterns

🔮 Future Work

Dynamic and adaptive thresholding

Bayesian hyperparameter optimization

Attention-based LSTM architectures

Hybrid ensemble anomaly detection systems

🛠 Technologies

Python

Scikit-learn

TensorFlow / Keras

NumPy, Pandas

Matplotlib / Seaborn

📎 Academic Context

This project was developed as part of an academic study on time-series anomaly detection for smart manufacturing systems and is suitable for:

Research

Industrial experimentation

Graduate-level machine learning studies
