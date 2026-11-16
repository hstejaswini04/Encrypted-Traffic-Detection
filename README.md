# Detecting Malicious Encrypted Traffic Without Decrypting Packets

## Project Overview

A machine learning-based system for detecting malicious encrypted
DNS-over-HTTPS (DoH) network traffic using statistical flow features,
without decrypting packet payloads.

## Key Features

- Encrypted DNS-over-HTTPS traffic analysis
- Statistical network flow feature analysis
- Random Forest classification
- Model testing and validation
- Confusion matrix and performance metrics
- Explainable AI using feature importance
- Record-level prediction verification
- Interactive Streamlit dashboard
- CSV-based analysis and reporting

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Streamlit
- Plotly
- Joblib
- Google Colab

## Project Structure

    Encrypted-Traffic-Detection/
        application/
            app.py
        notebooks/
            01_Data_Download_and_Setup.ipynb
            02_Data_Exploration_and_Audit.ipynb
            03_Final_Model_Testing.ipynb
            04_Streamlit_Dashboard.ipynb
            random_forest_aligned_28f.pkl
            random_forest_doh_model.pkl
        datasets/
            README.md
        requirements.txt
        README.md

## Dataset

The project uses DNS-over-HTTPS traffic datasets.

External testing uses the BCCC-CIRA-CIC-DoHBrw-2020 dataset.

Large CSV datasets are excluded from GitHub.

## Machine Learning

A Random Forest classifier analyzes 28 statistical network flow
features to identify benign and malicious traffic.

Evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Streamlit Dashboard

The dashboard provides:

1. Dataset Overview
2. Dataset Validation
3. Traffic Explorer
4. Flow Behaviour Analysis
5. Threat Detection Centre
6. Explainable AI
7. Model Performance
8. Reports and Export

## Limitations

- The application analyzes CSV flow features, not live packet captures.
- Dataset provenance supports the DoH traffic context, but encryption
  has not been independently verified for every record.
- The system does not decrypt packets or block network traffic.

## Project Team

- Tejaswini
- Sukanya
- Sripriya
- Suchitra
