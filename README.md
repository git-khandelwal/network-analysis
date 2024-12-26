
# Network Firewall Analysis & Prediction

This project focuses on analyzing a network firewall dataset, transforming the features, and predicting the **action** (e.g., allow/deny) using a neural network. A simple frontend interface is also provided to demonstrate how new data can be fed into the model for predictions.

---

## Table of Contents
1. [Overview](#overview)  
2. [Project Structure](#project-structure)  
3. [Getting Started](#getting-started)  
4. [Data Exploration (EDA)](#data-exploration-eda)  
5. [Model Pipeline](#model-pipeline)  
6. [Neural Network Training](#neural-network-training)  
7. [Evaluation](#evaluation)  
8. [Frontend Application](#frontend-application)  

---

## Overview

- **Goal**: Build a pipeline that processes network firewall data and predicts an “action” (e.g., allow/deny) with high accuracy.  
- **Dataset**: A network firewall dataset with multiple features representing network traffic characteristics and a binary label (action).  The dataset can be viewed [here](https://www.kaggle.com/datasets/tunguz/internet-firewall-data-set).
- **Key Steps**:
  - Perform Exploratory Data Analysis (EDA) to understand the data distribution and key features.
  - Create an end-to-end scikit-learn pipeline to handle data transformations.
  - Train a Neural Network on the transformed data.
  - Achieve a test accuracy of **99.53%** after 5 epochs of training.
  - Develop a simple **frontend** to let users input feature values and get real-time predictions.

---

## Project Structure
network-analysis/
├── templates/
│   └── form.html
├── app.py
├── dataset.py
├── eda.ipynb
├── log2.csv
├── model.py
├── my_pipeline.pkl
├── pipe.py
├── predict.py
├── preprocess.py
├── train.py
└── trained_weights.pth

## Data Exploration (EDA)

1. **Data Loading & Cleaning**  
   - Read the dataset (`log2.csv`) into a pandas DataFrame.
   - Check for missing values and remove or impute them if necessary.
   - Analyze outliers and decide whether to remove or cap them.
   - Normalize the numerical columns for better training.

2. **Basic Statistics**  
   - Use `df.describe()` and `df.info()` to understand the data’s shape and distribution.
   - Identify potential correlations using a correlation matrix.

3. **Visualization**  
   - Plot histograms and boxplots to get insights into feature distributions.
   - Use correlation plots to see how features relate to one another and to the target.

4. **Feature Insights**  
   - Select or engineer features that capture important properties of the network traffic.
   - Categorizing ports into well-known, registered, dynamic to get further insights on when the firewall is allowing/denying traffic.

## Model Pipeline

1. **Log Scaling**: Numeric columns are transformed using a `LogScaleTransformer` for better handling of skewed data.
2. **Categorize Ports**: Port columns (e.g., `Source Port`, `Destination Port`) are mapped to meaningful categories.
3. **One-Hot Encoding**: A `ColumnTransformer` applies `OneHotEncoder` to these port-related columns, while passing other columns through unchanged.
4. **Combined Pipeline**: By chaining these steps in a single `Pipeline`, the same transformations are applied consistently during both training and inference. Save and load this pipeline as `my_pipeline.pkl` for future inference or training.

## Neural Network Training
- **Architecture**: Define a simple feedforward model (e.g., input layer → hidden layer(s) → output layer).
- **Training Loop**: Use a library like PyTorch or TensorFlow, and run for a fixed number of epochs (e.g., 5).
- **Loss & Optimizer**: Typically `BCELoss`/`CrossEntropyLoss` for classification, optimized with Adam or SGD.
- **Model Persistence**: Save the trained weights (e.g., `trained_weights.pth`) for later inference.

## Evaluation
- **Test Data**: Split a portion of the dataset for unbiased performance measurement.
- **Metrics**: Calculate Accuracy, Precision, Recall, and F1-score to gauge classification performance.
- **Threshold**: Convert model outputs (probabilities) into binary predictions (e.g., “Allow” vs “Deny”).
- **Results**: Achieved a 99.59% accuracy on the test set with minimal overfitting. Here are the classification metrics:
     
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      9427
           1       1.00      0.98      0.99      3756
           2       0.98      1.00      0.99      3190
           3       1.00      0.20      0.33        10


## Frontend
- **Form Input**: An HTML page (`form.html`) to collect user input (e.g., port values, packet sizes).
- **Flask Backend**: A simple `app.py` that processes form data, applies the pipeline, and calls the trained model.
- **Real-Time Prediction**: User inputs -> preprocessed features -> NN model output -> displayed result (e.g., “Allow” / “Deny”).
- **Local Deployment**: Run `pip install -r requirements.txt`, then  `python app.py` and open `http://127.0.0.1:5000/` in your browser to test.

