# Fraud-Detection-System

Welcome to the Credit Card Fraud Detection project! This repository contains the code and resources for building, training, and evaluating a machine learning model to detect fraudulent credit card transactions.

#Table of Contents

Project Overview
Dataset
Usage
Model Training and Evaluation
Results

#Project Overview
The objective of this project is to identify fraudulent credit card transactions using a Logistic Regression model. The model is trained on a dataset of transactions labeled as either fraudulent or legitimate, and it aims to accurately predict the fraud status of new transactions.

#Dataset 
Link-www.kaggle.com/datasets/mlg-ulb/creditcardfraud

#Usage
To use this project, follow these steps:

Download the dataset: Ensure you have the dataset downloaded from Kaggle and placed in directory.
Preprocess the data: Run the data preprocessing script to clean and prepare the data for training.
python preprocess_data.py
Train the model: Execute the training script to train the machine learning model.
Evaluate the model: Run the evaluation script to assess the model's performance.
Alternatively, you can explore and run the Jupyter Notebook provided:
Code: jupyter notebook fraud_detection.ipynb

#Model Training and Evaluation
The model is built using Scikit-learn and involves the following steps:

Data Preprocessing: Scaling features, handling imbalanced data.
Model Selection: Used Logistic Regression model
Training: The selected model is trained on the training dataset.
Evaluation: The model's performance is evaluated on the test dataset using metric accuracy.

#Results
The results of the model can be seen on stremlit app by running test.py file
