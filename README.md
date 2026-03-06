Credit Card Fraud Detection System

Project Overview:
This project builds a machine learning system to detect fraudulent credit card transactions. Fraud detection is a critical problem in the financial industry because fraudulent transactions cause significant financial losses for banks and customers.

The objective of this project is to develop a predictive model that can identify suspicious transactions based on historical transaction data. The system uses machine learning techniques along with methods to handle imbalanced datasets, which are common in fraud detection problems.

Dataset:
The dataset used for this project is the Credit Card Fraud Detection dataset available on Kaggle.
It contains transactions made by European cardholders over a two-day period.

Key characteristics of the dataset:
Total transactions: 284,807
Fraudulent transactions: 492
Fraud rate: ~0.17%
Highly imbalanced dataset

The dataset contains:
V1 – V28: anonymized features generated using PCA
Time: seconds elapsed between transactions
Amount: transaction amount
Class: target variable
0 → Normal transaction
1 → Fraudulent transaction

Technologies Used:
Python
Pandas
NumPy
Scikit-learn
XGBoost
Imbalanced-learn (SMOTE)
Matplotlib
Seaborn
Joblib
Git & GitHub
'''
Project Structure:
Fraud-Detection-Project
│
├── data/
│   └── creditcard.csv
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│
├── models/
│   └── fraud_model.pkl
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
'''
Description of Files:
'''
1. preprocessing.py
Loads the dataset
Splits data into training and testing sets
Scales the transaction amount
Applies SMOTE to handle class imbalance

2. train.py
Contains functions for training machine learning models

3. evaluate.py
Evaluates models using metrics such as: 
Confusion Matrix
Precision
Recall
F1 Score
ROC-AUC Score

4. main.py
Runs the complete pipeline:
Data loading
Preprocessing
Model training
Model evaluation
Model saving
Machine Learning Approach
'''
The project follows a structured machine learning pipeline:

1. Load and explore transaction data

2. Split dataset into training and testing sets

3. Scale numerical features

4. Handle class imbalance using SMOTE

5. Train machine learning models

6. Evaluate models using classification metrics

7. Save the trained model for future use

Model Evaluation Metrics:
Since fraud detection datasets are highly imbalanced, accuracy alone is not a reliable metric. Instead, the following metrics are used:

1. Precision – how many predicted fraud cases were actually fraud
2. Recall – how many fraud cases were correctly detected
3. F1 Score – balance between precision and recall
4. ROC-AUC Score – overall model performance
5. These metrics help evaluate how effectively the model detects fraudulent transactions.

Running the Project
1. Clone the repository
git clone https://github.com/rajp13037-lang/Fraud-Detection-Project.git
cd Fraud-Detection-Project
2. Install dependencies
pip install -r requirements.txt
3. Add dataset
Download the dataset and place it inside the data/ folder:
data/creditcard.csv
4. Run the pipeline
python main.py

The trained model will be saved in:
models/fraud_model.pkl

Business Impact:
Fraud detection systems are widely used by banks and financial institutions to prevent financial losses and protect customers. A reliable fraud detection model helps identify suspicious transactions in real time, allowing institutions to take preventive actions.

Future Improvements:
1. Train and compare multiple models such as Random Forest and XGBoost
2. Add visualization dashboards for fraud detection analysis
3. Deploy the model using a web application
4. Implement real-time fraud detection pipelines

Author
Raj Parmar

