🚢 Titanic Survival Classifier (Custom Edition)

A production-style machine learning project that predicts whether a passenger survived the Titanic disaster using a modern, modular, and explainable ML pipeline.

This project focuses on robust preprocessing, domain-driven feature engineering, systematic model evaluation, and interpretability, following industry best practices.

📌 Project Overview

This project implements an end-to-end supervised machine learning workflow, covering:

Exploratory Data Analysis (EDA)

Feature engineering

Automated preprocessing pipelines

Model training, tuning, and comparison

Performance evaluation using multiple metrics

Model explainability

Reproducible prediction pipelines

The goal is to demonstrate real-world ML engineering skills, not just algorithm usage.

🧠 ML Pipeline Architecture
Raw Data (CSV)
     │
     ▼
Exploratory Data Analysis (EDA)
     │
     ▼
Feature Engineering
 ├─ Title extraction
 ├─ Deck extraction
 ├─ Family size
 ├─ Fare per person
 ├─ Name length
     │
     ▼
Preprocessing Pipeline
 ├─ KNN Imputation (numeric features)
 ├─ One-Hot Encoding (categorical features)
 ├─ Feature Scaling
     │
     ▼
Model Training
 ├─ Logistic Regression
 ├─ Random Forest
 ├─ Gradient Boosting
 ├─ Support Vector Machine (RBF)
     │
     ▼
Model Selection (Stratified Cross-Validation)
     │
     ▼
Explainability
 ├─ Permutation Feature Importance
 ├─ SHAP Analysis
     │
     ▼
Saved Pipeline + Predictions

🚀 Key Features

Automated preprocessing using Pipeline and ColumnTransformer

KNN imputation for missing numerical values

One-hot encoding and feature scaling

Advanced feature engineering:

Title extraction from passenger names

Deck extraction from cabin information

Family size computation

Fare per person normalization

Name length as a text-complexity feature

Hyperparameter tuning using RandomizedSearchCV

Probability-based survival predictions

Model explainability using permutation importance and SHAP

📊 Evaluation Metrics

The final selected model was evaluated using multiple performance metrics to ensure robustness and reliability:

Accuracy

Precision

Recall

F1-score

Stratified Cross-Validation (mean and standard deviation)

Cross-validation results demonstrate stable performance with low variance, indicating good generalization to unseen data.

🔍 Model Comparison

Multiple classification models were trained and evaluated using stratified cross-validation:

Logistic Regression

Random Forest

Gradient Boosting

Support Vector Machine (RBF)

Models were compared using mean cross-validation accuracy and standard deviation.

Logistic Regression was selected as the final model due to its:

Strong and stable performance

Lower variance

High interpretability

Suitability for explainability analysis

⚙️ Model Optimization

A baseline model was first trained to establish reference performance.

Hyperparameter tuning was then performed using RandomizedSearchCV with stratified cross-validation to:

Improve predictive performance

Reduce overfitting

Identify optimal hyperparameter configurations

🔎 Model Explainability

To ensure transparency and trust in predictions, multiple explainability techniques were applied:

🔹 Permutation Feature Importance

Measures performance degradation when feature values are randomly shuffled

Identifies globally important features influencing model predictions

🔹 SHAP (SHapley Additive Explanations)

Global feature importance using SHAP summary plots

Feature contribution analysis using SHAP bar plots

These methods confirm that the model learns domain-relevant and interpretable patterns, rather than spurious correlations.

📈 Prediction Probability Analysis

The distribution of predicted survival probabilities was analyzed to understand model confidence.

This analysis helps distinguish between:

High-certainty predictions

Borderline or uncertain cases

♻️ Reproducibility & Pipeline Design

A single scikit-learn Pipeline handles preprocessing and modeling

Identical transformations are applied during training and inference

The trained pipeline is saved using joblib

Predictions are generated using the saved pipeline without retraining

This design ensures consistent, reproducible, and reliable predictions.

🧠 How to Run the Project
1️⃣ Clone the repository
git clone https://github.com/burrapriyanka85-pixel/Titanic-Survival-Classifier-Custom-Edition.git
cd Titanic-Survival-Classifier-Custom-Edition

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the notebook
jupyter notebook


Open:

Titanic Survival Classifier (Custom Edition).ipynb

🛠 Tech Stack

Python 3.10+

pandas

NumPy

scikit-learn

matplotlib

seaborn

joblib

SHAP

📦 Project Outputs
File	Description
Titanic Survival Classifier (Custom Edition).ipynb	Complete ML pipeline notebook
Titanic-Dataset.csv	Dataset
titanic_pipeline_joblib_v1.pkl	Saved trained pipeline
titanic_predictions_with_probs.csv	Predictions with probabilities
titanic_predictions_custom.csv	Custom prediction output
feature_importance_plot.png	Feature importance visualization
requirements.txt	Project dependencies
📁 Project Structure
├── Titanic Survival Classifier (Custom Edition).ipynb
├── Titanic-Dataset.csv
├── titanic_pipeline_joblib_v1.pkl
├── titanic_predictions_with_probs.csv
├── titanic_predictions_custom.csv
├── feature_importance_plot.png
├── requirements.txt
└── README.md

🚧 Future Enhancements

Streamlit / Flask deployment

REST API using FastAPI

Modular Python package (src/ structure)

Automated testing and CI/CD

Model monitoring

Experimentation with XGBoost / LightGBM

📌 Disclaimer

This project is intended for educational, academic, and portfolio purposes.
It demonstrates machine learning concepts and should not be used for real-world decision-making.

📜 License

Released under the MIT License.

📚 Dataset Source

Kaggle Titanic Competition
https://www.kaggle.com/c/titanic
