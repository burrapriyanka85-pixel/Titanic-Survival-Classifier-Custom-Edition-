🚢 Titanic Survival Classifier (Custom Edition)

A machine learning project that predicts whether a passenger survived the Titanic disaster using a modern and modular ML pipeline.

This custom edition includes automated preprocessing, advanced feature engineering, model tuning, and explainability.

📌 Overview

This project implements:

Automated preprocessing

Feature engineering

Cross-validation

Hyperparameter tuning

Model selection

Permutation Feature Importance

Probability-based predictions

Built using Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib.

🚀 Features

KNN imputation for missing values

Automated preprocessing with ColumnTransformer

One-hot encoding + scaling

Title extraction

Deck extraction

Family size

Fare per person

Name length

RandomizedSearchCV for tuning

Permutation Feature Importance

Final model accuracy: ~82.68%

📊 Model Summary

Final Model: Logistic Regression
Accuracy: ~0.82 on test data

Most important features:

Sex

Title

Pclass

Age

Explainability is provided using Permutation Feature Importance.

📈 Key Results

Make sure you upload:

feature_importance_plot.png

🧠 How to Run the Project
1. Clone the repository
git clone https://github.com/burrapriyanka85-pixel/Titanic-Survival-Classifier-Custom-Edition.git
cd Titanic-Survival-Classifier-Custom-Edition

2. Install dependencies
pip install -r requirements.txt

3. Run the notebook
jupyter notebook


Open:

Titanic Survival Classifier (Custom Edition).ipynb

🛠 Tech Stack

Python 3.10+

pandas

NumPy

matplotlib

seaborn

scikit-learn

joblib

📦 Project Outputs

| File                                               | Description                 |
| -------------------------------------------------- | --------------------------- |
| Titanic Survival Classifier (Custom Edition).ipynb | Notebook with full pipeline |
| Titanic-Dataset.csv                                | Cleaned dataset             |
| titanic_pipeline_joblib_v1.pkl                     | Saved ML pipeline           |
| titanic_predictions_with_probs.csv                 | Predictions + probabilities |
| titanic_predictions_custom.csv                     | Custom prediction output    |
| feature_importance_plot.png                        | Feature importance chart    |
| requirements.txt                                   | Dependencies                |


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

Add Streamlit / Flask deployment

Add SHAP explainability

Convert pipeline into a Python package

Add automated tests and CI/CD

Experiment with XGBoost / LightGBM

📜 License

Released under the MIT License.

Dataset Source:
https://www.kaggle.com/c/titanic
