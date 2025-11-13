Titanic Survival Classifier (Custom Edition)
🧭 Overview

A machine learning project that predicts whether a passenger survived the Titanic disaster.
This custom edition implements a modern, modular ML pipeline, featuring:

Automated preprocessing

Advanced feature engineering

Cross-validation

Hyperparameter tuning

Model explainability using Permutation Feature Importance

Built with Python, scikit-learn, pandas, and matplotlib.

🚀 Features

🧩 KNN-based imputation for missing data

⚙️ Automated preprocessing using ColumnTransformer + OneHotEncoder

🧠 Feature engineering:

Title extraction

Deck extraction

Family Size

Fare per Person

Name Length

🔍 Model tuning with RandomizedSearchCV

📊 Permutation Feature Importance for interpretability

🎯 Probability-based predictions

✅ Test Accuracy: ≈ 82.68%

🧾 Model Summary

Final Model Used: Logistic Regression (Random Forest also tested)

Accuracy: ~0.82 on test data

Top Features:

Sex

Title

Pclass

Age

Explainability: Visualized using Permutation Feature Importance

📊 Key Results
🔍 Feature Importance Visualization

(Ensure feature_importance_plot.png is uploaded in the repository.)

🧠 How to Run
1️⃣ Clone the Repository
git clone https://github.com/burrapriyanka85-pixel/Titanic-Survival-Classifier-Custom-Edition.git
cd Titanic-Survival-Classifier-Custom-Edition

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Notebook
jupyter notebook
Open:
Titanic Survival Classifier (Custom Edition).ipynb

Tech Stack

Python 3.10+

pandas

NumPy

matplotlib

seaborn

scikit-learn

joblib

📦 Outputs

| File                                                 | Description                        |
| ---------------------------------------------------- | ---------------------------------- |
| `Titanic Survival Classifier (Custom Edition).ipynb` | Full notebook with code & analysis |
| `Titanic-Dataset.csv`                                | Cleaned dataset used for training  |
| `titanic_pipeline_joblib_v1.pkl`                     | Trained ML pipeline                |
| `titanic_predictions_with_probs.csv`                 | Predictions with probabilities     |
| `titanic_predictions_custom.csv`                     | Custom prediction outputs          |
| `requirements.txt`                                   | Dependencies list                  |

📂 Project Structure

├── Titanic Survival Classifier (Custom Edition).ipynb
├── Titanic-Dataset.csv
├── titanic_pipeline_joblib_v1.pkl
├── titanic_predictions_with_probs.csv
├── titanic_predictions_custom.csv
├── feature_importance_plot.png
├── requirements.txt
└── README.md

🚧 Future Enhancements

Add Streamlit or Flask deployment

Integrate SHAP explainability

Convert the model pipeline into a Python package

Add automated tests & CI/CD

Experiment with XGBoost / LightGBM

📜 License

This project is released under the MIT License.

Dataset Source:
🔗 https://www.kaggle.com/c/titanic
