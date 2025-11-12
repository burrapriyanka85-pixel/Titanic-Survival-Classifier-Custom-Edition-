# Titanic Survival Classifier (Custom Edition)

A machine learning project that predicts whether a passenger survived the Titanic disaster.  
This version implements a modern, modular ML pipeline with automated preprocessing, feature engineering, 
cross-validation, and model interpretation.

---

## 🚀 Features
- KNN-based imputation for handling missing data
- Automated preprocessing using `ColumnTransformer`
- Feature engineering (`Title`, `Deck`, `Family_Size`, `Fare_per_person`, `Name_length`, etc.)
- Model selection with `RandomizedSearchCV`
- Feature importance analysis using permutation importance
- Probability-based predictions for interpretability
- Achieved test accuracy: **≈ 82.68%**

---

## 🧠 Tech Stack
- **Python**  
- **pandas, numpy, seaborn, matplotlib**  
- **scikit-learn**  
- **joblib**

---

## 📊 Outputs
- `titanic_pipeline_joblib_v1.pkl` – Trained model pipeline  
- `titanic_predictions_with_probs.csv` – Predictions with survival probabilities  
- `Titanic Survival Classifier (Custom Edition).ipynb` – Full notebook with code and visualizations

---

## ⚙️ How to Run
```bash
git clone https://github.com/yourusername/Titanic-Survival-Classifier-Custom-Edition.git
cd Titanic-Survival-Classifier-Custom-Edition

📜 License
This project is released under the MIT License.
Dataset © Kaggle – Titanic: Machine Learning from Disaster.
pip install -r requirements.txt
jupyter notebook
