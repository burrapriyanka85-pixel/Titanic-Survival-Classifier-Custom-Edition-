# 🚢 Titanic Survival Classifier (Custom Edition)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 🧭 Overview
A **machine learning project** that predicts whether a passenger survived the Titanic disaster.  
This custom edition implements a **modern, modular ML pipeline** with:
- Automated preprocessing  
- Feature engineering  
- Cross-validation  
- Model explainability (Permutation Feature Importance)  
Built with **Python**, **scikit-learn**, **pandas**, and **matplotlib**.

---

## 🚀 Features
- 🧩 **KNN-based imputation** for handling missing data  
- ⚙️ **Automated preprocessing** using `ColumnTransformer` and `OneHotEncoder`  
- 🧠 **Feature engineering** (Title, Deck, Family Size, Fare per Person, Name Length, etc.)  
- 🔍 **Model selection** with `RandomizedSearchCV`  
- 📊 **Permutation Feature Importance** for interpretability  
- 🎯 **Probability-based predictions** for better decision support  
- ✅ **Achieved test accuracy:** ≈ **82.68%**

---

## 🧾 Model Summary
- **Model Used:** Logistic Regression / Random Forest (depending on experiment)  
- **Accuracy:** ~0.82 on test data  
- **Top Features:** `Sex`, `Title`, `Pclass`, `Age`  
- **Explainability:** Used *Permutation Feature Importance* to visualize model insights  

---

## 📊 Key Results
### Feature Importance Visualization
![Feature Importance](feature_importance_plot.png)

*(Upload your permutation importance plot as `feature_importance_plot.png` in this repo.)*

---

## 🧠 How to Run
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/burrapriyanka85-pixel/Titanic-Survival-Classifier-Custom-Edition.git
cd Titanic-Survival-Classifier-Custom-Edition
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Notebook
jupyter notebook
Then open:
java
Copy code
Titanic Survival Classifier (Custom Edition).ipynb
and run all cells to reproduce the results.

🧰 Tech Stack
Python 3.10+
pandas, numpy, seaborn, matplotlib
scikit-learn
joblib

📦 Outputs
File	Description
Titanic Survival Classifier (Custom Edition).ipynb	Full notebook with code & visualizations
Titanic-Dataset.csv	Cleaned dataset used for model training
titanic_pipeline_joblib_v1.pkl	Trained ML pipeline
titanic_predictions_with_probs.csv	Predictions with survival probabilities
titanic_predictions_custom.csv	Custom prediction outputs
requirements.txt	Python dependencies list

📜 License
This project is released under the MIT License.
Dataset © Kaggle – Titanic: Machine Learning from Disaster.

