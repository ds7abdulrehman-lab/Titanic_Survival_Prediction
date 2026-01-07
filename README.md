# Titanic Machine Learning Project

This repository contains a complete, end-to-end machine learning solution for predicting passenger survival on the Titanic dataset.  
The project emphasizes **clean pipelines, feature engineering, robust evaluation, and model interpretability**, following industry-aligned best practices.

The project is now **complete**, covering the full lifecycle from baseline modeling to advanced explainability.

---

## Project Objectives
- Build reproducible and leakage-free machine learning pipelines
- Apply structured feature engineering to improve predictive performance
- Compare baseline and advanced models
- Perform thorough model evaluation beyond accuracy
- Interpret model behavior using modern explainability techniques

---

## Dataset
- Source: Kaggle Titanic Dataset
- Target Variable: `Survived` (0 = No, 1 = Yes)
- Features include passenger demographics, ticket class, fare, and family information

---

## Feature Engineering
Custom features were created to improve signal quality:
- **Title** extracted from passenger names (with rare-title grouping)
- **FamilySize** = SibSp + Parch + 1
- **IsAlone** indicator
- **FarePerPerson** = Fare / FamilySize

All feature engineering is implemented inside a Scikit-learn pipeline to prevent data leakage.

---

## Preprocessing
Implemented using `ColumnTransformer`:
- Numerical features:
  - Median imputation
  - Standard scaling
- Categorical features:
  - Most-frequent imputation
  - One-hot encoding
- Fully integrated into pipelines for reproducibility

---

## Models Implemented

### 1. Logistic Regression (Baseline)
- Used as a benchmark model
- Max iterations set to 1000
- Helps establish baseline performance and interpretability

### 2. XGBoost Classifier (Final Model)
- Implemented within a pipeline
- Hyperparameter tuning using **GridSearchCV**
- Optimized using **5-fold cross-validation**
- Primary metric: **ROC-AUC**

Tuned hyperparameters include:
- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`

---

## Model Evaluation
Evaluation goes beyond accuracy to fully understand model behavior:

- **Cross-Validation ROC-AUC** for robust performance estimation
- **Confusion Matrix** to analyze false positives and false negatives
- **Classification Report** (precision, recall, F1-score per class)
- **Precision–Recall Curve** for class-level trade-off analysis
- **Learning Curves** to diagnose bias vs. variance and data sufficiency

---

## Model Explainability (SHAP)
SHAP was used to interpret the final XGBoost model:

### Global Explainability
- SHAP summary plots show overall feature importance
- Key drivers include:
  - Age
  - FarePerPerson
  - FamilySize
  - Passenger class–related features

### Local Explainability
- SHAP waterfall plots explain individual passenger predictions
- Demonstrates how each feature pushes predictions toward survival or non-survival

These analyses improve trust and transparency in model decisions.

---

## Key Learnings
- Feature engineering can significantly improve model performance
- Pipelines are essential for preventing data leakage
- Evaluation metrics must align with the problem, not just accuracy
- Model interpretability is critical for real-world ML applications

---

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- SHAP

---

## Project Status
**Completed**

Future improvements may include:
- Testing additional ensemble models
- Feature refinement
- Validation on external or unseen datasets

---

## Author
Aspiring Data Scientist actively seeking **Data Science / Machine Learning internship opportunities**.  
Feedback, suggestions, and collaboration are welcome.
