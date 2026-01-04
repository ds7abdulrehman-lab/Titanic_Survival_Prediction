# Titanic Machine Learning Project

This repository contains my explorations of machine learning models on the **Titanic dataset**. The goal is to predict passenger survival using structured feature engineering, pipelines, and model evaluation. The project is ongoing — I plan to add more models, hyperparameter tuning, and performance analysis over time.  

---

## Current Implementations

### 1. Logistic Regression Pipeline
- **Preprocessing:**  
  - Impute missing numerical values with median  
  - Standardize numerical features (`Age`, `Fare`, `SibSp`, `Parch`, `FamilySize`, `IsAlone`)  
  - Impute missing categorical values with most frequent  
  - One-Hot Encode categorical features (`Sex`, `Embarked`, `Pclass`)  

- **Feature Engineering:**  
  - `FamilySize` = `SibSp` + `Parch` + 1  
  - `IsAlone` = 1 if passenger is alone, else 0  

- **Model:** Logistic Regression (`max_iter=1000`)  

- **Results (Current Run):**  
  - Train Accuracy: ~[Insert value]  
  - Test Accuracy: ~[Insert value]  

---

### 2. XGBoost Pipeline with Grid Search and Feature Importance
- **Upgraded from logistic regression:** Using **XGBoost** for better predictive power  
- **Hyperparameter Tuning:** GridSearchCV within a pipeline for `n_estimators`, `max_depth`, `learning_rate`, and `subsample`  
- **Feature Importance Analysis:**  
  - Extracted post-preprocessing importances  
  - Top contributors: `Sex`, `Title`, `FarePerPerson`  
  - Visualization included in the repository  

- **Notes:**  
  - Ensures **no data leakage**  
  - Correctly handles categorical features and pipeline transformations  
---

## Recent Progress: Model Evaluation & Explainability (Ongoing)

In the current stage of the project, the focus has shifted from model construction to **understanding model behavior, evaluation, and interpretability**.

### Cross-Validation (CV)
- Used cross-validation to obtain a more robust estimate of model performance.
- Learned how CV reduces dependency on a single data split and helps identify overfitting.
- Evaluation primarily guided by ROC-AUC to balance class performance.

### Confusion Matrix Analysis
- Analyzed the confusion matrix to understand **where the model is making incorrect predictions**.
- Identified false positives vs. false negatives to study trade-offs between predicting survival and non-survival.
- Helped contextualize precision and recall beyond aggregate accuracy.

### Classification Metrics
- Interpreted precision, recall, and F1-score for each class individually.
- Observed differences in model behavior when predicting survivors versus non-survivors.
- Reinforced the importance of class-wise evaluation in imbalanced datasets.

### Model Explainability with SHAP
Implemented SHAP to explain both **global model behavior** and **individual predictions**.

**SHAP Summary Plot (Global Explanation):**
- Visualized feature impact across the entire dataset.
- Identified key drivers such as Age, FarePerPerson, FamilySize, and passenger class–related features.
- Understood how feature values influence prediction direction and magnitude.

**SHAP Waterfall Plot (Individual Explanation):**
- Analyzed a single passenger’s prediction in detail.
- Tracked how each feature contributed to pushing the model output toward survival or non-survival.
- Improved interpretability and trust in model decisions.

### Notes
- This project remains **iterative and exploratory**.
- Future steps include further feature refinement, validation on unseen data, and comparison with additional models.


