# ML Assignment 2 – Classification Models

## Problem Statement
The objective of this assignment is to implement and evaluate multiple machine learning
classification models on a real-world dataset. The models are compared using standard
evaluation metrics and deployed as an interactive Streamlit web application.

---

## Dataset Description
The dataset used for this assignment is the **Credit Card Default Dataset**, which contains
information about credit card clients and whether they defaulted on their payment.

- **Type of problem**: Binary classification  
- **Target variable**: Default payment (Yes / No)  
- **Number of instances**: ~30,000  
- **Number of features**: 23 (including demographic and financial attributes)  
- **Source**: Public dataset from UCI Machine Learning Repository / Kaggle  

This dataset was chosen because it satisfies the minimum feature and instance requirements
and is suitable for evaluating both linear and ensemble classification models.

---

## Models Used
The following machine learning models are implemented and evaluated:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes Classifier  
5. Random Forest Classifier (Ensemble)  
6. XGBoost Classifier (Ensemble)

Evaluation metrics include Accuracy, AUC, Precision, Recall, F1 Score, and Matthews
Correlation Coefficient (MCC).

## 📊 Evaluation Metrics Comparison

The following table compares the performance of all implemented classification models
using standard evaluation metrics.

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8077 | 0.7076 | 0.6868 | 0.2396 | 0.3553 | 0.3244 |
| Decision Tree | 0.8107 | 0.7223 | 0.6265 | 0.3564 | 0.4544 | 0.3705 |
| K-Nearest Neighbors | 0.7897 | 0.6999 | 0.5364 | 0.3610 | 0.4315 | 0.3176 |
| Naive Bayes | 0.4160 | 0.6516 | 0.2496 | 0.8176 | 0.3824 | 0.1111 |
| Random Forest (Ensemble) | 0.8162 | 0.7736 | 0.6618 | 0.3451 | 0.4537 | 0.3834 |
| XGBoost (Ensemble) | 0.8180 | 0.7748 | 0.6607 | 0.3640 | 0.4694 | 0.3945 |


## Model-wise Observations

| ML Model | Observation |
|---------|-------------|
| Logistic Regression | The model achieves good accuracy but low recall, indicating that it is conservative in predicting defaults. It performs well as a baseline linear classifier but struggles to capture complex non-linear patterns in the data. |
| Decision Tree | The Decision Tree improves recall and F1 score compared to Logistic Regression, showing its ability to model non-linear relationships. However, it may overfit and lacks the robustness of ensemble methods. |
| K-Nearest Neighbors | KNN shows moderate performance with reasonable recall but slightly lower accuracy. Its performance is affected by high dimensionality and sensitivity to feature scaling. |
| Naive Bayes | Naive Bayes achieves very high recall but extremely low accuracy and precision, indicating many false positives. This behavior is expected due to the strong independence assumptions made by the model. |
| Random Forest (Ensemble) | Random Forest provides a strong balance between precision and recall with improved AUC and MCC. The ensemble approach reduces overfitting and improves overall generalization. |
| XGBoost (Ensemble) | XGBoost delivers the best overall performance with the highest AUC, F1 score, and MCC. Gradient boosting effectively captures complex patterns and handles class imbalance better than other models. |




