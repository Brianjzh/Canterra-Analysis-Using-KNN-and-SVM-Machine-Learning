# Employee Attrition Prediction Using KNN and SVM Models

This project focuses on predicting employee attrition using two machine learning models: **K-Nearest Neighbors (KNN)** and **Support Vector Machines (SVM)**. The goal is to identify employees at risk of leaving the company, enabling proactive decision-making and targeted retention strategies.

---

## **Project Overview**

Employee attrition is a significant challenge for organizations, impacting talent retention and overall productivity. This project leverages machine learning models to predict employee attrition, providing actionable insights for HR and management to reduce turnover and improve employee satisfaction.

### **Key Features**
- **Data Preprocessing**: Handling missing values, removing non-informative variables, and balancing the dataset.
- **Model Training**: Implementation of KNN and SVM (with linear, radial, and polynomial kernels) to predict employee attrition.
- **Model Evaluation**: Comparison of models using ROC-AUC, precision, and recall metrics.
- **Recommendations**: Actionable insights for HR to design targeted retention programs.

---

## **Key Findings**

### **KNN Model**
- **AUC Score**: 0.9429
- **Precision**: 62.86%
- **Recall**: 66.33%
- The KNN model is highly effective in identifying employees at risk of leaving, providing a reliable tool for retention efforts.

### **SVM Model**
- **Radial Kernel**:
  - **AUC Score**: 0.9978
  - **Precision**: 99.18%
  - **Recall**: 99.09%
- The SVM Radial Kernel outperformed other models, achieving near-perfect accuracy in predicting employee attrition.

---


## Some Key Visualizations
### Marginal Effects wtih Confidence Intervals
![Attrition by Income](images/MarginalEffect.png)

### ROC Curve for Decision Tree
![Attrition by Business Travel](images/ROCCurveDecisionTree.png)

### Heatmap of Correlation
![ROC Curve](images/Heatmap.png)

---

## **Code Snippet: SVM Model Training**

Below is a snippet of the code used to train the SVM models with linear, radial, and polynomial kernels:

```R
### Feature & Model Selection using SVM method ---

# Define train control for cross-validation
tr_control = trainControl(
  method = "cv", number = 10, # 10-fold cross validation
  classProbs = TRUE,  
  summaryFunction = twoClassSummary  # Use twoClassSummary to compute AUC
)

# Train SVM models with linear, radial, and polynomial kernels

# Linear Kernel
svm_linear = train(
  Attrition ~ .,
  data = training_set[keep_index, ],
  method = "svmLinear",
  tuneGrid = expand.grid(C = c(0.01, 0.1, 1, 5, 10)),
  trControl = tr_control,
  metric = "ROC" # "ROC" gives us AUC & silences warning about Accuracy
)

# Plot results for linear kernel
plot(svm_linear)
svm_linear$bestTune

# Radial Kernel
svm_radial = train(
  Attrition ~ .,
  data = training_set[keep_index, ],
  method = "svmRadial",
  tuneGrid = expand.grid(C = c(0.01, 0.1, 1, 5, 10), sigma = c(0.5, 1, 2, 3)),
  trControl = tr_control,
  metric = "ROC"
)

# Plot results for radial kernel
plot(svm_radial)
svm_radial$bestTune

# Polynomial Kernel
svm_poly = train(
  Attrition ~ .,
  data = training_set[keep_index, ],
  method = "svmPoly",
  tuneLength = 4, # Automatically try different parameters
  trControl = tr_control,
  metric = "ROC"
)

# Plot results for polynomial kernel
plot(svm_poly)
svm_poly$bestTune
