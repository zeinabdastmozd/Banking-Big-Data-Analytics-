# 🏦 Banking Big Data Analytics — Loan Default Prediction
![Project Overview](images/enhancing-loan-approvals-through-ai-powered-automation.webp)

### Financial Risk Analysis & Machine Learning

A machine learning and financial analytics project investigating **loan default prediction** using historical borrower and loan information.

The project explores how different machine learning approaches can be used to model loan-default risk, compare predictive performance, investigate feature relationships, and identify variables that contribute most strongly to model predictions.

---

## 📌 Project Overview

Loan default prediction is an important problem in financial risk management. Banks and financial institutions need reliable methods to assess the likelihood that a borrower may default on a loan.

This project applies several machine learning techniques to historical financial data to investigate:

- Which borrower and loan characteristics are associated with default risk
- How different preprocessing strategies affect model performance
- How regularisation can improve generalisation
- How ensemble methods perform compared with linear models
- Whether neural networks can capture more complex relationships
- Which features are most influential in predicting loan outcomes

The project combines **data preprocessing, exploratory analysis, correlation analysis, machine learning, model evaluation, and feature-importance analysis**.

---

# 🎯 Project Objective

The main objective of this project is:

> **To investigate and compare machine learning approaches for predicting loan default using historical financial data.**

The project evaluates several models and examines their ability to generalise to unseen data.

The overall workflow is:

```text
        Historical Loan Data
                 │
                 ▼
        Data Exploration
                 │
                 ▼
       Missing-Value Analysis
                 │
                 ▼
        Data Preprocessing
                 │
                 ▼
       Feature Engineering
                 │
                 ▼
       Correlation Analysis
                 │
                 ▼
       ┌─────────┼─────────┐
       ▼         ▼         ▼
    Linear     Ridge     Lasso
  Regression Regression Regression
       │         │         │
       └─────────┼─────────┘
                 │
                 ▼
          Random Forest
                 │
                 ▼
         Neural Network
          (MLPRegressor)
                 │
                 ▼
        Model Evaluation
                 │
                 ▼
       Feature Importance
                 │
                 ▼
       Loan Default Analysis
```
# 💳 Financial Risk Problem

The project focuses on the application of machine learning to **financial risk management**.

Predicting loan default can help financial institutions better understand borrower risk and support more informed lending decisions.

The analysis investigates relationships between loan outcomes and variables such as:

- Loan amount
- Annual income
- Employment length
- Home ownership
- Loan grade
- Credit-related variables
- Installment
- Total repayment
- Other borrower and loan characteristics

The report highlights relationships between variables including **ApplicantIncome and LoanAmount**, **Credit_History and Loan_Status**, and **LoanAmount and CoapplicantIncome**.

---

# 🔎 Exploratory & Correlation Analysis

Before applying machine learning models, the project investigates relationships between the available variables.

Correlation analysis is used to understand the strength and direction of relationships between variables.

The analysis identifies important relationships between:

- Applicant income
- Loan amount
- Credit history
- Co-applicant income
- Loan status
- Employment characteristics
- Loan grades

This provides an initial understanding of which variables may contribute to loan-default prediction.

---

# 🧹 Data Preprocessing

A significant part of the project investigates different strategies for handling missing data.

Several preprocessing approaches were evaluated.

### Method 1 — Statistical Imputation

For numerical variables, missing values were replaced using the **median**.

For `emp_length`, the **mode** was used.

The variables `id` and `member_id` were removed because they did not provide useful predictive information and contained missing values.

### Method 2 — Removing High-Missingness Variables

Variables with substantial missing values were removed where appropriate, including:

```text
id
member_id
mths_since_last_delinq
```
### Method 3 — Alternative Missing-Value Treatment

A further preprocessing strategy was investigated by replacing selected missing values with `0` and removing remaining incomplete observations.

Comparing these approaches allowed the project to investigate how different preprocessing choices affect model performance.

---

# 🤖 Machine Learning Models

Several machine learning approaches were implemented and compared.

## 1. Linear Regression

**Linear Regression** was initially used as a baseline model.

The model provides a simple and interpretable way of investigating relationships between financial variables and the target outcome.

However, the initial Linear Regression model showed substantial problems with generalisation, with a very large testing MSE compared with the training MSE.

This motivated the investigation of regularised and nonlinear models.

---

## 2. Ridge Regression

**Ridge Regression** was introduced to reduce overfitting by adding a regularisation penalty to the model.

The best parameter identified was:

```text
Alpha = 0.01
```
The third preprocessing method produced:
```text
Training MSE: 0.0654679717
Testing MSE:  0.0662019499
```
The relatively similar training and testing errors indicate substantially better generalisation than the initial Linear Regression approach.

---

## 3. Lasso Regression

**Lasso Regression** was also investigated.

Lasso applies regularisation using the absolute values of model coefficients and can perform implicit feature selection by reducing some coefficients to zero.

The best parameter was:
```text
Alpha = 0.01
```
Using the third preprocessing approach produced:
```text
Training MSE: 0.0676274084
Testing MSE:  0.0684082220
```
The results were close between training and testing, suggesting relatively good generalisation.

---

## 🌲 4. Random Forest

**Random Forest** model was implemented to investigate nonlinear relationships and feature interactions.

Random Forest combines multiple decision trees and can also provide feature-importance information.

The best configuration identified in the analysis used:
```text
n_estimators = 200
```
The reported performance was:
```text
Training MSE: 0.0027131258
Testing MSE:  0.8867148884
```
The analysis also investigated feature importance to understand which variables contributed most strongly to the model's predictions.

---

## 🧠 5. Neural Network

A Multi-Layer Perceptron (MLPRegressor) was investigated as a neural-network approach.

Neural networks can model nonlinear relationships and learn complex patterns through optimisation of model weights.

The reported results were:

### Method 2
```text
Training Accuracy: 0.9701976710
Testing Accuracy:  0.9642888604
```
### Method 3
```text
Training Accuracy: 0.9707016342
Testing Accuracy:  0.9627227970
```

The report identified the neural-network approach as producing the strongest predictive performance among the investigated approaches according to the reported accuracy results.

---

# 📊 Model Comparison

The project compares the investigated machine learning models using their reported evaluation metrics.

| Model | Evaluation Metric | Best Reported Result |
|---|---|---:|
| Linear Regression | MSE | High test error |
| Ridge Regression | MSE | **0.0662 test MSE** |
| Lasso Regression | MSE | **0.0684 test MSE** |
| Random Forest | MSE | **0.8867 test MSE** |
| MLP Neural Network | Accuracy | **96.43% test accuracy** |

The regularised linear models substantially improved the generalisation behaviour observed with the initial Linear Regression model.

The neural-network approach achieved approximately **96% test accuracy** in the reported experiments.

> **Note:** The project evaluates models using different metrics because both regression-based and classification-based approaches were investigated. For a future version, the loan-default problem could be formulated consistently as a binary classification task and evaluated using **Accuracy, Precision, Recall, F1-score, ROC-AUC, and a Confusion Matrix**.

---
# 🔬 Feature Importance & Financial Insights

The project also investigates which variables have stronger relationships with the target outcome.

The analysis identified several highly influential features, including:

- `grade_E`
- `grade_F`
- `grade_D`
- `grade_B`
- `grade_G`
- `application_type_Joint App`
- `home_ownership_RENT`
- `home_ownership_MORTGAGE`
- `grade_C`
- `emp_length_10+ years`

These features provide useful insights into the factors that may be associated with loan outcomes and borrower risk.

### 📉 Features with Weaker Relationships

The analysis also examined variables with weaker correlations, including:

- `tot_coll_amt`
- `loan_amnt`
- `annual_inc`
- `total_rev_hi_lim`
- `installment`
- `total_rec_prncp`
- `id`
- `member_id`

### ⚠️ Identifier Variables

The report highlights that identifier variables such as `id` and `member_id` should **not** be treated as meaningful predictive features.

These variables primarily identify individual records rather than representing meaningful financial or borrower characteristics. Therefore, they were removed during preprocessing to reduce the risk of introducing irrelevant information into the models.

# 📈 Key Findings

The analysis produced several important observations.

### 1. 🧹 Preprocessing Matters

Different approaches to handling missing data produced substantially different model behaviour and predictive performance.

This highlights the importance of carefully selecting an appropriate preprocessing strategy for financial datasets.

### 2. 📉 Regularisation Improves Generalisation

**Ridge** and **Lasso** regression produced much more stable test performance than the initial Linear Regression model.

Regularisation helped reduce overfitting and improved the models' ability to generalise to unseen data.

### 3. 🔎 Feature Relationships Are Important

Correlation analysis revealed meaningful relationships between financial variables and loan outcomes.

These relationships provided useful insights into which borrower and loan characteristics may contribute to financial risk.

### 4. 🌲 Random Forest Provides Feature Importance

The **Random Forest** model allowed the project to investigate the relative contribution of individual features.

This provided additional insight into which variables were most influential in the model's predictions.

### 5. 🧠 Neural Networks Captured Complex Patterns

The **MLP neural network** achieved approximately **96% test accuracy** in the reported experiments.

This suggests that the neural-network approach was able to capture complex patterns within the financial dataset.

---

# 🛠️ Technologies & Libraries

The project was developed using **Python** and a range of machine-learning and data-analysis tools.

### 🐍 Programming

- **Python**
- **Jupyter Notebook**

### 📊 Data Analysis

- **Pandas**
- **NumPy**

### 🤖 Machine Learning

- **Scikit-learn**
- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Random Forest**
- **MLPRegressor**

### 📈 Data Analysis & Evaluation

- **Correlation analysis**
- **Mean Squared Error (MSE)**
- **Accuracy**
- **Feature importance**
- **Model comparison**

  # 📁 Repository Structure

The project is organised into the following structure:

```text
Banking-Big-Data-Analytics/
│
├── README.md
│
├── notebooks/
│   └── banking_analysis.ipynb
│
├── code/
│   └── machine_learning.py
│
├── data/
│   └── loan_data.csv
│
├── report/
│   └── report.pdf
│
├── presentation/
│   └── presentation.pdf
│
└── images/
    ├── correlation_matrix.png
    ├── model_comparison.png
    └── feature_importance.png
```
# 🚀 How to Run

Follow the steps below to set up and run the project locally.

## 1. Clone the Repository

Clone the repository using Git:

```bash
git clone https://github.com/zeinabdastmozd/Banking-Big-Data-Analytics-.git
```
## 2. Navigate to the Project
```bash
cd Banking-Big-Data-Analytics-
```
## 3. Install Required Libraries
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```
## 4. Open Jupyter Notebook
```bash
jupyter notebook
```
Open the project's notebook and run the cells sequentially.

# 📊 Evaluation

The project evaluates the models using different metrics depending on the modelling approach.

### 📈 Regression Models

- **Mean Squared Error (MSE)**

### 🧠 Neural Network

- **Training Accuracy**
- **Testing Accuracy**

The analysis compares training and testing performance to investigate **generalisation, model stability, and potential overfitting**.

---

# ⚠️ Important Interpretation

This project is an **academic machine-learning study** and should not be used directly to make real-world lending decisions.

Loan-default prediction involves sensitive financial decisions, and a production system would require additional validation, fairness analysis, calibration, explainability, regulatory review, and appropriate risk-management procedures.

The results presented here should therefore be interpreted within the context of the dataset, preprocessing decisions, modelling choices, and experimental methodology.

---

# 🎓 Academic Context

**Module:** Big Data for Computational Finance

**Academic Year:** 2023/24

**Project:** Banking / Financial Risk Analytics

The project investigates the application of machine learning to financial risk management, with a particular focus on understanding and predicting loan-default behaviour.

The work combines:

**Financial Data + Data Preprocessing + Statistical Analysis + Machine Learning + Model Evaluation**

---

# 📚 References

The project report draws on research relating to:

- Machine learning for loan-default prediction
- Financial risk management
- Correlation analysis
- Regression techniques
- Predictive modelling

Key references used in the original work include research on loan-default prediction using machine learning, as well as academic resources discussing correlation and regression techniques.

---

# 👩‍💻 Author

## Zeinab Dast Mozd

**MSc Artificial Intelligence**

### Areas of Interest

- 🤖 Artificial Intelligence
- 🧠 Machine Learning
- 📊 Data Science
- 💳 Financial Analytics
- 🏦 AI for Finance
- 📈 Predictive Modelling
- 🔬 AI Research

---

# ⭐ Project Status

**Completed Academic Project**

This repository contains the code, analysis, datasets/results, and report associated with the banking and financial machine-learning project.

---

# ⭐ Thank You

Thank you for exploring this project.

If you are interested in **machine learning, financial analytics, AI for finance, or predictive modelling**, feel free to explore the repository and connect with me.


