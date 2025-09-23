# Heart Failure Prediction Project
## Overview 📊
This project focuses on predicting heart failure outcomes using a dataset of clinical features. The dataset contains information about patients, including numerical and categorical features, and the target variable `DEATH_EVENT` (1: deceased, 0: alive). The project involves exploratory data analysis (EDA), data preprocessing, and training machine learning models to predict the target variable.

## Steps in the Project 🛠️

### 1. Exploratory Data Analysis (EDA) 🔍
- **Data Exploration**: Examined dataset structure, column descriptions, and missing values.
- **Outlier Detection**: Identified and removed outliers using the IQR method.
- **Univariate Analysis**: Visualized distributions of numerical and categorical features.
- **Bivariate Analysis**: Analyzed relationships between features and the target variable.
- **Correlation Analysis**: Generated a heatmap to identify correlations between numerical features.

### 2. Data Preprocessing 🧹
- Converted binary columns to categorical data types.
- Scaled numerical features using `StandardScaler` and `RobustScaler`.
- Encoded categorical features using `LabelEncoder` and `OneHotEncoder`.

### 3. Machine Learning Models 🤖
#### Non-Tree-Based Models
- **Logistic Regression**: Evaluated using stratified k-fold cross-validation.
- **Naive Bayes**: Applied Gaussian Naive Bayes for classification.
- **Support Vector Machines (SVM)**: Tested with linear, sigmoid, RBF, and polynomial kernels.
- **K-Nearest Neighbors (KNN)**: Used KNN with a specified number of neighbors.

#### Tree-Based Models 🌳
- **Decision Tree Classifier**: Trained using entropy as the splitting criterion.
- **Random Forest Classifier**: Evaluated feature importance and performance.
- **XGBoost Classifier**: Applied gradient boosting for classification.

### 4. Feature Selection ✂️
- Used Recursive Feature Elimination (RFE) with Logistic Regression to select the top 5 features.
- Evaluated model performance with selected features.

### 5. Model Evaluation 📈
- Compared model performance using metrics like accuracy, ROC-AUC, and classification reports.
- Addressed class imbalance and its impact on model performance.

### 6. Reporting 📝
- Generated a detailed report (`table_report.txt`) summarizing dataset characteristics, EDA findings, and model performance.
- Visualized key results and saved plots in the `plots` folder.

## Key Findings 🔑
- Class imbalance in the dataset affects model performance.
- Some features, such as `ejection_fraction` and `serum_creatinine`, are more predictive of heart failure outcomes.
- Tree-based models like Random Forest and XGBoost performed better than non-tree-based models.

## How to Run ▶️
1. Install required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `graphviz`.
2. Place the dataset (`heart_failure.csv`) in the working directory.
3. Run the Jupyter Notebook cells sequentially to reproduce the analysis and results.

## Folder Structure 📂
- `plots/`: Contains visualizations generated during the analysis.
- `table_report.txt`: Summary report of the dataset and findings.

## Future Work 🚀
- Address class imbalance using techniques like SMOTE or weighted loss functions.
- Experiment with hyperparameter tuning for better model performance.
- Explore deep learning models for improved predictions.

## Author ✍️
This project was developed as part of a heart failure prediction analysis.
