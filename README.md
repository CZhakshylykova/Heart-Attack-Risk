
# Heart Failure Prediction 
This project leverages machine learning to predict heart failure outcomes and provides model explainability using SHAP (SHapley Additive exPlanations). The workflow includes exploratory data analysis (EDA), data preprocessing, model training, and interpretable machine learning insights.

---

## Project Structure

- **`project_EDA.ipynb`** – Exploratory Data Analysis & visualization.
- **`project_model.ipynb`** – Model training, evaluation, and SHAP explainability.
- **`README.md`** – Project overview and instructions.

---

## Dataset

- **Source**: [Heart Failure Clinical Records Dataset (Kaggle)](https://www.kaggle.com/datasets/rithikkotha/heart-failure-clinical-records-dataset)
- The dataset contains medical records of patients with heart failure, including features such as age, anaemia, ejection fraction, serum creatinine, and more.

---

## Main Features

- Comprehensive data exploration and cleaning.
- Handling class imbalance with SMOTE.
- Predictive modeling using Random Forest and Logistic Regression.
- Model evaluation with ROC-AUC and other metrics.


---

## How to Run

1. Clone or download this repository.
2. Download the dataset from Kaggle and place `heart_failure_clinical_records_dataset.csv` in your working directory.
3. Open `project_EDA.ipynb` and follow the cells for data analysis.
4. Run `project_model.ipynb` for model building 

---

## Requirements

- Python 3.x
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`
- `imbalanced-learn`
- `shap`

### Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn shap
```

---

## Acknowledgements

- Dataset: [Heart Failure Clinical Records Dataset (Kaggle)](https://www.kaggle.com/datasets/rithikkotha/heart-failure-clinical-records-dataset)

---

## License

This project is for educational and research purposes.

---

## Contributing

Contributions are welcome! If you'd like to improve this project, feel free to fork the repository, make your changes, and submit a pull request. Please ensure your code follows the project's style guidelines and includes appropriate documentation.

---

## Contact

For any questions or feedback, please reach out via zhakshylykova@gmail.com

---

Thank you for exploring this project! If you find it helpful, consider giving it a star on GitHub. 😊
