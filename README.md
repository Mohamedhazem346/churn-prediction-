# 📉 Customer Churn Prediction

A machine learning project that predicts whether a customer will churn using a **Gaussian Naive Bayes** classifier. Built with Python and scikit-learn.

---

## 📌 Overview

Customer churn refers to when a customer stops doing business with a company. This project builds a binary classification model to predict churn based on customer demographics and account data, helping businesses take proactive retention actions. 

---

## 📂 Project Structure

```
churn-prediction/
│
├── churn_prediction.ipynb    # Main Jupyter Notebook
├── churn_dataset.xlsx        # Dataset (not included — see Dataset section)
└── README.md
```
---

## 🧰 Technologies Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `scikit-learn` | Model training, splitting, and evaluation |
| `seaborn` | Data visualization |
| `matplotlib` | Plotting | 

---

## 📊 Dataset

The dataset is an Excel file (`churn_dataset.xlsx`) containing the following key columns:

| Column | Type | Description |
|---|---|---|
| `Age` | Numerical | Customer age |
| `Tenure` | Numerical | Number of months as a customer |
| `Sex` | Categorical | Male / Female |
| `Churn` | Categorical (Target) | Yes / No |

> **  The dataset is included in this repository.
>---

## 🔄 Workflow

1. **Import Libraries** — Load all required Python packages
2. **Load Dataset** — Read the Excel file using pandas
3. **Exploratory Data Analysis (EDA)** — Inspect shape, columns, data types, missing values, and summary statistics
4. **Data Preprocessing** — Encode categorical variables (`Sex`, `Churn`) to numeric values
5. **Data Visualization** — Plot distributions of Age and Sex using seaborn
6. **Feature Engineering** — Select features `Age`, `Tenure`, `Sex` and target `Churn`
7. **Model Training** — Train a Gaussian Naive Bayes classifier with an 80/20 train-test split
8. **Evaluation** — Report accuracy score and visualize the confusion matrix
9. **Prediction** — Predict churn for a new customer given their profile .


> ---

## 📈 Model Performance

- **Algorithm:** Gaussian Naive Bayes
- **Evaluation Metrics:** Accuracy Score, Confusion Matrix

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Mohamedhazem346/churn-prediction-.git
cd churn-prediction-
```

### 2. Install dependencies
```bash
pip install pandas scikit-learn seaborn matplotlib openpyxl
```

### 3. Add your dataset
Place `churn_dataset.xlsx` in the project root and update the path in the notebook:
```python
path = "churn_dataset.xlsx"
```

### 4. Run the notebook
```bash
jupyter notebook churn_prediction.ipynb
```

---

## 🔮 Example Prediction

```python
new_customer = [[35, 12, 1]]  # Age=35, Tenure=12 months, Sex=Male
prediction = model.predict(new_customer)
# Output: "Churn" or "No Churn"
```
---

## 🙋 Author

**Mohamed Hazem**
- GitHub: [@Mohamedhazem346](https://github.com/Mohamedhazem346)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).


