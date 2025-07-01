# 🏠 House Price Prediction using Linear Regression

This project aims to predict house prices based on various features using a simple **Linear Regression** model. The dataset includes attributes such as area, number of bedrooms, bathrooms, and whether the house has features like air conditioning, a guest room, or a basement.

---

## 📁 Dataset

The dataset used is `Housing.csv`, which includes the following features:

- `area`
- `bedrooms`
- `bathrooms`
- `stories`
- `mainroad`
- `guestroom`
- `basement`
- `hotwaterheating`
- `airconditioning`
- `parking`
- `prefarea`
- `furnishingstatus`
- `price` (Target variable)

---

## 🔧 Feature Engineering

- Boolean text columns (`yes`/`no`) were converted to 1 and 0.
- Categorical feature `furnishingstatus` was label encoded.
- Missing values were dropped.
- Outliers were removed based on area and price thresholds.

---

## 📊 Data Visualization

Several visualizations were created for better data understanding:

- **Histogram** of house prices
- **Scatter plot** of Area vs Price
- **Correlation heatmap** of numerical features

---

## 🧠 Model Training

- Model Used: `LinearRegression` from `scikit-learn`
- Train/Test Split: 80% training, 20% testing

---

## 📈 Model Evaluation

- **Mean Squared Error (MSE):** _e.g.,_ `543829084.12`
- **R² Score:** _e.g.,_ `0.78`

A scatter plot of **Actual vs Predicted** prices was generated for visual comparison.

---

## 💾 Output

The predictions were saved in a CSV file:

```csv
Actual,Predicted
2135000,2181427.73
2940000,3371785.61
...