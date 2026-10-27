# Machine Learning Projects: Airbnb Price Prediction and Telecom Churn

Two end-to-end supervised learning projects built with Python and Scikit-learn.

| Part | Problem | Type | Best model |
|---|---|---|---|
| A | Predict Airbnb listing prices | Regression | XGBoost (R² 0.64) |
| B | Predict telecom customer churn | Binary classification | Logistic Regression (accuracy 82%) |

---

## Part A: Airbnb Rental Price Prediction

**Goal:** predict the log-transformed nightly price of a listing from its property, host, location and review features.

**Data:** 74,111 listings, 29 raw columns, which become 691 features after encoding.

**Approach**
1. Cleaned the data and filled missing numeric values with the median.
2. Engineered features: amenity count, host tenure and review recency.
3. One-hot encoded property type, room type, city and neighbourhood.
4. Trained a Linear Regression baseline, then XGBoost and HistGradientBoosting.
5. Tested a reduced feature set to see whether feature selection helped.

**Results** (log price target, hold-out test set)

| Model | R² | RMSE |
|---|---|---|
| Linear Regression (baseline) | 0.378 | 0.565 |
| HistGradientBoosting | 0.632 | 0.435 |
| **XGBoost** | **0.637** | **0.432** |
| XGBoost, reduced feature set | 0.327 | 0.588 |

**Takeaways**
- Tree-based models explain far more of the price variation than the linear baseline, which suggests non-linear relationships in the data.
- Dropping features hurt performance here, so the full feature set was kept.

---

## Part B: Telecom Customer Churn Prediction

**Goal:** identify customers likely to leave so retention teams can act early.

**Data:** 7,043 subscribers, 21 features. About 26% of customers churned.

**Approach**
1. Converted `TotalCharges` to numeric and filled gaps with the median.
2. One-hot encoded the categorical columns.
3. Explored churn against tenure and monthly charges.
4. Compared Logistic Regression, Random Forest and XGBoost.

**Results** (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC* |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.823** | 0.689 | 0.601 | 0.642 | 0.752 |
| XGBoost | 0.809 | 0.671 | 0.547 | 0.603 | 0.725 |
| Random Forest | 0.797 | 0.658 | 0.485 | 0.559 | 0.697 |

\*ROC-AUC here is computed from predicted class labels, not predicted probabilities, so it understates what a threshold-independent value would be. Recomputing it with `predict_proba` is a planned improvement.

**Takeaways**
- Logistic Regression gave the best balance on every metric and is also the easiest to explain to stakeholders.
- Churners tend to have shorter tenure and higher monthly charges.
- Recall is the weak spot for every model (about 49–60%), so the next steps are class weighting and threshold tuning.

---

## Tech stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Jupyter / Google Colab.

## How to run

```bash
git clone https://github.com/AkhilBinjola/Machine-Learning-Project
cd Machine-Learning-Project
pip install -r requirements.txt
jupyter notebook
```

Open the Part A or Part B notebook. The notebooks read their CSV files from `/content/` (Google Colab). To run locally, change the `pd.read_csv(...)` path to wherever you saved the dataset.

## Author

Akhil Binjola · [LinkedIn](https://linkedin.com/in/akhilbinjola) · [GitHub](https://github.com/AkhilBinjola)
