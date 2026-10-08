# Machine Learning Project

By Akhil Binjola

Two end-to-end ML case studies in Python (pandas, scikit-learn, XGBoost, matplotlib/seaborn).

## Part A – Airbnb Price Prediction (Regression)
File: `Machine_learning_Project_Part_A.ipynb`
Dataset: `Airbnb_data - airbnb_data.csv`

Predict log-transformed listing price (`log_price`) from property/room type, amenities, host features, location and reviews.
- Cleaning: datetime parsing, host_response_rate %, boolean flags (cleaning_fee, profile pic, verified, instant_bookable), median imputation
- Feature engineering: `amenity_count`, `host_duration`, `days_since_last_review`, `review_duration`, `has_review`
- Encoding: one-hot for property_type, room_type, bed_type, cancellation_policy, city, neighbourhood (~691 cols)
- Model: LinearRegression with train_test_split + SimpleImputer, evaluated with MAE / RMSE / R2, residual + actual-vs-predicted plots

Video explanation: https://drive.google.com/file/d/1dYe1Ju-aDV_yeAtG4guijOf4aQknr9yJ/view?usp=sharing

## Part B – Telecom Customer Churn Prediction (Classification)
File: `Machine_learning_Project_Part_B.ipynb`
Dataset: `Customer_data - customer_data.csv` (7043 rows x 21 cols)

Predict `Churn` from contract, tenure, charges, payment method and demographics.
- Cleaning: TotalCharges to numeric + median impute, drop customerID, Yes/No to 0/1
- Encoding: one-hot gender, InternetService, Contract, PaymentMethod
- Models: LogisticRegression (baseline ~82% acc, ROC AUC ~0.75), RandomForest, XGBoost
- Evaluation: accuracy, precision, recall, F1, ROC AUC, confusion matrix, stratified split

Video explanation: https://drive.google.com/file/d/1NJYrwbCPxx0_AdT_34GlH91ameroKMjK/view?usp=sharing

## How to run
1. Open notebooks in Colab / Jupyter
2. Upload the matching CSV (`/content/...` path in notebooks or edit path)
3. Run all cells top to bottom
