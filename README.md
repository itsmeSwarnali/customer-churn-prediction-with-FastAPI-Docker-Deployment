# customer-churn-prediction
Customer Churn Prediction | Machine Learning, FastAPI & Docker
An end-to-end machine learning project that predicts customer churn, exposes predictions through a FastAPI REST API, and is fully containerized using Docker.


## Run the APP with Docker
### 1. Clone the repository

```bash
git clone https://github.com/itsmeSwarnali/customer-churn-prediction-with-FastAPI-Docker-Deployment.git
cd customer-churn-prediction-with-FastAPI-Docker-Deployment
```

2. Build Docker image
```bash
docker build -t churn-api .
```

3. Run Docker container
```bash  
docker run -p 8000:8000 churn-api
```

4. Open the API docs 
http://localhost:8000/docs

Then test /predict from Swagger UI.

![The successful Swagger response](Images/predict.png)


# Customer Churn Prediction

## Business Problem

Customer churn is a major challenge for subscription-based businesses. This project aims to predict whether a customer is likely to leave a telecom company using machine learning techniques.

## Dataset

Telco Customer Churn Dataset

- 7,043 customers
- 23 features
- Binary target variable: Churn

## Exploratory Data Analysis

## Churn Distribution

![Churn Distribution](Images/churn_distribution.png)

## Payment Method vs Churn

![Churn vs Payment Type](Images/churn_vs_Payment_Type.png)

![Churn vs Contract Type](Images/churn_vs_contract_type.png)

![Churn vs Internet Service](Images/churn_vs_internet_service.png)

![Churn Rate vs Monthly Charge Rate](Images/churn_rate_vs_monthly_charge_rate.png)


Key findings:

- Customers on month-to-month contracts have significantly higher churn rates.
- Electronic Check users exhibit the highest churn rate (45.29%).
- Customers without technical support are more likely to churn.
- Customers without online security services are more likely to leave.
- Customers without dependents show increased churn risk.

## Models Evaluated

| Model | Accuracy | Recall | F1 | ROC-AUC |
|---------|---------:|---------:|---------:|---------:|
| Logistic Regression | 0.8041 | 0.5668 | 0.6057 | 0.8409 |
| Random Forest | 0.7913 | 0.5106 | 0.5685 | 0.8419 |
| XGBoost | 0.78426 | 0.5053 | 0.5543 | 0.8242 |

## Best Model

Logistic Regression achieved the best overall performance with a ROC-AUC score of **0.8454**..

## Feature Importance

![Feature Importance](Images/feature_imp.png)

## Key Business Recommendations

- Convert month-to-month customers into long-term contracts.
- Promote automatic payment methods.
- Encourage adoption of technical support services.
- Bundle online security with internet plans.
- Design retention campaigns for high-risk customer segments.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
