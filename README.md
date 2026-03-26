# COVID-19 Time Series Forecasting | Machine Learning and Deep Learning

## Problem Statement
Accurate forecasting of COVID-19 cases is essential for healthcare planning, resource allocation, and policy decision-making. This project aims to predict daily new COVID-19 cases using historical time series data.

---

## Objectives
- Transform time series data into supervised learning format using lag features  
- Build and compare multiple forecasting models  
- Evaluate model performance using regression metrics  
- Identify the most effective model for short-term forecasting  

---

## Dataset
- Source: Public COVID-19 dataset (United States daily cases)  
- Target Variable: Daily new confirmed cases  
- Features: Previous 14 days (lag features)  

---

## Tools and Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- Matplotlib  

---

## Approach

### 1. Data Preparation
- Converted time series into supervised format using lag features  
- Handled missing values  
- Scaled data appropriately to avoid leakage  

### 2. Model Building
- Linear Regression  
- Artificial Neural Network (ANN)  
- LSTM Neural Network  

### 3. Model Evaluation
- RMSE (Root Mean Squared Error)  
- MAE (Mean Absolute Error)  
- R² Score  

---

## Key Insights
- Linear Regression performed competitively despite its simplicity  
- ANN achieved the best overall performance  
- LSTM underperformed, indicating that complex models are not always necessary  
- Recent past values strongly influence short-term predictions  

---

## Business Impact
- Supports healthcare authorities in predicting case trends  
- Helps in planning hospital resources and staffing  
- Enables faster and simpler deployment using lightweight models  

---

## Results
- Best Model: Artificial Neural Network  
- Linear Regression showed comparable performance with lower complexity  
- LSTM did not outperform simpler models  

---

## How to Run
1. Clone the repository  
2. Install required libraries  
3. Run the notebook step-by-step  

---

## Future Improvements
- Incorporate external features such as mobility or vaccination data  
- Use advanced models like Prophet or XGBoost  
- Extend forecasting to multiple countries  

---

## Author
Rutuja Shinde
MSc Statistics Student | Aspiring Data Analyst  
GitHub: https://github.com/Rutuja1423
