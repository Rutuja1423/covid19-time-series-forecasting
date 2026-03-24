# COVID-19 Time Series Forecasting

## Project Overview

This project focuses on forecasting daily COVID 19 new cases in the United States using time series modeling techniques.
It compares traditional machine learning and deep learning models to understand their effectiveness in predicting real world sequential data.

The models implemented in this project:

* Linear Regression
* Artificial Neural Network (ANN)
* Long Short Term Memory (LSTM)

The goal is to evaluate how well different models can capture trends and fluctuations in pandemic data.

---

## Dataset

* Source: Public COVID 19 dataset (Our World in Data)
* Country: United States
* Time Period: January 2020 to May 2021
* Target Variable: Daily New Cases

Missing values were handled by replacing them with zero, and negative values were clipped to ensure data consistency.

---

## Methodology

### 1. Time Series Transformation

The dataset was converted into a supervised learning problem using lag features:

* Previous 14 days of cases used as input features
* Current day cases used as target

### 2. Train-Test Split

* 70% Training Data
* 30% Testing Data
* Chronological split (no shuffling)

### 3. Data Scaling

* MinMaxScaler used
* Fitted only on training data to avoid data leakage

---

## Models Implemented

### Linear Regression

* Baseline model
* Captures linear relationships in lagged features

### Artificial Neural Network (ANN)

* Feedforward neural network
* Captures non linear relationships

### LSTM Neural Network

* Designed for sequential data
* Captures temporal dependencies

---

## Evaluation Metrics

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R² Score

---

## Results Summary

| Model             | RMSE  | MAE   | R²    |
| ----------------- | ----- | ----- | ----- |
| ANN               | 23946 | 15964 | 0.875 |
| Linear Regression | 24032 | 15071 | 0.874 |
| LSTM              | 31426 | 22116 | 0.785 |

---

## Key Insights

* ANN achieved the best performance with the lowest RMSE
* Linear Regression performed nearly as well as ANN
* LSTM underperformed despite being a sequential model
* This shows that more complex models do not always guarantee better results

---

## Visualizations

### Model Predictions

* Actual vs Predicted values plotted for all models

### Loss Curves

* Training vs Validation loss for ANN and LSTM

---

## Project Structure

```
covid19-time-series-forecasting/
│
├── data/
│   └── covid-data.csv
│
├── src/
│   └── covid19_time_series_forecasting.py
│
├── outputs/
│   ├── covid19_model_predictions.png
│   ├── ann_loss_curve.png
│   ├── lstm_loss_curve.png
│   └── model_comparison_results.csv
│
├── README.md
└── requirements.txt
```

---

## How to Run the Project

1. Clone the repository:

```
git clone https://github.com/Rutuja1423/covid19-time-series-forecasting.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the script:

```
python src/covid19_time_series_forecasting.py
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit learn
* TensorFlow / Keras

---

## Conclusion

This project demonstrates how different modeling techniques behave on time series data.
It highlights that simpler models can sometimes outperform complex deep learning models depending on data characteristics and feature engineering.

---

## Future Improvements

* Use more recent COVID 19 data
* Try advanced models like GRU or Prophet
* Hyperparameter tuning
* Cross validation for time series

---

## Author

Rutuja Shinde

