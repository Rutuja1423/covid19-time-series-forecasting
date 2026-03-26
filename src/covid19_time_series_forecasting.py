# COVID-19 Time Series Forecasting in the United States
# Models Compared:
# 1. Linear Regression
# 2. Artificial Neural Network (ANN)
# 3. LSTM Neural Network

import os
import random
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress most TensorFlow warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = "data\\covid-data.csv"
COUNTRY = "United States"
TARGET_COLUMN = "new_cases"
DATE_COLUMN = "date"
N_LAGS = 14
TRAIN_RATIO = 0.70
OUTPUT_PREDICTION_PLOT = "covid19_model_predictions.png"
OUTPUT_ANN_LOSS_PLOT = "ann_loss_curve.png"
OUTPUT_LSTM_LOSS_PLOT = "lstm_loss_curve.png"
OUTPUT_RESULTS_CSV = "model_comparison_results.csv"

def load_and_prepare_data(file_path, country, date_col, target_col):
    """
    Load COVID dataset, filter by country, and prepare the target series.
    """
    data = pd.read_csv(file_path)
    data[date_col] = pd.to_datetime(data[date_col])

    country_data = data[data["location"] == country].copy()
    country_data = country_data[[date_col, target_col]].copy()
    country_data.sort_values(date_col, inplace=True)
    country_data.reset_index(drop=True, inplace=True)

    # Fill missing values and clip negatives
    country_data[target_col] = country_data[target_col].fillna(0)
    country_data[target_col] = country_data[target_col].clip(lower=0)

    return country_data

def create_lag_features(data, target_col, n_lags):
    """
    Create lag-based supervised learning features.
    Example:
    new_cases_lag_1, new_cases_lag_2, ..., new_cases_lag_14
    """
    df = data.copy()

    for lag in range(1, n_lags + 1):
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

# TRAIN-TEST SPLIT
def split_time_series_data(data, feature_cols, target_col, date_col, train_ratio):
    """
    Split the data chronologically into train and test sets.
    """
    split_index = int(len(data) * train_ratio)

    X_train = data.loc[:split_index - 1, feature_cols]
    X_test = data.loc[split_index:, feature_cols]

    y_train = data.loc[:split_index - 1, target_col]
    y_test = data.loc[split_index:, target_col]

    test_dates = data.loc[split_index:, date_col]

    return X_train, X_test, y_train, y_test, test_dates

def scale_features_and_target(X_train, X_test, y_train, y_test):
    """
    Fit scalers only on training data to avoid data leakage.
    """
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1))

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, x_scaler, y_scaler

# MODEL BUILDERS
def build_ann_model(input_dim):
    """
    Build a feedforward neural network for regression.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def build_lstm_model(n_lags):
    """
    Build an LSTM model for time series regression.
    """
    model = Sequential([
        Input(shape=(n_lags, 1)),
        LSTM(50),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# EVALUATION FUNCTION
def evaluate_predictions(model_name, y_true, y_pred):
    """
    Calculate regression metrics and return them as a dictionary.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- {model_name} ---")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    if r2 >= 0.80:
        interpretation = "Strong performance. The model captures much of the variation in daily new cases."
    elif r2 >= 0.50:
        interpretation = "Moderate performance. The model captures general trends but misses some fluctuations."
    else:
        interpretation = "Weak performance. The model struggles to track daily case movements accurately."

    print("Interpretation:", interpretation)

    return {
        "Model": model_name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

# PLOT FUNCTIONS
def plot_predictions(dates, actual, linear_pred, ann_pred, lstm_pred, output_path):
    """
    Plot actual vs predicted values for all models.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(dates, actual, label="Actual")
    plt.plot(dates, linear_pred, label="Linear Regression")
    plt.plot(dates, ann_pred, label="ANN")
    plt.plot(dates, lstm_pred, label="LSTM")

    plt.title("COVID-19 New Cases Prediction in the United States")
    plt.xlabel("Date")
    plt.ylabel("Daily New Cases")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()


def plot_loss_curve(history, title, output_path):
    """
    Plot training and validation loss curves.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

# MAIN WORKFLOW
def main():
    print("Loading and preparing data...")
    covid_data = load_and_prepare_data(
        file_path=DATA_PATH,
        country=COUNTRY,
        date_col=DATE_COLUMN,
        target_col=TARGET_COLUMN
    )

    print(f"Total rows for {COUNTRY}: {len(covid_data)}")
    print(f"Date range: {covid_data[DATE_COLUMN].min().date()} to {covid_data[DATE_COLUMN].max().date()}")

    supervised_data = create_lag_features(covid_data, TARGET_COLUMN, N_LAGS)

    feature_columns = [f"{TARGET_COLUMN}_lag_{lag}" for lag in range(1, N_LAGS + 1)]

    X_train, X_test, y_train, y_test, test_dates = split_time_series_data(
        data=supervised_data,
        feature_cols=feature_columns,
        target_col=TARGET_COLUMN,
        date_col=DATE_COLUMN,
        train_ratio=TRAIN_RATIO
    )

    print("\nTrain-test split completed.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, _, y_scaler = scale_features_and_target(
        X_train, X_test, y_train, y_test
    )

    results_summary = []

    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train_scaled)

    linear_pred_scaled = linear_model.predict(X_test_scaled)
    linear_pred = y_scaler.inverse_transform(linear_pred_scaled).flatten()

    results_summary.append(
        evaluate_predictions("Linear Regression", y_test.values, linear_pred)
    )

    # ANN
    ann_model = build_ann_model(X_train_scaled.shape[1])

    ann_early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    ann_history = ann_model.fit(
        X_train_scaled,
        y_train_scaled,
        validation_data=(X_test_scaled, y_test_scaled),
        epochs=100,
        batch_size=16,
        verbose=0,
        callbacks=[ann_early_stopping]
    )

    ann_pred_scaled = ann_model.predict(X_test_scaled, verbose=0)
    ann_pred = y_scaler.inverse_transform(ann_pred_scaled).flatten()

    results_summary.append(
        evaluate_predictions("Artificial Neural Network (ANN)", y_test.values, ann_pred)
    )

    # LSTM
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], N_LAGS, 1))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], N_LAGS, 1))

    lstm_model = build_lstm_model(N_LAGS)

    lstm_early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    lstm_history = lstm_model.fit(
        X_train_lstm,
        y_train_scaled,
        validation_data=(X_test_lstm, y_test_scaled),
        epochs=100,
        batch_size=16,
        verbose=0,
        shuffle=False,
        callbacks=[lstm_early_stopping]
    )

    lstm_pred_scaled = lstm_model.predict(X_test_lstm, verbose=0)
    lstm_pred = y_scaler.inverse_transform(lstm_pred_scaled).flatten()

    results_summary.append(
        evaluate_predictions("LSTM Neural Network", y_test.values, lstm_pred)
    )

    # Results Summary
    results_df = pd.DataFrame(results_summary).sort_values(by="RMSE", ascending=True)

    print("\n=========================")
    print("MODEL COMPARISON SUMMARY")
    print("=========================")
    print(results_df.to_string(index=False))

    results_df.to_csv(OUTPUT_RESULTS_CSV, index=False)
    print(f"\nSaved model comparison table as '{OUTPUT_RESULTS_CSV}'")

    # Visualizations
    plot_predictions(
        dates=test_dates,
        actual=y_test.values,
        linear_pred=linear_pred,
        ann_pred=ann_pred,
        lstm_pred=lstm_pred,
        output_path=OUTPUT_PREDICTION_PLOT
    )
    print(f"Saved prediction plot as '{OUTPUT_PREDICTION_PLOT}'")

    plot_loss_curve(
        history=ann_history,
        title="ANN Training vs Validation Loss",
        output_path=OUTPUT_ANN_LOSS_PLOT
    )
    print(f"Saved ANN loss curve as '{OUTPUT_ANN_LOSS_PLOT}'")

    plot_loss_curve(
        history=lstm_history,
        title="LSTM Training vs Validation Loss",
        output_path=OUTPUT_LSTM_LOSS_PLOT
    )
    print(f"Saved LSTM loss curve as '{OUTPUT_LSTM_LOSS_PLOT}'")

    # Final Interpretation
    best_model = results_df.iloc[0]["Model"]

    print("\n=========================")
    print("FINAL INTERPRETATION")
    print("=========================")
    print(
        f"This project compares Linear Regression, ANN, and LSTM models for forecasting "
        f"daily COVID-19 new cases in the United States using the previous {N_LAGS} days as lag features."
    )
    print(f"Based on RMSE, the best-performing model in this run is: {best_model}.")
    print(
        "The results show that simpler and feedforward models can outperform sequential deep learning models "
        "depending on the dataset structure, lag design, and forecasting horizon."
    )
