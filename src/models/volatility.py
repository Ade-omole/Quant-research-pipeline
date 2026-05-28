"""Volatility estimation models used in project one and strategy one.

This module contains two volatility views:
1. a machine-learning forecast based on engineered features,
2. a GARCH model based on classic financial time-series theory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def split_train_test(feature_frame: pd.DataFrame, train_fraction: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the feature table in time order to avoid look-ahead bias."""

    # Compute the row number where the training sample ends.
    split_index = int(len(feature_frame) * train_fraction)
    # The first block is the historical training period.
    train_frame = feature_frame.iloc[:split_index].copy()
    # The later block is the out-of-sample testing period.
    test_frame = feature_frame.iloc[split_index:].copy()
    return train_frame, test_frame


def fit_random_forest_volatility_model(
    train_frame: pd.DataFrame,
    random_state: int = 7,
) -> RandomForestRegressor:
    """Fit a simple nonlinear benchmark for next-period volatility.

    A random forest is useful here because it can capture nonlinear
    relationships without much manual specification.
    """

    # Every column except the target is an input feature.
    feature_columns = [column for column in train_frame.columns if column != "target_volatility"]
    # Limit tree depth so the model stays less overfit.
    model = RandomForestRegressor(
        n_estimators=300, # set number of trees to 300.
        max_depth=4, # limit tree depth so the model stays less overfit.
        random_state=random_state, # set random state for reproducibility.
    )
    # Fit the model to predict next-period realized volatility.
    model.fit(train_frame[feature_columns], train_frame["target_volatility"])
    return model


def predict_random_forest_volatility(
    model: RandomForestRegressor,
    feature_frame: pd.DataFrame,
) -> pd.Series:
    """Generate predictions while keeping the original index.

    Keeping the time index is important because later backtests align forecasts
    with returns by date.
    """

    # Use the same feature columns that were available during training.
    feature_columns = [column for column in feature_frame.columns if column != "target_volatility"]
    # Ask the fitted model for a volatility forecast on each row.
    predictions = model.predict(feature_frame[feature_columns])
    # Wrap the numpy array back into a labeled pandas Series.
    return pd.Series(predictions, index=feature_frame.index, name="rf_volatility_forecast")


def fit_garch_model(returns: pd.Series):
    """Fit a GARCH(1,1) model to returns expressed in percent."""

    # Convert decimal returns like 0.01 into percent returns like 1.0.
    scaled_returns = returns.dropna() * 100
    # `mean="Zero"` keeps the model focused on volatility rather than drift.
    model = arch_model(scaled_returns, vol="Garch", p=1, q=1, mean="Zero")
    # Fit quietly because callers should show chosen outputs, not optimizer logs.
    fitted_model = model.fit(disp="off")
    return fitted_model


def extract_garch_conditional_volatility(fitted_model, index: pd.Index) -> pd.Series:
    """Convert the fitted conditional volatility back to decimal units."""

    # The model was fit on percent returns, so divide by 100 to return to decimals.
    conditional_volatility = fitted_model.conditional_volatility / 100
    # Use the tail of the original index in case the fitted model dropped early NaNs.
    aligned_volatility = pd.Series(conditional_volatility, index=index[-len(conditional_volatility) :])
    return aligned_volatility.rename("garch_conditional_volatility")


def evaluate_forecasts(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Return MSE, MAE, and correlation for aligned forecast evaluation."""

    # Align by date so we only compare observations that exist in both series.
    aligned_actual, aligned_predicted = actual.align(predicted, join="inner")
    # Build one table so we can drop any row where either side is missing.
    evaluation_frame = pd.concat(
        [aligned_actual.rename("actual"), aligned_predicted.rename("predicted")],
        axis=1,
    ).dropna(how="any")
    # If there are no valid overlapping observations, fail with a clear message.
    if evaluation_frame.empty:
        raise ValueError("No overlapping non-missing observations are available for forecast evaluation.")
    # Mean squared error emphasizes large forecast misses.
    mse_value = mean_squared_error(evaluation_frame["actual"], evaluation_frame["predicted"])
    # Mean absolute error measures the average absolute forecast miss.
    mae_value = mean_absolute_error(evaluation_frame["actual"], evaluation_frame["predicted"])
    # Correlation measures directional co-movement between actual and forecast.
    correlation_value = np.corrcoef(evaluation_frame["actual"], evaluation_frame["predicted"])[0, 1]
    return {
        "mse": float(mse_value),
        "mae": float(mae_value),
        "correlation": float(correlation_value),
    }
