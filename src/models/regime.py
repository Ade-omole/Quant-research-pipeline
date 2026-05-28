"""Hidden Markov Model helpers for regime detection.

An HMM assumes the market has hidden states, such as calm, bullish, or crisis,
that we cannot observe directly. We infer those states from returns,
volatility, and momentum.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler


def _load_gaussian_hmm() -> Any:
    """Import GaussianHMM only when the regime model is actually fitted."""

    try:
        hmm_module = import_module("hmmlearn.hmm")
    except ImportError as error:
        raise ImportError(
            "The regime model requires `hmmlearn`. Install project dependencies "
            "with `python -m pip install -r requirements.txt`, then restart the "
            "kernel."
        ) from error

    return hmm_module.GaussianHMM


def build_regime_features(
    returns: pd.Series,
    realized_volatility: pd.Series,
    window: int = 21,
) -> pd.DataFrame:
    """Create a compact feature set that mixes direction and risk."""

    # Start with the same date index as the return series.
    feature_frame = pd.DataFrame(index=returns.index)
    # Daily return captures market direction.
    feature_frame["returns"] = returns
    # Realized volatility captures market risk/stress.
    feature_frame["realized_volatility"] = realized_volatility.reindex(returns.index)
    # Rolling average return gives a simple momentum feature.
    feature_frame["momentum"] = returns.rolling(window).mean()
    # HMM fitting cannot use missing values, so remove early rolling-window NaNs.
    return feature_frame.dropna(how="any")


def fit_hmm_regime_model(
    feature_frame: pd.DataFrame,
    n_regimes: int = 3,
    random_state: int = 7,
    *,
    covariance_type: str = "full",
    min_covar: float = 1e-3,
    n_iter: int = 500,
    tol: float = 1e-2,
    verbose: bool = False,
    standardize_features: bool = True,
) -> Any:
    """Fit a Gaussian HMM with ``n_regimes`` latent states (default 3)."""

    # GaussianHMM models each hidden state with its own multivariate normal distribution.
    GaussianHMM = _load_gaussian_hmm()
    hmm_kw: dict[str, Any] = dict(
        n_components=n_regimes,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
        min_covar=min_covar,
        tol=tol,
        verbose=verbose,
    )
    try:
        model = GaussianHMM(**hmm_kw)
    except TypeError:
        hmm_kw.pop("tol", None)
        hmm_kw.pop("verbose", None)
        model = GaussianHMM(**hmm_kw)
    scaler: StandardScaler | None = None
    values = np.asarray(feature_frame, dtype=np.float64)
    if standardize_features:
        scaler = StandardScaler()
        values = scaler.fit_transform(values)
    setattr(model, "_regime_feature_scaler", scaler)
    # Fit the HMM parameters: transition probabilities, state means, covariances.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings(
            "ignore",
            message=r".*[Nn]ot converging.*",
            category=UserWarning,
        )
        model.fit(values)
    return model


def infer_regimes(model: Any, feature_frame: pd.DataFrame) -> pd.Series:
    """Map HMM state labels back to the time index."""

    scaler: StandardScaler | None = getattr(model, "_regime_feature_scaler", None)
    values = np.asarray(feature_frame, dtype=np.float64)
    if scaler is not None:
        values = scaler.transform(values)
    # Predict the most likely hidden state for each date.
    regimes = model.predict(values)
    # Convert the state array into a labeled Series for easy plotting and joining.
    return pd.Series(regimes, index=feature_frame.index, name="regime")


def summarize_regimes(feature_frame: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
    """Describe each latent state using simple averages.

    This step helps us interpret anonymous state numbers. For example, a state
    with low returns and high volatility may be interpreted as crisis-like.
    """

    # Copy the feature table so we can add state labels without mutating the input.
    summary_frame = feature_frame.copy()
    # Attach the inferred state label to each date.
    summary_frame["regime"] = regimes
    # Compute mean, standard deviation, and count for each feature in each state.
    regime_summary = summary_frame.groupby("regime").agg(["mean", "std", "count"])
    return regime_summary
