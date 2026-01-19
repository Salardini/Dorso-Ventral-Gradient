"""
meg_axes/metrics.py

Parcel-wise metrics computation: tau (intrinsic timescale) and rho (rotational index).

Tau estimators:
    - ACF integral (primary): integral of ACF from lag_min to lag_max
    - Exponential fit (secondary): fit ACF to exp(-t/tau) and extract tau

Rho estimator:
    - Delay-embedded VAR(1) rotational index
    - Includes fit quality metric (one-step R²)

Additional QC metrics:
    - ts_var: time series variance
    - ts_rms: RMS amplitude (SNR proxy)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit


# =============================================================================
# Tau: Intrinsic Timescale
# =============================================================================

def autocorr_fft(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute autocorrelation function using FFT.

    Parameters
    ----------
    x : np.ndarray
        1D time series
    max_lag : int
        Maximum lag to compute

    Returns
    -------
    np.ndarray
        Normalized ACF from lag 0 to max_lag
    """
    x = x - np.mean(x)
    n = len(x)
    if n < 3:
        return np.zeros(max_lag + 1)

    # FFT-based autocorrelation
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]

    # Normalize by zero-lag (variance)
    if acf[0] != 0:
        acf = acf / acf[0]

    return acf[:max_lag + 1]


def tau_acf_integral(
    ts: np.ndarray,
    fs: float,
    lag_min_s: float = 0.005,
    lag_max_s: float = 0.300,
) -> float:
    """
    Compute intrinsic timescale as ACF integral.

    tau = integral from lag_min to lag_max of max(ACF(lag), 0) d(lag)

    Parameters
    ----------
    ts : np.ndarray
        1D time series (should be demeaned/standardized)
    fs : float
        Sampling frequency in Hz
    lag_min_s : float
        Minimum lag in seconds
    lag_max_s : float
        Maximum lag in seconds

    Returns
    -------
    float
        Timescale in seconds
    """
    max_lag = int(np.ceil(lag_max_s * fs))
    acf = autocorr_fft(ts, max_lag=max_lag)

    lags_s = np.arange(len(acf)) / fs
    mask = (lags_s >= lag_min_s) & (lags_s <= lag_max_s)

    # Clip negative ACF values (only integrate positive part)
    acf_pos = np.clip(acf[mask], 0.0, None)

    return float(trapezoid(acf_pos, lags_s[mask]))


def tau_exponential_fit(
    ts: np.ndarray,
    fs: float,
    max_lag_s: float = 0.300,
) -> Tuple[float, float, float]:
    """
    Compute intrinsic timescale by fitting ACF to exponential decay.

    ACF(t) = exp(-t / tau)

    Parameters
    ----------
    ts : np.ndarray
        1D time series
    fs : float
        Sampling frequency in Hz
    max_lag_s : float
        Maximum lag for fitting in seconds

    Returns
    -------
    tau : float
        Fitted timescale in seconds (NaN if fit fails)
    r_squared : float
        R² of the exponential fit
    rmse : float
        Root mean squared error of fit
    """
    max_lag = int(np.ceil(max_lag_s * fs))
    acf = autocorr_fft(ts, max_lag=max_lag)

    lags_s = np.arange(len(acf)) / fs

    # Skip lag 0 (always 1.0) for fitting
    lags_fit = lags_s[1:]
    acf_fit = acf[1:]

    # Only fit positive ACF values
    valid = acf_fit > 0
    if np.sum(valid) < 3:
        return np.nan, np.nan, np.nan

    lags_fit = lags_fit[valid]
    acf_fit = acf_fit[valid]

    def exp_decay(t, tau):
        return np.exp(-t / tau)

    try:
        # Initial guess: lag at which ACF drops to 1/e
        idx_e = np.argmin(np.abs(acf_fit - 1/np.e))
        tau_init = lags_fit[idx_e] if idx_e > 0 else 0.05

        popt, _ = curve_fit(
            exp_decay,
            lags_fit,
            acf_fit,
            p0=[tau_init],
            bounds=(1e-6, max_lag_s * 2),
            maxfev=1000,
        )
        tau = popt[0]

        # Compute fit quality
        acf_pred = exp_decay(lags_fit, tau)
        ss_res = np.sum((acf_fit - acf_pred) ** 2)
        ss_tot = np.sum((acf_fit - np.mean(acf_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rmse = np.sqrt(np.mean((acf_fit - acf_pred) ** 2))

        return float(tau), float(r_squared), float(rmse)

    except Exception:
        return np.nan, np.nan, np.nan


@dataclass
class TauResult:
    """Result container for tau computation."""
    tau_integral: float
    tau_exp: float
    tau_exp_r2: float
    tau_exp_rmse: float

    @property
    def primary(self) -> float:
        """Return primary tau estimate (integral method)."""
        return self.tau_integral


def compute_tau(
    ts: np.ndarray,
    fs: float,
    lag_min_s: float = 0.005,
    lag_max_s: float = 0.300,
) -> TauResult:
    """
    Compute tau using both estimators.

    Parameters
    ----------
    ts : np.ndarray
        1D time series
    fs : float
        Sampling frequency in Hz
    lag_min_s : float
        Minimum lag for integral method
    lag_max_s : float
        Maximum lag for both methods

    Returns
    -------
    TauResult
        Container with both estimates and QC metrics
    """
    tau_int = tau_acf_integral(ts, fs, lag_min_s, lag_max_s)
    tau_exp, r2, rmse = tau_exponential_fit(ts, fs, lag_max_s)

    return TauResult(
        tau_integral=tau_int,
        tau_exp=tau_exp,
        tau_exp_r2=r2,
        tau_exp_rmse=rmse,
    )


# =============================================================================
# Rho: Rotational Index
# =============================================================================

def delay_embed(x: np.ndarray, m: int, d: int) -> np.ndarray:
    """
    Create delay embedding matrix.

    Parameters
    ----------
    x : np.ndarray
        1D time series
    m : int
        Embedding dimension
    d : int
        Delay in samples

    Returns
    -------
    np.ndarray
        (T_eff, m) embedding matrix
    """
    T = len(x)
    T_eff = T - (m - 1) * d

    if T_eff <= 30:
        return np.empty((0, m))

    E = np.zeros((T_eff, m), dtype=np.float64)
    for k in range(m):
        start = (m - 1 - k) * d
        E[:, k] = x[start:start + T_eff]

    return E


def fit_var1_ridge(X: np.ndarray, ridge: float) -> np.ndarray:
    """
    Fit VAR(1) model with ridge regularization.

    X[t+1] = A @ X[t]

    Parameters
    ----------
    X : np.ndarray
        (T, p) state matrix
    ridge : float
        Ridge regularization parameter

    Returns
    -------
    np.ndarray
        (p, p) transition matrix A
    """
    X0 = X[:-1]
    X1 = X[1:]

    XtX = X0.T @ X0
    p = XtX.shape[0]

    # Ridge regression: A = (X0'X0 + ridge*I)^{-1} X0'X1
    A_T = np.linalg.solve(XtX + ridge * np.eye(p), X0.T @ X1)

    return A_T.T


def compute_var1_r2(X: np.ndarray, A: np.ndarray) -> float:
    """
    Compute one-step prediction R² for VAR(1) model.

    Parameters
    ----------
    X : np.ndarray
        (T, p) state matrix
    A : np.ndarray
        (p, p) transition matrix

    Returns
    -------
    float
        R² of one-step predictions
    """
    X0 = X[:-1]
    X1 = X[1:]

    X1_pred = X0 @ A.T

    # Total variance
    ss_tot = np.sum((X1 - X1.mean(axis=0)) ** 2)

    # Residual variance
    ss_res = np.sum((X1 - X1_pred) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1 - ss_res / ss_tot)


@dataclass
class RhoResult:
    """Result container for rho computation."""
    rho: float
    var1_r2: float
    n_eigenvalues_used: int

    @property
    def primary(self) -> float:
        """Return primary rho estimate."""
        return self.rho


def compute_rho(
    ts: np.ndarray,
    embed_dim: int = 10,
    embed_delay: int = 1,
    ridge_alpha: float = 1e-3,
    mag_min: float = 1e-2,
) -> RhoResult:
    """
    Compute rotational index from delay-embedded VAR(1).

    rho = mean(|imag(lambda)| / |lambda|) for eigenvalues with |lambda| > mag_min

    High rho indicates rotational/oscillatory dynamics.
    Low rho indicates decaying/stable dynamics.

    Parameters
    ----------
    ts : np.ndarray
        1D time series (should be standardized)
    embed_dim : int
        Embedding dimension
    embed_delay : int
        Delay in samples
    ridge_alpha : float
        Ridge regularization
    mag_min : float
        Minimum eigenvalue magnitude to include

    Returns
    -------
    RhoResult
        Container with rho and quality metrics
    """
    # Create delay embedding
    E = delay_embed(ts, m=embed_dim, d=embed_delay)

    if E.shape[0] < 30:
        return RhoResult(rho=np.nan, var1_r2=np.nan, n_eigenvalues_used=0)

    # Standardize each column
    E = (E - E.mean(axis=0)) / (E.std(axis=0) + 1e-8)

    # Fit VAR(1)
    A = fit_var1_ridge(E, ridge=ridge_alpha)

    # Compute eigenvalues
    lam = np.linalg.eigvals(A)
    mag = np.abs(lam)

    # Filter by magnitude
    keep = mag > mag_min

    if np.sum(keep) == 0:
        return RhoResult(rho=np.nan, var1_r2=np.nan, n_eigenvalues_used=0)

    # Rotational index: ratio of imaginary part to magnitude
    ratio = np.abs(np.imag(lam[keep])) / (mag[keep] + 1e-12)
    rho = float(np.mean(ratio))

    # Compute fit quality
    r2 = compute_var1_r2(E, A)

    return RhoResult(
        rho=rho,
        var1_r2=r2,
        n_eigenvalues_used=int(np.sum(keep)),
    )


# =============================================================================
# QC Metrics
# =============================================================================

def compute_ts_qc(ts: np.ndarray) -> Tuple[float, float]:
    """
    Compute time series quality control metrics.

    Parameters
    ----------
    ts : np.ndarray
        1D time series

    Returns
    -------
    ts_var : float
        Variance of time series
    ts_rms : float
        Root mean square (SNR proxy)
    """
    ts_var = float(np.var(ts))
    ts_rms = float(np.sqrt(np.mean(ts ** 2)))

    return ts_var, ts_rms


def standardize_ts(x: np.ndarray) -> np.ndarray:
    """
    Z-score normalize time series.

    Parameters
    ----------
    x : np.ndarray
        Input time series

    Returns
    -------
    np.ndarray
        Standardized time series (mean=0, std=1)
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    s = np.std(x)

    if not np.isfinite(s) or s == 0:
        return np.zeros_like(x)

    return x / s


def preprocess_parcel_ts(ts: np.ndarray) -> np.ndarray:
    """
    Preprocess parcel time series: detrend and standardize.

    Parameters
    ----------
    ts : np.ndarray
        Raw parcel time series

    Returns
    -------
    np.ndarray
        Preprocessed time series
    """
    # Linear detrend
    ts_detrend = signal.detrend(ts, type="linear")

    # Standardize
    return standardize_ts(ts_detrend)


# =============================================================================
# Batch Computation
# =============================================================================

@dataclass
class ParcelMetrics:
    """Complete metrics for one parcel."""
    parcel_idx: int
    label: str

    # Tau
    tau: float  # Primary (integral)
    tau_exp: float
    tau_exp_r2: float

    # Rho
    rho: float
    rho_r2: float

    # QC
    ts_var: float
    ts_rms: float


def compute_all_parcel_metrics(
    ts_matrix: np.ndarray,
    labels: list,
    fs: float,
    tau_lag_min_s: float = 0.005,
    tau_lag_max_s: float = 0.300,
    embed_dim: int = 10,
    embed_delay: int = 1,
    ridge_alpha: float = 1e-3,
    rho_mag_min: float = 1e-2,
) -> list:
    """
    Compute all metrics for all parcels.

    Parameters
    ----------
    ts_matrix : np.ndarray
        (n_parcels, n_time) parcel time series matrix
    labels : list
        Label names for each parcel
    fs : float
        Sampling frequency in Hz
    tau_lag_min_s, tau_lag_max_s : float
        Tau integration window
    embed_dim, embed_delay : int
        Rho embedding parameters
    ridge_alpha : float
        Ridge regularization for VAR(1)
    rho_mag_min : float
        Minimum eigenvalue magnitude

    Returns
    -------
    list of ParcelMetrics
        Metrics for each parcel
    """
    n_parcels = ts_matrix.shape[0]
    results = []

    for i in range(n_parcels):
        ts_raw = ts_matrix[i]

        # Preprocess
        ts = preprocess_parcel_ts(ts_raw)

        # QC metrics (on preprocessed ts)
        ts_var, ts_rms = compute_ts_qc(ts)

        # Tau
        tau_result = compute_tau(ts, fs, tau_lag_min_s, tau_lag_max_s)

        # Rho
        rho_result = compute_rho(ts, embed_dim, embed_delay, ridge_alpha, rho_mag_min)

        results.append(ParcelMetrics(
            parcel_idx=i,
            label=labels[i] if i < len(labels) else f"parcel_{i}",
            tau=tau_result.tau_integral,
            tau_exp=tau_result.tau_exp,
            tau_exp_r2=tau_result.tau_exp_r2,
            rho=rho_result.rho,
            rho_r2=rho_result.var1_r2,
            ts_var=ts_var,
            ts_rms=ts_rms,
        ))

    return results
