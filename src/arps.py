# src/arps.py
# ── Arps Decline Curve Model Definitions ────────────────────────────────────
# Contains the three Arps model functions, the AIC scorer, and the unified
# fit_arps_models() dispatcher used by the forecasting pipeline.

import numpy as np
from scipy.optimize import curve_fit


def exponential_decline(t, qi, Di):
    """
    Exponential decline (b=0).
    Most conservative EUR estimate. Appropriate for boundary-dominated
    flow in homogeneous reservoirs.

    Parameters
    ----------
    t  : array-like — time on production (months)
    qi : float      — initial production rate (Sm³/day)
    Di : float      — initial decline rate (1/month)
    """
    return qi * np.exp(-Di * t)


def hyperbolic_decline(t, qi, Di, b):
    """
    Hyperbolic decline (0 < b <= 1).
    Most physically representative model for conventional reservoirs.
    b capped at 1.0 for conventional sandstone (Volve Hugin Formation).
    Values above 1.0 imply accelerating production — physically unrealistic
    for this reservoir type.

    Parameters
    ----------
    t  : array-like — time on production (months)
    qi : float      — initial production rate (Sm³/day)
    Di : float      — initial decline rate (1/month)
    b  : float      — hyperbolic exponent (0 < b <= 1)
    """
    return qi / (1.0 + b * Di * t) ** (1.0 / b)


def harmonic_decline(t, qi, Di):
    """
    Harmonic decline (b=1).
    Most optimistic model. Appropriate for strong water drive where
    pressure support extends late-life production.

    Parameters
    ----------
    t  : array-like — time on production (months)
    qi : float      — initial production rate (Sm³/day)
    Di : float      — initial decline rate (1/month)
    """
    return qi / (1.0 + Di * t)


def compute_aic(n_pts, rss, k_params):
    """
    Akaike Information Criterion — penalises model complexity relative
    to goodness of fit. Lower AIC = better model.

    AIC = n * ln(RSS/n) + 2k

    Parameters
    ----------
    n_pts    : int   — number of data points
    rss      : float — residual sum of squares
    k_params : int   — number of free parameters in model
    """
    if rss <= 0 or n_pts <= k_params:
        return np.inf
    return n_pts * np.log(rss / n_pts) + 2 * k_params


def fit_arps_models(t, q):
    """
    Fit all three Arps models to rate-time data and return results
    keyed by model name.

    Uses SciPy curve_fit with Trust Region Reflective (trf) algorithm —
    robust to noisy production data with bounded parameter spaces.

    Parameters
    ----------
    t : array-like — time on production (months)
    q : array-like — oil production rate (Sm³/day)

    Returns
    -------
    dict with keys: 'exponential', 'hyperbolic', 'harmonic'
    Each entry: {params, pcov, q_pred, rss, aic, model_fn}
    None entries indicate a failed fit (RuntimeError from curve_fit).
    """
    t = np.asarray(t, dtype=float)
    q = np.asarray(q, dtype=float)

    model_specs = {
        "exponential": {
            "fn":     exponential_decline,
            "p0":     [q.max(), 0.05],
            "bounds": ([0, 1e-6], [np.inf, 5.0]),
            "k":      2,
        },
        "hyperbolic": {
            "fn":     hyperbolic_decline,
            "p0":     [q.max(), 0.1, 0.5],
            "bounds": ([0, 1e-6, 1e-6], [np.inf, 5.0, 1.0]),
            "k":      3,
        },
        "harmonic": {
            "fn":     harmonic_decline,
            "p0":     [q.max(), 0.05],
            "bounds": ([0, 1e-6], [np.inf, 5.0]),
            "k":      2,
        },
    }

    results = {}
    for name, spec in model_specs.items():
        try:
            popt, pcov = curve_fit(
                spec["fn"], t, q,
                p0=spec["p0"],
                bounds=spec["bounds"],
                maxfev=15000,
                method="trf",
            )
            q_pred = spec["fn"](t, *popt)
            rss    = float(np.sum((q - q_pred) ** 2))
            aic    = compute_aic(len(t), rss, spec["k"])

            results[name] = {
                "params":   popt,
                "pcov":     pcov,
                "q_pred":   q_pred,
                "rss":      rss,
                "aic":      aic,
                "model_fn": spec["fn"],
            }
        except RuntimeError:
            results[name] = None    # fit failed — flagged in summary table

    return results


def select_best_model(fit_results):
    """
    Select the best-fitting Arps model by minimum AIC.

    Parameters
    ----------
    fit_results : dict — output of fit_arps_models()

    Returns
    -------
    best_name : str  — name of best model
    best      : dict — corresponding result entry
    """
    valid = {k: v for k, v in fit_results.items() if v is not None}
    if not valid:
        raise ValueError("All Arps model fits failed for this well.")
    best_name = min(valid, key=lambda k: valid[k]["aic"])
    return best_name, valid[best_name]