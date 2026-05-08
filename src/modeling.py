# src/modeling.py
# ── DCA Model Fitting Orchestration ─────────────────────────────────────────
# Runs the per-well Arps fitting loop, handles edge cases (insufficient
# history, all-fits-failed), and assembles the fit_summary dict consumed
# by forecast.py and the notebook's visualization cells.

import numpy as np
from sklearn.metrics import r2_score
from src.arps import fit_arps_models, select_best_model


# ── Fitting constants ────────────────────────────────────────────────────────
MIN_HISTORY_MONTHS = 6       # wells with fewer data points are skipped
OIL_COL            = "BORE_OIL_VOL"
T_COL              = "T_MONTHS"
WELL_COL           = "WELL_BORE_CODE"


def fit_all_wells(
    df,
    well_col: str  = WELL_COL,
    oil_col: str   = OIL_COL,
    t_col: str     = T_COL,
    min_pts: int   = MIN_HISTORY_MONTHS,
) -> dict:
    """
    Run Arps DCA fitting for every well in the dataset.

    For each well:
    - Extracts clean rate-time arrays (no NaNs)
    - Fits Exponential, Hyperbolic, and Harmonic models via curve_fit
    - Selects best model by AIC
    - Stores fit params, fitted rates, R², AIC table

    Parameters
    ----------
    df       : pd.DataFrame — cleaned, feature-engineered dataset
               (output of preprocess.run_preprocessing())
    well_col : str — well identifier column
    oil_col  : str — daily rate column (Sm³/day)
    t_col    : str — time-on-production column (months)
    min_pts  : int — minimum data points required to attempt a fit

    Returns
    -------
    fit_summary : dict
        Keys are well names. Each value is a dict with:
        {t, q_actual, best_model, params, pcov, model_fn,
         aic_table, q_fitted, r2, rss}
        Wells that fail all fits or have insufficient history
        are excluded with a printed warning.
    """
    wells = sorted(df[well_col].unique())
    fit_summary = {}

    for well in wells:
        sub = (df[df[well_col] == well]
               .sort_values(t_col)
               .dropna(subset=[t_col, oil_col]))

        t_arr = sub[t_col].values
        q_arr = sub[oil_col].values

        # Skip wells with insufficient production history
        if len(t_arr) < min_pts:
            print(f"WARNING {well}: insufficient history ({len(t_arr)} pts) - skipped")
            continue

        fit_results = fit_arps_models(t_arr, q_arr)

        # Skip wells where all three model fits failed
        valid = {k: v for k, v in fit_results.items() if v is not None}
        if not valid:
            print(f"WARNING {well}: all model fits failed - skipped")
            continue

        best_name, best = select_best_model(fit_results)

        fit_summary[well] = {
            "t":          t_arr,
            "q_actual":   q_arr,
            "best_model": best_name,
            "params":     best["params"],
            "pcov":       best["pcov"],
            "model_fn":   best["model_fn"],
            "aic_table":  {k: (v["aic"] if v else np.inf) for k, v in fit_results.items()},
            "q_fitted":   best["q_pred"],
            "r2":         r2_score(q_arr, best["q_pred"]),
            "rss":        best["rss"],
        }

        print(
            f"OK {well:30s} -> best: {best_name:12s} | "
            f"AIC: {best['aic']:8.1f} | R2: {r2_score(q_arr, best['q_pred']):.3f}"
        )

    print(f"\nOK Fitting complete - {len(fit_summary)}/{len(wells)} wells fitted")
    return fit_summary
