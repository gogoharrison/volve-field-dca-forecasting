# src/forecast.py
# ── EUR Computation, Forward Forecasting & Export ───────────────────────────
# Handles all post-fit operations: per-well forward forecast generation,
# economic limit application, EUR computation, unit conversion,
# field-level aggregation, and results export.

import numpy as np
import pandas as pd

# ── Forecasting constants ────────────────────────────────────────────────────
DAYS_PER_MONTH   = 30.4375      # average days/month
SMC3_TO_BBL      = 6.2898       # 1 Sm³ = 6.2898 bbl (Norwegian standard)
Q_ECONOMIC_LIMIT = 5.0          # Sm³/day — production below this is uneconomic
FORECAST_MONTHS  = 60           # 5-year forward horizon


def forecast_well(
    well: str,
    fit_info: dict,
    hist_df: pd.DataFrame,
    oil_monthly_col: str = "OIL_MONTHLY",
    oil_rate_col: str    = "BORE_OIL_VOL",
    t_col: str           = "T_MONTHS",
    q_limit: float       = Q_ECONOMIC_LIMIT,
    n_months: int        = FORECAST_MONTHS,
) -> dict:
    """
    Generate a forward production forecast and compute EUR for one well.

    EUR = cumulative historical production (from OIL_MONTHLY)
          + cumulative forecast production (Arps model integral, monthly steps)

    The economic limit (q_limit) truncates forecast volumes below the
    minimum economic rate — production below this threshold is uneconomic
    and not included in EUR.

    Parameters
    ----------
    well         : str  — well identifier
    fit_info     : dict — entry from fit_summary (output of Cell 7)
    hist_df      : pd.DataFrame — cleaned production data for this well
    oil_monthly_col : str — column name for actual monthly volume (Sm³)
    oil_rate_col    : str — column name for daily rate (Sm³/day)
    t_col           : str — column name for time-on-production (months)
    q_limit      : float — economic limit (Sm³/day), default 5.0
    n_months     : int   — forecast horizon in months, default 60

    Returns
    -------
    dict with keys:
        t_future, q_future       — forecast time axis and rates
        cum_hist_sm3             — historical cumulative (Sm³)
        cum_fore_sm3             — forecast cumulative (Sm³)
        eur_sm3, eur_bbl         — EUR in both units
        economic_limit_month     — month index where rate drops below q_limit
    """
    t_hist = fit_info["t"]
    t_future = t_hist[-1] + np.arange(1, n_months + 1, dtype=float)

    # Evaluate best-fit Arps model at future time steps
    q_future = fit_info["model_fn"](t_future, *fit_info["params"])

    # Apply economic cutoff
    economic_limit_month = None
    below_limit = np.where(q_future < q_limit)[0]
    if len(below_limit) > 0:
        economic_limit_month = int(t_future[below_limit[0]])
    q_future = np.where(q_future < q_limit, 0.0, q_future)

    # Historical cumulative from actual OIL_MONTHLY (most accurate)
    cum_hist_sm3 = float(hist_df[oil_monthly_col].sum())

    # Forecast cumulative: rate (Sm³/day) × avg days per month
    cum_fore_sm3 = float(np.sum(q_future * DAYS_PER_MONTH))

    eur_sm3 = cum_hist_sm3 + cum_fore_sm3
    eur_bbl = eur_sm3 * SMC3_TO_BBL

    return {
        "well":                  well,
        "best_model":            fit_info["best_model"],
        "t_future":              t_future,
        "q_future":              q_future,
        "cum_hist_sm3":          cum_hist_sm3,
        "cum_fore_sm3":          cum_fore_sm3,
        "eur_sm3":               eur_sm3,
        "eur_bbl":               eur_bbl,
        "economic_limit_month":  economic_limit_month,
        "dca_r2":                fit_info["r2"],
    }


def run_all_forecasts(
    fit_summary: dict,
    df: pd.DataFrame,
    well_col: str        = "WELL_BORE_CODE",
    oil_monthly_col: str = "OIL_MONTHLY",
    q_limit: float       = Q_ECONOMIC_LIMIT,
    n_months: int        = FORECAST_MONTHS,
) -> dict:
    """
    Run forecast_well() for all wells in fit_summary and print a
    field-level EUR summary.

    Parameters
    ----------
    fit_summary     : dict — output of Cell 7 (model fitting loop)
    df              : pd.DataFrame — full cleaned dataset
    well_col        : str  — well identifier column
    oil_monthly_col : str  — actual monthly volume column
    q_limit         : float — economic limit (Sm³/day)
    n_months        : int   — forecast horizon (months)

    Returns
    -------
    dict — {well_name: forecast_result_dict}
    """
    forecast_results = {}

    for well, info in fit_summary.items():
        hist_df = df[df[well_col] == well].sort_values("T_MONTHS")
        result  = forecast_well(
            well, info, hist_df,
            oil_monthly_col=oil_monthly_col,
            q_limit=q_limit,
            n_months=n_months,
        )
        forecast_results[well] = result
        print(
            f"{well:30s} | EUR: {result['eur_sm3']/1e6:.3f} MMSm3  "
            f"({result['eur_bbl']/1e6:.2f} MMbbl)"
        )

    # Field total
    total_sm3 = sum(v["eur_sm3"] for v in forecast_results.values())
    total_bbl = sum(v["eur_bbl"] for v in forecast_results.values())
    print(f"\n{'-'*65}")
    print(f"{'FIELD TOTAL EUR':30s} | {total_sm3/1e6:.3f} MMSm3  ({total_bbl/1e6:.2f} MMbbl)")
    print(f"(Volve reported: ~63 MMbbl)")

    return forecast_results


def build_summary_table(fit_summary: dict, forecast_results: dict,
                        ml_results: dict = None) -> pd.DataFrame:
    """
    Compile a per-well summary DataFrame combining DCA fit quality,
    EUR estimates, and optional ML benchmark metrics.

    Parameters
    ----------
    fit_summary      : dict — model fitting results
    forecast_results : dict — EUR and forecast outputs
    ml_results       : dict — optional ML benchmark results

    Returns
    -------
    pd.DataFrame — one row per well + FIELD TOTAL row
    """
    def format_percent(value):
        return "-" if pd.isna(value) else f"{value:.1%}"

    rows = []
    for well in fit_summary:
        if well not in forecast_results:
            continue
        ml = (ml_results or {}).get(well, {})
        fore = forecast_results[well]
        rows.append({
            "Well":               well.replace("NO 15/9-", ""),
            "Best DCA Model":     fit_summary[well]["best_model"].capitalize(),
            "DCA R2":             round(fit_summary[well]["r2"], 3),
            "DCA MAPE":           format_percent(ml.get("dca_mape")),
            "GBR MAPE (CV)":      format_percent(ml.get("ml_mape_cv")),
            "Hist. Prod (kSm3)":  f"{fore['cum_hist_sm3']/1e3:.1f}",
            "Forecast EUR (kSm3)":f"{fore['eur_sm3']/1e3:.1f}",
            "EUR (MMbbl)":        round(fore["eur_bbl"] / 1e6, 3),
        })

    summary_df = pd.DataFrame(rows).set_index("Well")

    # Append field total row
    total_sm3 = sum(forecast_results[w]["eur_sm3"] for w in forecast_results)
    total_bbl = sum(forecast_results[w]["eur_bbl"] for w in forecast_results)
    totals = pd.DataFrame([{
        "Well":               "FIELD TOTAL",
        "Best DCA Model":     "-",
        "DCA R2":             "-",
        "DCA MAPE":           "-",
        "GBR MAPE (CV)":      "-",
        "Hist. Prod (kSm3)":  f"{sum(forecast_results[w]['cum_hist_sm3'] for w in forecast_results)/1e3:.1f}",
        "Forecast EUR (kSm3)":f"{total_sm3/1e3:.1f}",
        "EUR (MMbbl)":        round(total_bbl / 1e6, 3),
    }]).set_index("Well")

    return pd.concat([summary_df, totals])


def export_forecast_csv(fit_summary: dict, forecast_results: dict,
                        output_path: str = "outputs/volve_dca_forecast_data.csv"):
    """
    Export full per-well forecast data (history + forecast) to CSV.
    Suitable for downstream dashboard consumption or further analysis.

    Output columns
    --------------
    well, t_months, q_actual_sm3d, q_fitted_sm3d, q_forecast_sm3d, period

    Parameters
    ----------
    fit_summary      : dict — model fitting results
    forecast_results : dict — EUR and forecast outputs
    output_path      : str  — destination CSV path
    """
    rows = []
    for well, info in fit_summary.items():
        fore = forecast_results[well]

        # Historical period: actual + fitted rates
        for t_val, q_act, q_fit in zip(info["t"], info["q_actual"], info["q_fitted"]):
            rows.append({
                "well":             well,
                "t_months":         t_val,
                "q_actual_sm3d":    q_act,
                "q_fitted_sm3d":    q_fit,
                "q_forecast_sm3d":  np.nan,
                "period":           "history",
            })

        # Forecast period
        for t_val, q_val in zip(fore["t_future"], fore["q_future"]):
            rows.append({
                "well":             well,
                "t_months":         t_val,
                "q_actual_sm3d":    np.nan,
                "q_fitted_sm3d":    np.nan,
                "q_forecast_sm3d":  q_val,
                "period":           "forecast",
            })

    export_df = pd.DataFrame(rows)
    export_df.to_csv(output_path, index=False)
    print(f"Exported : {len(export_df):,} rows -> {output_path}")
    print(f"Columns  : {export_df.columns.tolist()}")
    return export_df
