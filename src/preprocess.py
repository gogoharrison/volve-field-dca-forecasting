# src/preprocess.py
# ── Data Ingestion, Cleaning & Feature Engineering ──────────────────────────
# Handles all data preparation steps upstream of DCA modelling:
# producer filtering, shut-in removal, monthly normalisation,
# and derived reservoir diagnostic features (GOR, WOR, cumulative volume).

import pandas as pd
import numpy as np


# ── Column name constants (single source of truth) ──────────────────────────
DATE_COL     = "DATEPRD"
WELL_COL     = "WELL_BORE_CODE"
OIL_COL      = "BORE_OIL_VOL"
GAS_COL      = "BORE_GAS_VOL"
WAT_COL      = "BORE_WAT_VOL"
WI_COL       = "BORE_WI_VOL"
PRESSURE_COL = "AVG_DOWNHOLE_PRESSURE"
WHP_COL      = "AVG_WHP_P"
TEMP_COL     = "AVG_DOWNHOLE_TEMPERATURE"
ONSTREAM_COL = "ON_STREAM_HRS"
FLOW_COL     = "FLOW_KIND"
TYPE_COL     = "WELL_TYPE"

# ── Processing constants ─────────────────────────────────────────────────────
DAYS_PER_MONTH   = 30.4375      # average days/month (365.25 / 12)
WELL_TYPE_PROD   = "OP"         # oil producer code in Volve dataset
MIN_OIL_RATE     = 0.0          # Sm³/day — rows at or below this are shut-in
SMC3_TO_BBL      = 6.2898       # Norwegian standard conversion factor


def load_raw(filepath: str) -> pd.DataFrame:
    """
    Load the Volve production Excel file into a raw DataFrame.

    Parameters
    ----------
    filepath : str — path to 'Volve production data.xlsx'

    Returns
    -------
    pd.DataFrame — raw data with DATE_COL parsed as datetime
    """
    df = pd.read_excel(filepath, parse_dates=[DATE_COL])
    print(f"Loaded   : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Date range: {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")
    print(f"Wells    : {sorted(df[WELL_COL].unique())}")
    return df


def filter_producers(df: pd.DataFrame, well_type: str = WELL_TYPE_PROD) -> pd.DataFrame:
    """
    Retain only oil-producing wells. Injectors and observation wells
    have no production decline and are excluded from DCA.

    Parameters
    ----------
    df        : pd.DataFrame — raw dataset
    well_type : str          — WELL_TYPE value for producers (default: 'OP')

    Returns
    -------
    pd.DataFrame — producer rows only
    """
    df_prod = df[df[TYPE_COL] == well_type].copy()
    print(f"Producers : {df_prod.shape[0]:,} rows "
          f"({df_prod[WELL_COL].nunique()} wells) after filtering on WELL_TYPE='{well_type}'")
    return df_prod


def remove_shutin(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove shut-in periods — records where ON_STREAM_HRS = 0 or
    oil rate = 0. These rows carry no decline curve information and
    bias the curve_fit towards a flat (zero) baseline.

    Note: shut-in rows are removed from model input only. They are
    NOT removed from the calendar timeline — T_MONTHS is computed
    from first production date across all records including gaps.

    Parameters
    ----------
    df : pd.DataFrame — producer-filtered dataset

    Returns
    -------
    pd.DataFrame — active production rows only
    """
    mask = (df[ONSTREAM_COL] > 0) & (df[OIL_COL] > MIN_OIL_RATE)
    df_active = df[mask].copy()
    dropped = len(df) - len(df_active)
    print(f"Shut-in removed : {dropped:,} rows -> {df_active.shape[0]:,} active rows remain")
    return df_active


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all derived features required for DCA modelling and the
    ML benchmark layer.

    Features added
    --------------
    OIL_MONTHLY : actual produced volume per record (Sm³)
                  = daily rate × (ON_STREAM_HRS / 24)
                  Handles partial production days and variable on-stream hours.

    T_MONTHS    : time since first production for each well (months)
                  Used as the x-axis in all Arps model fits.

    CUM_OIL     : cumulative oil produced per well (Sm³)
                  Proxy for reservoir depletion state.

    GOR         : gas-oil ratio (Sm³/Sm³)
                  Rising GOR indicates gas cap expansion or solution gas
                  depletion — a key reservoir energy diagnostic.

    WOR         : water-oil ratio (Sm³/Sm³)
                  Rising WOR signals water breakthrough or coning —
                  affects late-life recovery and economic limit timing.

    Parameters
    ----------
    df : pd.DataFrame — cleaned producer dataset

    Returns
    -------
    pd.DataFrame — with all derived features appended
    """
    df = df.sort_values([WELL_COL, DATE_COL]).reset_index(drop=True)

    # Actual produced volume per record (corrects for partial on-stream days)
    df["OIL_MONTHLY"] = df[OIL_COL] * (df[ONSTREAM_COL] / 24.0)

    # Time-on-production per well (months since first production date)
    df["FIRST_PROD"] = df.groupby(WELL_COL)[DATE_COL].transform("min")
    df["T_MONTHS"]   = ((df[DATE_COL] - df["FIRST_PROD"])
                        .dt.days / DAYS_PER_MONTH).round(2)

    # Cumulative produced oil per well
    df["CUM_OIL"] = df.groupby(WELL_COL)["OIL_MONTHLY"].cumsum()

    # Reservoir diagnostic ratios (NaN-safe: avoid division by zero)
    df["GOR"] = np.where(df[OIL_COL] > 0, df[GAS_COL] / df[OIL_COL], np.nan)
    df["WOR"] = np.where(df[OIL_COL] > 0, df[WAT_COL] / df[OIL_COL], np.nan)

    print(f"Features added : OIL_MONTHLY, T_MONTHS, CUM_OIL, GOR, WOR")
    return df


def run_preprocessing(filepath: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline: load → filter → clean → engineer.
    Single entry point for the notebook and any downstream scripts.

    Parameters
    ----------
    filepath : str — path to raw Excel file

    Returns
    -------
    pd.DataFrame — analysis-ready dataset
    """
    df = load_raw(filepath)
    df = filter_producers(df)
    df = remove_shutin(df)
    df = engineer_features(df)
    print(f"\nPreprocessing complete: {df.shape[0]:,} rows, "
          f"{df[WELL_COL].nunique()} wells ready for modelling.")
    return df
