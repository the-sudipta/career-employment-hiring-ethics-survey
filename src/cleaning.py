import pandas as pd
from .config import CONSENT_YES, MAX_MISSING_RATIO
from .schema import CANONICAL_COLS, LIKERT_5, STRESS_5, UNI_SUPPORT_5, READINESS_5


def assert_columns_exist(df: pd.DataFrame):
    missing = [col for col in CANONICAL_COLS.values() if col not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns (after stripping): {missing}")


def _map_ordinal(series: pd.Series, mapping: dict) -> pd.Series:
    """
    Robust mapping:
    - case-insensitive
    - supports numeric strings too (e.g. '1','2'...)
    - unmatched text -> NaN (not crash)
    """
    s = series.astype(str).str.strip()
    s_low = s.str.lower()

    mapping_low = {str(k).strip().lower(): v for k, v in mapping.items()}
    mapped = s_low.map(mapping_low)

    # If answers already numeric or numeric-strings
    num = pd.to_numeric(s, errors="coerce")

    out = mapped.where(mapped.notna(), num)
    return pd.to_numeric(out, errors="coerce")


def build_working_df(df_stripped: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (df_work, df_discarded)
    Raw dataset is NOT modified.
    """
    assert_columns_exist(df_stripped)
    df = df_stripped.copy()

    reason = pd.Series("", index=df.index, dtype=object)

    consent_col = CANONICAL_COLS["consent"]
    consent_ok = df[consent_col].astype(str).str.strip().isin(CONSENT_YES)
    reason.loc[~consent_ok] += "no_consent;"

    missing_ratio = df.isna().mean(axis=1)
    miss_ok = missing_ratio <= MAX_MISSING_RATIO
    reason.loc[~miss_ok] += "too_missing;"

    dup_mask = df.duplicated(keep="first")
    reason.loc[dup_mask] += "duplicate;"

    keep_mask = reason == ""
    df_work = df.loc[keep_mask].copy()
    df_disc = df.loc[~keep_mask].copy()
    df_disc["discard_reason"] = reason.loc[~keep_mask].values

    # --- Ordinal mapping (robust) ---
    df_work[CANONICAL_COLS["q18_transparency"]] = _map_ordinal(df_work[CANONICAL_COLS["q18_transparency"]], LIKERT_5)
    df_work[CANONICAL_COLS["q19_nepotism"]] = _map_ordinal(df_work[CANONICAL_COLS["q19_nepotism"]], LIKERT_5)
    df_work[CANONICAL_COLS["q20_fairness"]] = _map_ordinal(df_work[CANONICAL_COLS["q20_fairness"]], LIKERT_5)

    df_work[CANONICAL_COLS["q16_stress"]] = _map_ordinal(df_work[CANONICAL_COLS["q16_stress"]], STRESS_5)
    df_work[CANONICAL_COLS["q12_readiness"]] = _map_ordinal(df_work[CANONICAL_COLS["q12_readiness"]], READINESS_5)

    # Uni support: "Not sure" -> None -> NaN automatically
    df_work[CANONICAL_COLS["q9_uni_support"]] = _map_ordinal(df_work[CANONICAL_COLS["q9_uni_support"]], UNI_SUPPORT_5)

    return df_work, df_disc
