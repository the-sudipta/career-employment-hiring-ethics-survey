import pandas as pd
from .schema import CANONICAL_COLS
from .config import EMPLOYED_SET

def make_target(df_work: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Creates X, y from df_work.
    """
    ts_col = CANONICAL_COLS["timestamp"]
    df = df_work.copy()
    if ts_col in df.columns:
        df = df.drop(columns=[ts_col])

    if target == "employed_binary":
        status_col = CANONICAL_COLS["q17_employment_status"]
        y = df[status_col].astype(str).str.strip().map(lambda s: 1 if s in EMPLOYED_SET else 0).astype(int)
        X = df.drop(columns=[status_col])
        return X, y

    if target == "stress_level":
        y_col = CANONICAL_COLS["q16_stress"]
        y = pd.to_numeric(df[y_col], errors="coerce")
        X = df.drop(columns=[y_col])
        return X, y

    if target == "employment_status_multiclass":
        y_col = CANONICAL_COLS["q17_employment_status"]
        y = df[y_col].astype(str).str.strip()
        X = df.drop(columns=[y_col])
        return X, y

    raise ValueError("Unknown target")
