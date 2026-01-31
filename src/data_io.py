import pandas as pd
from .config import RAW_XLSX, SHEET_CSV_URL_1, SHEET_CSV_URL_2


def read_raw_xlsx() -> pd.DataFrame:
    """
    Priority:
    1) Live Google Sheet (CSV)
    2) Fallback: local dataset.xlsx
    """
    try:
        try:
            df_raw = pd.read_csv(SHEET_CSV_URL_1)
        except Exception:
            df_raw = pd.read_csv(SHEET_CSV_URL_2)
        return df_raw

    except Exception as e:
        print(f"[WARN] Live sheet read failed. Falling back to local dataset.xlsx. Error: {e}")
        return pd.read_excel(RAW_XLSX)


def strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy(deep=True)
    df2.columns = [str(c).strip() for c in df2.columns]
    return df2
