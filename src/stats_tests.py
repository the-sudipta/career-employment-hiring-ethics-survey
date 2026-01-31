import pandas as pd
from scipy.stats import chi2_contingency, spearmanr

def chi_square(df: pd.DataFrame, a: str, b: str) -> dict:
    ct = pd.crosstab(df[a], df[b])
    chi2, p, dof, _ = chi2_contingency(ct)
    return {"chi2": float(chi2), "p": float(p), "dof": int(dof), "table": ct}

def spearman(x: pd.Series, y: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    r, p = spearmanr(x, y, nan_policy="omit")
    return {"rho": float(r), "p": float(p)}
