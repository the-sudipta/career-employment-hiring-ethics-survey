import pandas as pd
from .schema import FEATURE_GROUPS, CANONICAL_COLS
from .modeling import build_preprocessor, cv_score


def run_ablation(df_work: pd.DataFrame, X, y, model_type: str, seed: int, folds: int) -> pd.DataFrame:
    ordinal_cols = [
        CANONICAL_COLS["q9_uni_support"],
        CANONICAL_COLS["q12_readiness"],
        CANONICAL_COLS["q16_stress"],
        CANONICAL_COLS["q18_transparency"],
        CANONICAL_COLS["q19_nepotism"],
        CANONICAL_COLS["q20_fairness"],
    ]

    # group -> actual df column names
    group_cols = {}
    for g, keys in FEATURE_GROUPS.items():
        cols = [CANONICAL_COLS[k] for k in keys if CANONICAL_COLS[k] in X.columns]
        group_cols[g] = cols

    order = ["demographics", "career_choice", "preparation", "employment_problems", "ethics"]

    results = []
    current = []

    # Incremental add groups
    for g in order:
        current += group_cols.get(g, [])
        current = [c for c in current if c in X.columns]

        if len(current) == 0:
            results.append({"step": f"Add_{g}", "n_features_raw": 0, "note": "Skipped: 0 columns"})
            continue

        Xg = X[current].copy()
        pre = build_preprocessor(Xg, ordinal_cols)
        score = cv_score(Xg, y, pre, model_type=model_type, seed=seed, folds=folds)

        results.append({"step": f"Add_{g}", "n_features_raw": int(Xg.shape[1]), **score})

    # Each group alone
    for g in order:
        cols = [c for c in group_cols.get(g, []) if c in X.columns]
        if len(cols) == 0:
            results.append({"step": f"Only_{g}", "n_features_raw": 0, "note": "Skipped: 0 columns"})
            continue

        Xg = X[cols].copy()
        pre = build_preprocessor(Xg, ordinal_cols)
        score = cv_score(Xg, y, pre, model_type=model_type, seed=seed, folds=folds)

        results.append({"step": f"Only_{g}", "n_features_raw": int(Xg.shape[1]), **score})

    return pd.DataFrame(results)
