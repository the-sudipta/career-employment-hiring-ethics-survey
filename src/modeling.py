import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class FillNaTransformer(BaseEstimator, TransformerMixin):
    """
    Fill missing values WITHOUT dropping any column.
    This avoids "Skipping features without any observed values" problems.
    """
    def __init__(self, fill_value, as_float=False, as_str=False):
        self.fill_value = fill_value
        self.as_float = as_float
        self.as_str = as_str

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X may be DataFrame / ndarray
        if hasattr(X, "copy"):
            X2 = X.copy()
            try:
                X2 = X2.fillna(self.fill_value)
            except Exception:
                pass

            if self.as_float:
                return np.asarray(X2, dtype=float)
            if self.as_str:
                return np.asarray(X2, dtype=str)
            return np.asarray(X2)

        X_arr = np.array(X, dtype=object)
        mask = pd.isna(X_arr)
        X_arr[mask] = self.fill_value

        if self.as_float:
            return X_arr.astype(float)
        if self.as_str:
            return X_arr.astype(str)
        return X_arr


def build_preprocessor(X: pd.DataFrame, ordinal_cols: list[str]) -> ColumnTransformer:
    """
    FIX:
    If a transformer gets 0 columns (common during ablation like "Only_ethics"),
    scikit-learn may not fit that transformer and later raises NotFittedError.

    So we ONLY add ("ord"/"nom") when that column list is non-empty.
    """
    ordinal_cols = [c for c in ordinal_cols if c in X.columns]
    nominal_cols = [c for c in X.columns if c not in ordinal_cols]

    transformers = []

    if len(ordinal_cols) > 0:
        ord_pipe = Pipeline(steps=[
            ("fill", FillNaTransformer(0.0, as_float=True)),
        ])
        transformers.append(("ord", ord_pipe, ordinal_cols))

    if len(nominal_cols) > 0:
        nom_pipe = Pipeline(steps=[
            ("fill", FillNaTransformer("Missing", as_str=True)),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("nom", nom_pipe, nominal_cols))

    if len(transformers) == 0:
        raise ValueError("No usable feature columns found. Check your X columns and cleaning step.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def build_model(model_type: str, seed: int):
    if model_type == "rf":
        return RandomForestClassifier(n_estimators=400, random_state=seed)
    return LogisticRegression(max_iter=3000)


def train_holdout(X: pd.DataFrame, y: pd.Series, pre: ColumnTransformer, model_type: str, seed: int):
    """
    Trains a final model for saving.
    If data is too small or preprocessing produces 0 features, it will NOT crash.
    """
    mask = ~pd.isna(y)
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    if y.nunique() < 2:
        return None, {
            "rows_used": int(len(X)),
            "note": "Skipped training: target has only 1 class (need more data)."
        }, None, None

    n = len(y)
    n_classes = int(y.nunique())
    test_frac = 0.2
    n_test = max(1, int(round(test_frac * n)))
    n_train = n - n_test

    # Can we even do a holdout split that covers all classes?
    # For stratified split, both train and test must be able to contain all classes.
    can_holdout = (n_test >= n_classes) and (n_train >= n_classes)

    # Stratify only if every class has at least 2 samples AND holdout is possible
    min_count = int(y.value_counts().min())
    can_stratify = can_holdout and (min_count >= 2)

    if not can_holdout:
        # Skip holdout safely (still return 4 values)
        return None, {
            "rows_used": int(n),
            "note": f"Skipped holdout split: too few rows for {n_classes} classes with test_size={test_frac}. "
                    f"Need at least {int((n_classes / test_frac) + 0.5)} rows for holdout. Using CV only."
        }, None, None

    stratify = y if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=seed, stratify=stratify
    )

    # small-data fallback
    if pd.Series(y_train).nunique() < 2:
        X_train, y_train = X, y
        X_test, y_test = X, y

    model = build_model(model_type, seed)
    pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])

    try:
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
    except ValueError as e:
        return None, {
            "rows_used": int(len(X)),
            "note": f"Training failed due to preprocessing/feature issue: {str(e)}"
        }, None, None
    except Exception as e:
        return None, {
            "rows_used": int(len(X)),
            "note": f"Training failed: {str(e)}"
        }, None, None

    metrics = {
        "rows_used": int(len(X)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    }
    return pipe, metrics, y_test, pred


def cv_score(X: pd.DataFrame, y: pd.Series, pre: ColumnTransformer, model_type: str, seed: int, folds: int = 5):
    """
    Cross-validation scoring (for paper).
    If a fold becomes invalid or has preprocessing issues, we skip that fold instead of crashing.
    """
    mask = ~pd.isna(y)
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    if y.nunique() < 2 or len(X) < 4:
        return {
            "folds": 0,
            "rows_used": int(len(X)),
            "note": "CV skipped: too little data or only 1 class."
        }

    # small data -> low folds
    folds = min(folds, 2) if len(X) < 20 else min(folds, 5)

    min_class = y.value_counts().min()
    if min_class >= 2:
        folds = min(folds, int(min_class))
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(X)

    accs, f1s = [], []
    skipped = 0

    for tr_idx, te_idx in split_iter:
        X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
        y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]

        if pd.Series(y_train).nunique() < 2:
            skipped += 1
            continue

        model = build_model(model_type, seed)
        pipe = Pipeline(steps=[("preprocess", pre), ("model", model)])

        try:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
        except Exception:
            skipped += 1
            continue

        accs.append(accuracy_score(y_test, pred))
        f1s.append(f1_score(y_test, pred, average="macro"))

    if len(accs) == 0:
        return {
            "folds": int(folds),
            "rows_used": int(len(X)),
            "note": f"CV produced 0 valid folds (skipped={skipped}). Dataset too small / imbalance."
        }

    return {
        "folds": int(folds),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "f1_macro_mean": float(np.mean(f1s)),
        "f1_macro_std": float(np.std(f1s)),
        "rows_used": int(len(X)),
        "skipped_folds": int(skipped),
    }
