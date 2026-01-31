import sys
import os
import argparse
import json
from pathlib import Path

import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    OUT_DIR, TABLE_DIR, FIG_DIR, MET_DIR, MODEL_DIR,
    CLEAN_CSV, DISCARDED_CSV, METRICS_JSON, ABLATION_CSV, MODEL_FILE,
    RANDOM_SEED
)
from src.data_io import read_raw_xlsx, strip_columns  # noqa: E402
from src.cleaning import build_working_df  # noqa: E402
from src.features import make_target  # noqa: E402
from src.schema import CANONICAL_COLS  # noqa: E402
from src.stats_tests import chi_square, spearman  # noqa: E402
from src.modeling import build_preprocessor, train_holdout, cv_score  # noqa: E402
from src.ablation import run_ablation  # noqa: E402
from src.plots import (
    save_dashboard,
    save_ordinal_corr_heatmap,
    save_ablation_plot,
    save_3d_scatter,
    save_3d_scatter_interactive,
    save_individual_figures,
    save_conf_matrix,
    save_crosstab_heatmap,
    save_scatter_matrix_ordinal,
)


from src.viewer import show_saved_figures  # noqa: E402


def save_freq_tables(df: pd.DataFrame):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for key, col in CANONICAL_COLS.items():
        if col not in df.columns:
            continue
        if key == "timestamp":
            continue
        vc = df[col].value_counts(dropna=False).reset_index()
        vc.columns = ["option", "count"]
        vc["percent"] = (vc["count"] / vc["count"].sum() * 100).round(2)
        vc.to_csv(TABLE_DIR / f"freq_{key}.csv", index=False)


def run_one_pipeline(df_work: pd.DataFrame, target: str, model_name: str, folds: int, do_3d: bool, do_ablation: bool):
    """
    Runs: CV + holdout model save + optional ablation + optional 3D.
    Returns dict of results.
    """
    X, y = make_target(df_work, target)

    ordinal_cols = [
        CANONICAL_COLS["q9_uni_support"],
        CANONICAL_COLS["q12_readiness"],
        CANONICAL_COLS["q16_stress"],
        CANONICAL_COLS["q18_transparency"],
        CANONICAL_COLS["q19_nepotism"],
        CANONICAL_COLS["q20_fairness"],
    ]

    pre = build_preprocessor(X, ordinal_cols)

    cv = cv_score(X, y, pre, model_type=model_name, seed=RANDOM_SEED, folds=folds)

    model_obj, holdout_metrics, y_test, y_pred = train_holdout(
        X, y, pre, model_type=model_name, seed=RANDOM_SEED
    )

    # Confusion matrix (paper-ready)
    if y_test is not None and y_pred is not None:
        if target == "employed_binary":
            save_conf_matrix(
                y_test, y_pred,
                filename=f"confusion_{target}_{model_name}.png",
                class_names=["Not employed", "Employed"]
            )
        else:
            save_conf_matrix(
                y_test, y_pred,
                filename=f"confusion_{target}_{model_name}.png"
            )

    # Save model per (target,model)
    model_path = MODEL_DIR / f"model_{target}_{model_name}.joblib"
    if model_obj is not None:
        joblib.dump(model_obj, model_path)

    # Ablation
    ablation_path = None
    if do_ablation:
        df_ab = run_ablation(df_work, X, y, model_type=model_name, seed=RANDOM_SEED, folds=folds)
        ablation_path = MET_DIR / f"ablation_{target}_{model_name}.csv"
        df_ab.to_csv(ablation_path, index=False)
        save_ablation_plot(df_ab)  # single plot file is fine

    # Optional 3D (only meaningful when target is employed_binary)
    if do_3d and target == "employed_binary":
        _, emp_bin = make_target(df_work, "employed_binary")
        save_3d_scatter(df_work, employed_binary=emp_bin)

    return {
        "target": target,
        "model": model_name,
        "cv": cv,
        "holdout": holdout_metrics,
        "model_file": str(model_path),
        "ablation_csv": str(ablation_path) if ablation_path else None
    }


def main():
    running_in_pycharm = os.environ.get("PYCHARM_HOSTED") == "1"

    parser = argparse.ArgumentParser()

    # Optional: if user wants to run only one config
    parser.add_argument("--target", choices=["employed_binary", "stress_level", "employment_status_multiclass"])
    parser.add_argument("--model", choices=["rf", "logreg"])
    parser.add_argument("--folds", type=int, default=5)

    # Optional switches (but default is ON)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-3d", action="store_true")
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-show", action="store_true")

    args = parser.parse_args()

    # ✅ DEFAULT behavior: everything ON
    do_plots = not args.no_plots
    do_3d = not args.no_3d
    do_ablation = not args.no_ablation

    # show figures default: ON in PyCharm, OFF otherwise (you can change)
    if args.no_show:
        do_show = False
    elif args.show:
        do_show = True
    else:
        do_show = running_in_pycharm

    # Output dirs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    MET_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Read raw (never modify)
    df_raw = read_raw_xlsx()
    df_stripped = strip_columns(df_raw)

    # Clean only on df_work
    df_work, df_discarded = build_working_df(df_stripped)
    df_work.to_csv(CLEAN_CSV, index=False)
    df_discarded.to_csv(DISCARDED_CSV, index=False)

    save_freq_tables(df_work)

    # Stats tests (core)
    stats_out = {}
    try:
        c_intern = CANONICAL_COLS["q10_internship"]
        c_emp = CANONICAL_COLS["q17_employment_status"]
        chi = chi_square(df_work, c_intern, c_emp)
        chi["table"].to_csv(TABLE_DIR / "chi_internship_vs_employment.csv")
        stats_out["chi_internship_vs_employment"] = {"chi2": chi["chi2"], "p": chi["p"], "dof": chi["dof"]}
    except Exception as e:
        stats_out["chi_internship_vs_employment_error"] = str(e)

    try:
        c_ready = CANONICAL_COLS["q12_readiness"]
        c_stress = CANONICAL_COLS["q16_stress"]
        stats_out["spearman_readiness_vs_stress"] = spearman(df_work[c_ready], df_work[c_stress])
    except Exception as e:
        stats_out["spearman_readiness_vs_stress_error"] = str(e)

    # ✅ PLOTS always (unless disabled)
    made_figs = []
    if do_plots:
        save_dashboard(df_work)
        save_ordinal_corr_heatmap(df_work)
        # Paper-ready single figures
        save_individual_figures(df_work)

        # Complex insight figures
        save_crosstab_heatmap(df_work)
        save_scatter_matrix_ordinal(df_work)

        made_figs += [
            FIG_DIR / "dashboard_summary.png",
            FIG_DIR / "ordinal_spearman_heatmap.png"
        ]

    # ✅ AUTO MODE (no args) -> run ALL targets + ALL models
    auto_mode = (args.target is None and args.model is None)

    results = []
    targets = ["employed_binary", "stress_level", "employment_status_multiclass"]
    models = ["rf", "logreg"]

    if auto_mode:
        for t in targets:
            for m in models:
                results.append(run_one_pipeline(df_work, t, m, args.folds, do_3d, do_ablation))
    else:
        # run only requested
        t = args.target or "employed_binary"
        m = args.model or "rf"
        results.append(run_one_pipeline(df_work, t, m, args.folds, do_3d, do_ablation))

    # add 3D figure path if created
    if do_3d:
        p3d = FIG_DIR / "3d_scatter_readiness_fairness_stress.png"
        _, emp_bin = make_target(df_work, "employed_binary")
        save_3d_scatter(df_work, employed_binary=emp_bin)
        save_3d_scatter_interactive(df_work, employed_binary=emp_bin)
        if p3d.exists():
            made_figs.append(p3d)

    # ablation plot path
    ab_plot = FIG_DIR / "ablation_f1_plot.png"
    if ab_plot.exists():
        made_figs.append(ab_plot)

    out = {
        "rows_raw": int(len(df_raw)),
        "rows_after_cleaning": int(len(df_work)),
        "discarded_rows": int(len(df_discarded)),
        "cleaned_csv": str(CLEAN_CSV),
        "discarded_csv": str(DISCARDED_CSV),
        "stats": stats_out,
        "runs": results,
        "figures": [str(p) for p in made_figs if p.exists()],
    }

    METRICS_JSON.write_text(json.dumps(out, indent=2))

    print("Done.")
    print("Cleaned:", CLEAN_CSV)
    print("Discarded:", DISCARDED_CSV)
    print("Tables:", TABLE_DIR)
    print("Figures:", FIG_DIR)
    print("Metrics:", METRICS_JSON)
    print("Models:", MODEL_DIR)

    if do_show and len(made_figs) > 0:
        try:
            show_saved_figures(made_figs, window_title_prefix="RESEARCH • ")
        except Exception as e:
            print("Could not show figures (but they are saved). Reason:", str(e))


if __name__ == "__main__":
    main()
