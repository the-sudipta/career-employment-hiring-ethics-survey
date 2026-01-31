import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mpl_toolkits.mplot3d import Axes3D  # keep
import numpy as np

from .config import FIG_DIR
from .schema import CANONICAL_COLS


def save_dashboard(df: pd.DataFrame):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    c_career = CANONICAL_COLS["q6_target_career"]
    c_problem = CANONICAL_COLS["q15_biggest_problem"]
    c_stress = CANONICAL_COLS["q16_stress"]
    c_intern = CANONICAL_COLS["q10_internship"]
    c_emp = CANONICAL_COLS["q17_employment_status"]
    c_tr = CANONICAL_COLS["q18_transparency"]
    c_nep = CANONICAL_COLS["q19_nepotism"]
    c_fair = CANONICAL_COLS["q20_fairness"]
    c_ready = CANONICAL_COLS["q12_readiness"]

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 3)

    # 1) Target career (Top 10)
    ax1 = fig.add_subplot(gs[0, 0])
    top = df[c_career].value_counts().head(10)
    ax1.barh(list(reversed(top.index.tolist())), list(reversed(top.values.tolist())))
    ax1.set_title("Top Target Careers (Top 10)")
    ax1.set_xlabel("Count")

    # 2) Biggest problem (Top 10)
    ax2 = fig.add_subplot(gs[0, 1])
    top2 = df[c_problem].value_counts().head(10)
    ax2.barh(list(reversed(top2.index.tolist())), list(reversed(top2.values.tolist())))
    ax2.set_title("Biggest Job Problems (Top 10)")
    ax2.set_xlabel("Count")

    # 3) Stress distribution
    ax3 = fig.add_subplot(gs[0, 2])
    stress_counts = df[c_stress].value_counts().sort_index()
    ax3.bar(stress_counts.index.astype(str), stress_counts.values)
    ax3.set_title("Stress Level Distribution")
    ax3.set_xlabel("Stress (1 low → 5 high)")
    ax3.set_ylabel("Count")

    # 4) Ethics boxplot
    ax4 = fig.add_subplot(gs[1, 0])
    ethics = df[[c_tr, c_nep, c_fair]].copy()
    ethics.columns = ["Transparency", "Nepotism/Favoritism", "Fairness"]
    ax4.boxplot([ethics[c].dropna().values for c in ethics.columns], labels=ethics.columns, showmeans=True)
    ax4.set_title("Hiring Ethics Perception (Likert 1–5)")
    ax4.set_ylabel("Score")
    ax4.tick_params(axis="x", rotation=15)

    # 5) Internship vs Employment (row % stacked)
    ax5 = fig.add_subplot(gs[1, 1])
    ct = pd.crosstab(df[c_intern], df[c_emp], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, ax=ax5)
    ax5.set_title("Internship vs Employment Status (Row %)")
    ax5.set_ylabel("Percent")
    ax5.legend(fontsize=9, title="Employment", loc="upper right")

    # 6) Readiness vs Stress (box)
    ax6 = fig.add_subplot(gs[1, 2])
    groups, labels = [], []
    for r in sorted(df[c_ready].dropna().unique()):
        g = df.loc[df[c_ready] == r, c_stress].dropna().values
        if len(g) > 0:
            groups.append(g)
            labels.append(str(int(r)) if float(r).is_integer() else str(r))
    if groups:
        ax6.boxplot(groups, labels=labels, showmeans=True)
    ax6.set_title("Readiness vs Stress")
    ax6.set_xlabel("Readiness (1–5)")
    ax6.set_ylabel("Stress (1–5)")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "dashboard_summary.png", dpi=250)
    plt.close(fig)


def save_ordinal_corr_heatmap(df: pd.DataFrame):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cols = [
        CANONICAL_COLS["q9_uni_support"],
        CANONICAL_COLS["q12_readiness"],
        CANONICAL_COLS["q16_stress"],
        CANONICAL_COLS["q18_transparency"],
        CANONICAL_COLS["q19_nepotism"],
        CANONICAL_COLS["q20_fairness"],
    ]
    sub = df[cols].copy()
    corr = sub.corr(method="spearman").values

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, aspect="auto")
    ax.set_title("Spearman Correlation (Ordinal Variables)")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(["UniSupport", "Readiness", "Stress", "Transparency", "Nepotism", "Fairness"], rotation=30, ha="right")
    ax.set_yticklabels(["UniSupport", "Readiness", "Stress", "Transparency", "Nepotism", "Fairness"])

    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ordinal_spearman_heatmap.png", dpi=250)
    plt.close(fig)


# def save_3d_scatter(df: pd.DataFrame, employed_binary=None):
#     FIG_DIR.mkdir(parents=True, exist_ok=True)
#
#     x_col = CANONICAL_COLS["q12_readiness"]
#     y_col = CANONICAL_COLS["q20_fairness"]
#     z_col = CANONICAL_COLS["q16_stress"]
#
#     x = pd.to_numeric(df[x_col], errors="coerce")
#     y = pd.to_numeric(df[y_col], errors="coerce")
#     z = pd.to_numeric(df[z_col], errors="coerce")
#
#     mask = x.notna() & y.notna() & z.notna()
#     x, y, z = x[mask], y[mask], z[mask]
#
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection="3d")
#
#     if employed_binary is not None:
#         c = employed_binary.loc[mask].values
#         ax.scatter(x, y, z, c=c)
#     else:
#         ax.scatter(x, y, z)
#
#     ax.set_xlabel("Readiness (1–5)")
#     ax.set_ylabel("Fairness (1–5)")
#     ax.set_zlabel("Stress (1–5)")
#     ax.set_title("3D: Readiness vs Fairness vs Stress")
#     plt.tight_layout()
#     plt.savefig(FIG_DIR / "3d_scatter_readiness_fairness_stress.png", dpi=250)
#     plt.close(fig)


def save_ablation_plot(df_ablation: pd.DataFrame):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    inc = df_ablation[df_ablation["step"].astype(str).str.startswith("Add_")].copy()
    if inc.empty:
        return

    inc = inc.sort_values("n_features_raw")
    x = inc["n_features_raw"].values
    y = inc["f1_macro_mean"].values
    ystd = inc["f1_macro_std"].values

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y, marker="o")
    ax.fill_between(x, y - ystd, y + ystd, alpha=0.2)
    ax.set_title("Ablation Study (CV): F1 Macro vs Feature Groups Added")
    ax.set_xlabel("Number of raw features (before one-hot)")
    ax.set_ylabel("F1 macro (mean ± std)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ablation_f1_plot.png", dpi=250)
    plt.close(fig)



def save_3d_scatter_interactive(df: pd.DataFrame, employed_binary: pd.Series | None = None):
    import plotly.express as px

    x_col = CANONICAL_COLS["q12_readiness"]
    y_col = CANONICAL_COLS["q20_fairness"]
    z_col = CANONICAL_COLS["q16_stress"]

    sub = df[[x_col, y_col, z_col]].copy()
    sub.columns = ["Readiness", "Fairness", "Stress"]

    if employed_binary is not None:
        sub["Employed"] = employed_binary.values
        fig = px.scatter_3d(
            sub, x="Readiness", y="Fairness", z="Stress",
            color="Employed",
            hover_data=["Readiness","Fairness","Stress"]
        )
    else:
        fig = px.scatter_3d(sub, x="Readiness", y="Fairness", z="Stress",
                            hover_data=["Readiness","Fairness","Stress"])

    fig.update_layout(title="Interactive 3D: Readiness vs Fairness vs Stress")
    fig.write_html(FIG_DIR / "3d_scatter_readiness_fairness_stress.html")



def save_individual_figures(df: pd.DataFrame):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    c_career = CANONICAL_COLS["q6_target_career"]
    c_problem = CANONICAL_COLS["q15_biggest_problem"]
    c_stress = CANONICAL_COLS["q16_stress"]

    c_intern = CANONICAL_COLS["q10_internship"]
    c_emp = CANONICAL_COLS["q17_employment_status"]

    c_tr = CANONICAL_COLS["q18_transparency"]
    c_nep = CANONICAL_COLS["q19_nepotism"]
    c_fair = CANONICAL_COLS["q20_fairness"]
    c_ready = CANONICAL_COLS["q12_readiness"]

    # 1) Target career (top 10)
    plt.figure(figsize=(10, 6))
    top = df[c_career].value_counts().head(10)
    plt.barh(top.index.astype(str), top.values)
    plt.title("Top Target Careers (Top 10)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "bar_target_career.png", dpi=300)
    plt.close()

    # 2) Biggest problem (top 10)
    plt.figure(figsize=(10, 6))
    top2 = df[c_problem].value_counts().head(10)
    plt.barh(top2.index.astype(str), top2.values)
    plt.title("Biggest Job Problems (Top 10)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "bar_biggest_problem.png", dpi=300)
    plt.close()

    # 3) Stress distribution
    plt.figure(figsize=(7, 5))
    s = df[c_stress].dropna()
    vc = s.value_counts().sort_index()
    plt.bar(vc.index.astype(int).astype(str), vc.values)
    plt.title("Stress Level Distribution (1-5)")
    plt.xlabel("Stress level")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "stress_distribution.png", dpi=300)
    plt.close()

    # 4) Ethics boxplot
    plt.figure(figsize=(9, 5))
    ethics = df[[c_tr, c_nep, c_fair]].copy()
    ethics.columns = ["Transparency", "Nepotism/Favoritism", "Fairness"]
    plt.boxplot([ethics[c].dropna().values for c in ethics.columns], labels=ethics.columns)
    plt.title("Hiring Ethics Perception (Likert 1-5)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ethics_boxplot.png", dpi=300)
    plt.close()

    # 5) Internship vs Employment (row % stacked)
    ct = pd.crosstab(df[c_intern], df[c_emp], normalize="index") * 100
    plt.figure(figsize=(11, 6))
    bottom = np.zeros(len(ct))
    for col in ct.columns:
        plt.bar(ct.index.astype(str), ct[col].values, bottom=bottom, label=str(col))
        bottom += ct[col].values
    plt.title("Internship vs Employment Status (Row %)")
    plt.ylabel("Percent")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "internship_vs_employment.png", dpi=300)
    plt.close()

    # 6) Readiness vs Stress (box)
    plt.figure(figsize=(8, 5))
    x = df[c_ready].dropna()
    y = df[c_stress].dropna()
    tmp = df[[c_ready, c_stress]].dropna()
    groups = [tmp[tmp[c_ready] == v][c_stress].values for v in sorted(tmp[c_ready].unique())]
    plt.boxplot(groups, labels=[str(v) for v in sorted(tmp[c_ready].unique())])
    plt.title("Stress by Readiness Level")
    plt.xlabel("Readiness (1-5)")
    plt.ylabel("Stress (1-5)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "readiness_vs_stress.png", dpi=300)
    plt.close()




def save_3d_scatter(df: pd.DataFrame, employed_binary=None, rays=None, annotate=None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    x_col = CANONICAL_COLS["q12_readiness"]
    y_col = CANONICAL_COLS["q20_fairness"]
    z_col = CANONICAL_COLS["q16_stress"]

    sub = df[[x_col, y_col, z_col]].dropna().copy()
    sub.columns = ["Readiness", "Fairness", "Stress"]

    n = len(sub)
    # AUTO behavior
    if rays is None:
        rays = (n <= 30)
    if annotate is None:
        annotate = (n <= 20)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    if employed_binary is not None:
        eb = pd.Series(employed_binary).loc[sub.index]
        colors = eb.astype(int).values
        sc = ax.scatter(sub["Readiness"], sub["Fairness"], sub["Stress"], c=colors)
    else:
        ax.scatter(sub["Readiness"], sub["Fairness"], sub["Stress"])

    ax.set_xlabel("Readiness (1-5)")
    ax.set_ylabel("Fairness (1-5)")
    ax.set_zlabel("Stress (1-5)")
    ax.set_title("3D: Readiness vs Fairness vs Stress")

    # baseline in likert space
    bx, by, bz = 1, 1, 1

    if rays:
        for (x, y, z) in sub[["Readiness", "Fairness", "Stress"]].values:
            ax.plot([bx, x], [by, y], [bz, z], linewidth=0.6)

    if annotate:
        for i, (x, y, z) in enumerate(sub[["Readiness", "Fairness", "Stress"]].values):
            ax.text(x, y, z, f"({int(x)},{int(y)},{int(z)})", fontsize=7)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "3d_scatter_readiness_fairness_stress.png", dpi=300)
    plt.close(fig)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def save_conf_matrix(y_true, y_pred, filename="confusion_matrix.png", class_names=None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    labels = sorted(list(set(y_true.tolist()) | set(y_pred.tolist())))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if class_names is None:
        display_labels = [str(x) for x in labels]
    else:
        display_labels = class_names

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=300)
    plt.close(fig)

def save_crosstab_heatmap(df: pd.DataFrame):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    c_problem = CANONICAL_COLS["q15_biggest_problem"]
    c_emp = CANONICAL_COLS["q17_employment_status"]

    ct = pd.crosstab(df[c_emp], df[c_problem], normalize="index") * 100
    data = ct.values

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(data)

    ax.set_xticks(range(ct.shape[1]))
    ax.set_yticks(range(ct.shape[0]))
    ax.set_xticklabels(ct.columns.astype(str), rotation=45, ha="right")
    ax.set_yticklabels(ct.index.astype(str))
    ax.set_title("Heatmap: Biggest Problem by Employment Status (Row %)")

    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax.text(j, i, f"{data[i, j]:.1f}%", ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "heatmap_problem_vs_employment.png", dpi=300)
    plt.close(fig)


def save_scatter_matrix_ordinal(df: pd.DataFrame):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cols = [
        CANONICAL_COLS["q12_readiness"],
        CANONICAL_COLS["q16_stress"],
        CANONICAL_COLS["q18_transparency"],
        CANONICAL_COLS["q19_nepotism"],
        CANONICAL_COLS["q20_fairness"],
    ]
    sub = df[cols].dropna().copy()
    if len(sub) < 4:
        return

    sub.columns = ["Readiness", "Stress", "Transparency", "Nepotism", "Fairness"]
    names = list(sub.columns)

    k = len(names)
    fig, axes = plt.subplots(k, k, figsize=(12, 12))

    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if i == j:
                ax.hist(sub[names[i]].values, bins=5)
            else:
                ax.scatter(sub[names[j]].values, sub[names[i]].values, s=12)

            if i == k - 1:
                ax.set_xlabel(names[j], fontsize=8)
            else:
                ax.set_xticks([])
            if j == 0:
                ax.set_ylabel(names[i], fontsize=8)
            else:
                ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(FIG_DIR / "scatter_matrix_ordinal.png", dpi=300)
    plt.close(fig)
