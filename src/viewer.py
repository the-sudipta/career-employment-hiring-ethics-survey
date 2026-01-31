from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def show_saved_figures(fig_paths: list[Path], window_title_prefix: str = "") -> None:
    """
    Displays already-saved PNG files using matplotlib.

    In PyCharm:
    - SciView enabled থাকলে plots PyCharm-এর tool window-তে দেখাবে
    - না থাকলে normal matplotlib window open হবে
    """
    shown = 0
    for p in fig_paths:
        if not p.exists():
            continue
        try:
            img = mpimg.imread(p)
        except Exception:
            continue

        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{window_title_prefix}{p.name}")
        shown += 1

    if shown > 0:
        plt.show()
