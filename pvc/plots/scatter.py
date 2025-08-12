# pvc/plots/scatter.py
import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import r2_score
except Exception:
    def r2_score(y_true, y_pred):
        y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        return float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot

def scatter(targets,
    preds,
    *,
    title=None,
    units="s",
    max_points=8000,
    clip_quantile=0.999,   # trims extreme outliers for nicer axes
    dot_size=18,
    alpha=0.55,
    dpi=200):
    """
    targets, preds: 1D arrays
    returns: matplotlib Figure
    """
    t = np.asarray(targets).reshape(-1)
    p = np.asarray(preds).reshape(-1)
    m = np.isfinite(t) & np.isfinite(p)
    t, p = t[m], p[m]

    # Subsample for readability
    if max_points and t.size > max_points:
        idx = np.random.default_rng(0).choice(t.size, max_points, replace=False)
        t, p = t[idx], p[idx]

    # Optional clipping to hide extreme outliers (keeps legend lines meaningful)
    if clip_quantile:
        lo = 0.0 if t.min() >= 0 and p.min() >= 0 else float(np.min([t.min(), p.min()]))
        hi = float(np.quantile(np.r_[t, p], clip_quantile))
    else:
        lo = float(np.min([t.min(), p.min()]))
        hi = float(np.max([t.max(), p.max()]))

    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lo, hi = lo - pad, hi + pad

    # Best-fit line: p ≈ a * t + b
    a, b = np.polyfit(t, p, 1)
    p_fit = a * t + b
    r2 = r2_score(t, p_fit)
    x_line = np.linspace(lo, hi, 200)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=dpi)
    ax.scatter(t, p, s=dot_size, alpha=alpha, color="#1f77b4", edgecolors="none", zorder=2)

    # Ideal (identity) line
    ax.plot([lo, hi], [lo, hi], ls="--", lw=2.0, color="0.6", label="Perfect Prediction", zorder=1)

    # Best-fit line
    ax.plot(x_line, a * x_line + b, lw=2.5, color="#d62728",
            label=f"Best Fit (R² = {r2:.4f})", zorder=3)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Time to PVC ({units})", fontsize=12)
    ax.set_ylabel(f"Predicted Time to PVC ({units})", fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)

    # Grid & styling
    # ax.grid(True, which="major", color="0.90", linewidth=1)
    # ax.grid(True, which="minor", color="0.95", linewidth=0.8)
    # ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    leg = ax.legend(loc="upper left", frameon=True, framealpha=0.9, facecolor="white")
    for legline in leg.get_lines():
        legline.set_linewidth(2.5)

    plt.tight_layout()
    return fig
