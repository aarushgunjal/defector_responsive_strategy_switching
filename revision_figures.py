"""Create the revised finite-population figures from the C++ simulation output."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "revision_stochastic_results.csv"

COLORS = {"constant": "#264653", "responsive": "#7A5195"}
LABELS = {"constant": "Constant switching", "responsive": "Defector-responsive switching"}


def band(ax, x, mean, se, mode, label=True):
    color = COLORS[mode]
    ax.plot(x, mean, color=color, lw=1.8, marker="o", ms=3.5,
            label=LABELS[mode] if label else None)
    ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se,
                    color=color, alpha=0.16, linewidth=0)


def make_core_figure(data: pd.DataFrame) -> None:
    core = data[data.scenario == "core"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.1))
    axes = axes.ravel()

    for mode in ("constant", "responsive"):
        d = core[core["mode"] == mode].sort_values("mu")
        band(axes[0], d.mu.to_numpy(), d.rescue_probability.to_numpy(),
             d.rescue_se.to_numpy(), mode)
        band(axes[1], d.mu.to_numpy(), d.allc_at_rescue.to_numpy(),
             d.allc_at_rescue_se.to_numpy(), mode)
        band(axes[3], d.mu.to_numpy(), d.allc_g100.to_numpy(),
             d.allc_g100_se.to_numpy(), mode)

    generations = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    for mode in ("constant", "responsive"):
        row = core[(core["mode"] == mode) & np.isclose(core.mu, 1.0)].iloc[0]
        means = np.array([row.allc_at_rescue] + [row[f"allc_g{g}"] for g in generations[1:]])
        ses = np.array([row.allc_at_rescue_se] + [row[f"allc_g{g}_se"] for g in generations[1:]])
        band(axes[2], generations, means, ses, mode)

    axes[0].set(xlabel=r"Switching intensity $\mu_0$", ylabel="Rescue probability", ylim=(-0.02, 1.03))
    axes[1].set(xlabel=r"Switching intensity $\mu_0$", ylabel="ALLC frequency at rescue", ylim=(-0.02, 1.03))
    axes[2].set(xlabel="Birth-death generations after rescue", ylabel="Mean ALLC frequency", ylim=(-0.02, 1.03))
    axes[3].set(xlabel=r"Switching intensity $\mu_0$", ylabel="ALLC frequency after $100N$ updates", ylim=(-0.02, 1.03))
    axes[0].legend(fontsize=8, frameon=True, loc="lower right")
    for idx, ax in enumerate(axes):
        ax.text(-0.14, 1.04, chr(65 + idx), transform=ax.transAxes, fontweight="bold")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "Fig2_revised.pdf", bbox_inches="tight")
    plt.close(fig)


def paired_matrix(robust: pd.DataFrame, metric: str) -> np.ndarray:
    rows = []
    for n in (50, 100, 200):
        values = []
        for w in (0.25, 0.5, 1.0):
            for ratio in (3.0, 5.0, 7.0):
                subset = robust[(robust.N == n) & np.isclose(robust.w, w) & np.isclose(robust.b / robust.c, ratio)]
                const = subset[subset["mode"] == "constant"].iloc[0][metric]
                resp = subset[subset["mode"] == "responsive"].iloc[0][metric]
                values.append(resp - const)
        rows.append(values)
    return np.asarray(rows)


def annotated_heatmap(ax, matrix, title, vmin, vmax, cmap):
    im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            color = "white" if abs(value) > 0.6 * max(abs(vmin), abs(vmax)) else "black"
            ax.text(col, row, f"{value:+.2f}", ha="center", va="center", fontsize=6.5, color=color)
    ax.set_yticks(range(3), ["50", "100", "200"])
    labels = [f"{ratio:g}\n{w:g}" for w in (0.25, 0.5, 1.0) for ratio in (3.0, 5.0, 7.0)]
    ax.set_xticks(range(9), labels, fontsize=7)
    ax.set(xlabel="$b/c$ (top), $w$ (bottom)", ylabel="Population size $N$", title=title)
    return im


def make_robustness_figure(data: pd.DataFrame) -> None:
    robust = data[data.scenario == "robustness"].copy()
    cost = data[data.scenario == "cost"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.3))
    axes = axes.ravel()

    preservation = paired_matrix(robust, "allc_g100")
    rescue = paired_matrix(robust, "rescue_probability")
    im0 = annotated_heatmap(axes[0], preservation,
                            r"Post-rescue ALLC advantage at $100N$",
                            -1, 1, "PiYG")
    im1 = annotated_heatmap(axes[1], rescue,
                            "Rescue-probability difference",
                            -0.08, 0.08, "RdBu")
    fig.colorbar(im0, ax=axes[0], fraction=0.045, pad=0.03)
    fig.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.03)

    responsive = cost[cost["mode"] == "responsive"].sort_values("k")
    constant = cost[cost["mode"] == "constant"].iloc[0]
    band(axes[2], responsive.k.to_numpy(), responsive.rescue_probability.to_numpy(),
         responsive.rescue_se.to_numpy(), "responsive", label=False)
    axes[2].axhline(constant.rescue_probability, color=COLORS["constant"], ls="--", lw=1.6,
                    label="Cost-free constant benchmark")
    band(axes[3], responsive.k.to_numpy(), responsive.allc_g100.to_numpy(),
         responsive.allc_g100_se.to_numpy(), "responsive", label=False)
    axes[3].axhline(constant.allc_g100, color=COLORS["constant"], ls="--", lw=1.6,
                    label="Cost-free constant benchmark")
    k_star = np.sqrt(2.0)
    for ax in axes[2:]:
        ax.axvline(k_star, color="#B56576", ls=":", lw=1.6,
                   label=r"Deterministic $k^*=\sqrt{2}$")
        ax.set_xlim(-0.03, 2.03)
        ax.set_ylim(-0.02, 1.03)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.2, frameon=True, loc="lower left")
    axes[2].set(xlabel="Readiness cost $k$", ylabel="Rescue probability")
    axes[3].set(xlabel="Readiness cost $k$", ylabel="ALLC frequency after $100N$ updates")

    for idx, ax in enumerate(axes):
        ax.text(-0.14, 1.06, chr(65 + idx), transform=ax.transAxes, fontweight="bold")
        ax.title.set_fontsize(10)
    fig.tight_layout()
    fig.savefig(ROOT / "Fig3_revised.pdf", bbox_inches="tight")
    plt.close(fig)


def validate(data: pd.DataFrame) -> None:
    if data.isna().any().any():
        raise ValueError("Simulation output contains missing values")
    if not np.allclose(data.censored_fraction, 0.0):
        raise ValueError("At least one trial was censored")
    core_zero = data[(data.scenario == "core") & np.isclose(data.mu, 0.0)]
    if core_zero.rescue_probability.nunique() != 1:
        raise ValueError("The two zero-switching controls do not match")
    robust = data[data.scenario == "robustness"]
    if (paired_matrix(robust, "allc_g100") <= 0).any():
        raise ValueError("Preservation advantage failed in a robustness condition")


def main() -> None:
    data = pd.read_csv(INPUT)
    validate(data)
    make_core_figure(data)
    make_robustness_figure(data)
    print(f"rows={len(data)}; no censoring; figures written to {ROOT}")


if __name__ == "__main__":
    main()
