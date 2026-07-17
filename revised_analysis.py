"""Reproducible analyses for the revised state-dependent switching manuscript.

The script deliberately distinguishes the continuous-time switching intensity
mu0 from a per-update probability in the finite Moran process.  A rate lambda
is mapped to a probability with p = 1 - exp(-lambda).

Only analyses reported in the submitted manuscript are run or included here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_revised"
FIGURES = ROOT / "figures_revised"
DATA.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

B = 5.0
C = 1.0
Q = B - C
INITIAL = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)


def payoffs(state: np.ndarray, b: float = B, c: float = C) -> tuple[np.ndarray, float]:
    """Return strategy payoffs and mean payoff for (ALLC, TFT, ALLD)."""
    x, y, z = state
    f = np.array([
        (b - c) * (x + y) - c * z,
        (b - c) * (x + y / 2),
        b * x,
    ])
    return f, float(state @ f)


def rhs_baseline(t: float, state: np.ndarray, mu0: float, mode: str) -> np.ndarray:
    x, y, z = state
    f, fbar = payoffs(state)
    mu = mu0 if mode == "fixed" else mu0 * z
    return np.array([
        x * (f[0] - fbar) - mu * x,
        y * (f[1] - fbar) + mu * x,
        z * (f[2] - fbar),
    ])


def rhs_cost(t: float, state: np.ndarray, mu0: float, k: float) -> np.ndarray:
    x, y, z = state
    f, fbar = payoffs(state)
    fbar_eff = fbar - k * x
    mu = mu0 * z
    return np.array([
        x * (f[0] - k - fbar_eff) - mu * x,
        y * (f[1] - fbar_eff) + mu * x,
        z * (f[2] - fbar_eff),
    ])


def integrate_ode(rhs, args: tuple, t_end: float = 800.0) -> np.ndarray:
    sol = solve_ivp(
        rhs,
        (0.0, t_end),
        INITIAL,
        args=args,
        method="DOP853",
        rtol=2e-11,
        atol=2e-13,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    terminal = sol.y[:, -1]
    if abs(terminal.sum() - 1.0) > 2e-8 or terminal.min() < -2e-8:
        raise RuntimeError(f"Simplex validation failed: {terminal}")
    return terminal


def fixed_edge_equilibrium(mu0: np.ndarray | float, b: float = B, c: float = C):
    """ALLC frequency on the fixed-mutation ALLC--TFT boundary."""
    return 1.0 - np.sqrt(2.0 * np.asarray(mu0) / (b - c))


def costly_adaptive_edge_equilibrium(k: np.ndarray | float, b: float = B, c: float = C):
    """ALLC frequency on the mutation-free ALLC--TFT boundary."""
    return 1.0 - 2.0 * np.asarray(k) / (b - c)


def break_even_cost(mu0: np.ndarray | float, b: float = B, c: float = C):
    """Exact boundary-regime threshold obtained by equating the two equilibria."""
    return np.sqrt((b - c) * np.asarray(mu0) / 2.0)


def baseline_and_cost_data() -> None:
    mu = np.linspace(0.0, 2.0, 81)
    fixed = np.array([integrate_ode(rhs_baseline, (m, "fixed"))[0] for m in mu])
    adaptive = np.array([integrate_ode(rhs_baseline, (m, "adaptive"))[0] for m in mu])

    mu_cost = np.linspace(0.45, 1.95, 31)
    k_exact = break_even_cost(mu_cost)
    residual = []
    for m, k in zip(mu_cost, k_exact):
        x_cost = integrate_ode(rhs_cost, (float(m), float(k)))[0]
        x_fixed = integrate_ode(rhs_baseline, (float(m), "fixed"))[0]
        residual.append(x_cost - x_fixed)
    residual = np.asarray(residual)

    np.savez(
        DATA / "deterministic_results.npz",
        mu=mu,
        fixed=fixed,
        adaptive=adaptive,
        mu_cost=mu_cost,
        k_exact=k_exact,
        residual=residual,
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    axes[0].plot(mu, fixed, "--", color="#264653", label="Fixed switching")
    axes[0].plot(mu, adaptive, color="#6D597A", label="Defector-responsive switching")
    axes[0].set(xlabel=r"Switching parameter $\mu_0$", ylabel=r"Terminal ALLC frequency")
    axes[0].set_ylim(-0.02, 0.82)
    axes[0].legend(frameon=True, fontsize=8)
    axes[0].text(-0.18, 1.03, "A", transform=axes[0].transAxes, fontweight="bold")

    axes[1].plot(mu_cost, k_exact, color="#6D597A", label=r"$\sqrt{(b-c)\mu_0/2}$")
    axes[1].scatter(mu_cost[::3], k_exact[::3], s=18, color="#B56576", label="Validated values")
    axes[1].set(xlabel=r"Switching parameter $\mu_0$", ylabel=r"Break-even cost $k^*$")
    axes[1].legend(frameon=True, fontsize=8)
    axes[1].text(-0.18, 1.03, "B", transform=axes[1].transAxes, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "deterministic_and_cost.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"maximum cost-threshold equilibrium residual: {np.max(np.abs(residual)):.3e}")


def expected_payoffs_counts(nc, nt, nd, n: int, b: float, c: float):
    denom = n - 1
    uc = ((np.maximum(nc - 1, 0) + nt) * (b - c) - nd * c) / denom
    ut = (nc * (b - c) + np.maximum(nt - 1, 0) * (b - c) / 2) / denom
    ud = nc * b / denom
    return uc, ut, ud


def moran_batch(
    mu0: float,
    mode: str,
    runs: int,
    seed: int,
    n: int = 100,
    selection: float = 1.0,
    max_pre_steps: int = 100_000,
    post_steps: int = 5_000,
):
    """Vectorized birth--death simulations with a finite post-rescue horizon."""
    rng = np.random.default_rng(seed)
    nc = np.full(runs, n - 1, dtype=np.int16)
    nt = np.zeros(runs, dtype=np.int16)
    nd = np.ones(runs, dtype=np.int16)
    stage = np.zeros(runs, dtype=np.int8)  # 0 pre-rescue; 1 post-rescue; 2 finished; 3 failed
    elapsed_pre = np.zeros(runs, dtype=np.int32)
    remaining = np.zeros(runs, dtype=np.int32)
    allc_at_rescue = np.full(runs, np.nan)
    allc_post = np.full(runs, np.nan)

    maximum_total_steps = max_pre_steps + post_steps
    for _ in range(maximum_total_steps):
        active = np.flatnonzero((stage == 0) | (stage == 1))
        if active.size == 0:
            break

        pre = active[stage[active] == 0]
        elapsed_pre[pre] += 1
        timed_out = pre[elapsed_pre[pre] > max_pre_steps]
        stage[timed_out] = 3
        active = np.flatnonzero((stage == 0) | (stage == 1))
        if active.size == 0:
            break

        a_nc, a_nt, a_nd = nc[active], nt[active], nd[active]
        uc, ut, ud = expected_payoffs_counts(a_nc, a_nt, a_nd, n, B, C)
        fc, ft, fd = np.exp(selection * uc), np.exp(selection * ut), np.exp(selection * ud)
        weights = np.column_stack((a_nc * fc, a_nt * ft, a_nd * fd))
        cumulative = np.cumsum(weights / weights.sum(axis=1, keepdims=True), axis=1)
        r = rng.random(active.size)
        parent = (r[:, None] > cumulative[:, :2]).sum(axis=1)

        if mode == "fixed":
            mutation_rate = np.full(active.size, mu0)
        else:
            mutation_rate = mu0 * (a_nd / n)
        mutation_probability = -np.expm1(-mutation_rate)
        offspring = parent.copy()
        mutate = (parent == 0) & (rng.random(active.size) < mutation_probability)
        offspring[mutate] = 1

        death_r = rng.random(active.size)
        death = np.where(death_r < a_nc / n, 0, np.where(death_r < (a_nc + a_nt) / n, 1, 2))

        for strategy, arr in enumerate((nc, nt, nd)):
            arr[active] += (offspring == strategy).astype(np.int16)
            arr[active] -= (death == strategy).astype(np.int16)

        pre = np.flatnonzero(stage == 0)
        rescued = pre[nd[pre] == 0]
        failed = pre[nd[pre] == n]
        if rescued.size:
            allc_at_rescue[rescued] = nc[rescued] / n
            stage[rescued] = 1
            remaining[rescued] = post_steps
        stage[failed] = 3

        post = np.flatnonzero(stage == 1)
        remaining[post] -= 1
        finished = post[remaining[post] <= 0]
        if finished.size:
            allc_post[finished] = nc[finished] / n
            stage[finished] = 2

    rescued_mask = np.isfinite(allc_at_rescue)
    return {
        "rescue_probability": rescued_mask.mean(),
        "rescue_se": np.sqrt(rescued_mask.mean() * (1 - rescued_mask.mean()) / runs),
        "allc_at_rescue": np.nanmean(allc_at_rescue),
        "allc_at_rescue_se": np.nanstd(allc_at_rescue, ddof=1) / np.sqrt(max(rescued_mask.sum(), 1)),
        "allc_post": np.nanmean(allc_post),
        "allc_post_se": np.nanstd(allc_post, ddof=1) / np.sqrt(max(np.isfinite(allc_post).sum(), 1)),
        "censored_fraction": np.mean((stage == 0) | (stage == 1)),
    }


def stochastic_data(runs: int) -> None:
    mu = np.linspace(0.0, 2.0, 11)
    metrics = [
        "rescue_probability",
        "rescue_se",
        "allc_at_rescue",
        "allc_at_rescue_se",
        "allc_post",
        "allc_post_se",
        "censored_fraction",
    ]
    results = {f"{mode}_{metric}": [] for mode in ("fixed", "adaptive") for metric in metrics}
    for mode_index, mode in enumerate(("fixed", "adaptive")):
        for index, value in enumerate(mu):
            summary = moran_batch(value, mode, runs, seed=20260717 + 1000 * mode_index + index)
            for metric in metrics:
                results[f"{mode}_{metric}"].append(summary[metric])
            print(
                mode,
                f"mu={value:.2f}",
                f"rescue={summary['rescue_probability']:.3f}",
                f"ALLC(post)={summary['allc_post']:.3f}",
            )
    results = {key: np.asarray(value) for key, value in results.items()}
    np.savez(DATA / "stochastic_results.npz", mu=mu, runs=runs, **results)

    colors = {"fixed": "#264653", "adaptive": "#6D597A"}
    labels = {"fixed": "Fixed switching", "adaptive": "Defector-responsive switching"}
    panels = [
        ("rescue_probability", "rescue_se", "Probability ALLD goes extinct"),
        ("allc_at_rescue", "allc_at_rescue_se", "ALLC frequency at rescue"),
        ("allc_post", "allc_post_se", r"ALLC frequency $50N$ updates after rescue"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharex=True)
    for panel_index, (metric, uncertainty, ylabel) in enumerate(panels):
        ax = axes[panel_index]
        for mode in ("fixed", "adaptive"):
            mean = results[f"{mode}_{metric}"]
            se = results[f"{mode}_{uncertainty}"]
            ax.plot(mu, mean, color=colors[mode], label=labels[mode])
            ax.fill_between(mu, mean - 1.96 * se, mean + 1.96 * se, color=colors[mode], alpha=0.15)
        ax.set(xlabel=r"Rate parameter $\mu_0$", ylabel=ylabel, ylim=(-0.03, 1.03))
        ax.text(-0.18, 1.03, chr(65 + panel_index), transform=ax.transAxes, fontweight="bold")
    axes[0].legend(fontsize=7.5, frameon=True)
    fig.tight_layout()
    fig.savefig(FIGURES / "finite_population.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use fewer stochastic trials")
    parser.add_argument("--skip-stochastic", action="store_true")
    args = parser.parse_args()
    baseline_and_cost_data()
    if not args.skip_stochastic:
        stochastic_data(250 if args.quick else 1000)


if __name__ == "__main__":
    main()
