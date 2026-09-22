"""Reproduce Figure 1 and the deterministic cost-threshold validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parent
B, C = 5.0, 1.0
Q = B - C
INITIAL = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)


def payoffs(state):
    x, y, z = state
    values = np.array([
        Q * (x + y) - C * z,
        Q * (x + y / 2),
        B * x,
    ])
    return values, float(state @ values)


def rhs_baseline(_time, state, mu0, mode):
    x, y, z = state
    f, fbar = payoffs(state)
    mu = mu0 if mode == "constant" else mu0 * z
    return np.array([
        x * (f[0] - fbar) - mu * x,
        y * (f[1] - fbar) + mu * x,
        z * (f[2] - fbar),
    ])


def rhs_cost(_time, state, mu0, k):
    x, y, z = state
    f, fbar = payoffs(state)
    effective_mean = fbar - k * x
    return np.array([
        x * (f[0] - k - effective_mean) - mu0 * z * x,
        y * (f[1] - effective_mean) + mu0 * z * x,
        z * (f[2] - effective_mean),
    ])


def integrate(rhs, args):
    solution = solve_ivp(
        rhs, (0.0, 800.0), INITIAL, args=args, method="DOP853",
        rtol=2e-11, atol=2e-13,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    terminal = solution.y[:, -1]
    if abs(terminal.sum() - 1.0) > 2e-8 or terminal.min() < -2e-8:
        raise RuntimeError(f"Simplex check failed: {terminal}")
    return terminal


def main():
    mu = np.linspace(0.0, 2.0, 81)
    constant = np.array([integrate(rhs_baseline, (value, "constant"))[0] for value in mu])
    responsive = np.array([integrate(rhs_baseline, (value, "responsive"))[0] for value in mu])

    mu_cost = np.linspace(0.45, 1.95, 31)
    k_exact = np.sqrt(Q * mu_cost / 2.0)
    residual = np.array([
        integrate(rhs_cost, (float(value), float(k)))[0]
        - integrate(rhs_baseline, (float(value), "constant"))[0]
        for value, k in zip(mu_cost, k_exact)
    ])
    np.savez(
        ROOT / "deterministic_results.npz",
        mu=mu, constant=constant, responsive=responsive,
        mu_cost=mu_cost, k_exact=k_exact, residual=residual,
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    axes[0].plot(mu, constant, "--", color="#264653", label="Constant switching")
    axes[0].plot(mu, responsive, color="#7A5195", label="Defector-responsive switching")
    axes[0].set(xlabel=r"Switching intensity $\mu_0$", ylabel="Terminal ALLC frequency", ylim=(-0.02, 0.82))
    axes[0].legend(frameon=True, fontsize=8)
    axes[0].text(-0.18, 1.03, "A", transform=axes[0].transAxes, fontweight="bold")

    axes[1].plot(mu_cost, k_exact, color="#7A5195", label=r"$\sqrt{(b-c)\mu_0/2}$")
    axes[1].scatter(mu_cost[::3], k_exact[::3], s=18, color="#B56576", label="Validated values")
    axes[1].set(xlabel=r"Switching intensity $\mu_0$", ylabel=r"Break-even cost $k^*$")
    axes[1].legend(frameon=True, fontsize=8)
    axes[1].text(-0.18, 1.03, "B", transform=axes[1].transAxes, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ROOT / "Fig1.pdf", bbox_inches="tight")
    plt.close(fig)

    maximum = np.max(np.abs(residual))
    if maximum > 1e-8:
        raise RuntimeError(f"Cost-threshold residual too large: {maximum}")
    print(f"maximum cost-threshold residual: {maximum:.3e}")


if __name__ == "__main__":
    main()
