"""
Convergence Order Verification Experiment
==========================================
Verify that each ODE solver achieves its theoretical convergence order
by solving a problem with known analytical solution.

Test problem:
    dy/dt = -y,  y(0) = 1
    Exact solution: y(t) = exp(-t)

Procedure:
    1. Solve with step sizes h = [1/10, 1/20, 1/50, 1/100, 1/200, 1/500]
    2. Compute global error at t=1: |y_numerical(1) - exp(-1)|
    3. Plot log(error) vs log(h) — slope = convergence order
    4. Verify: Euler → slope≈1, Midpoint → slope≈2, RK4 → slope≈4

For high-dimensional validation:
    Same test but with y ∈ R^{100}, each component independent.
    This simulates a high-dimensional latent space.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from solvers import EulerSolver, MidpointSolver, RK4Solver


def run_convergence_test(dim: int = 1):
    """Run convergence test for all fixed-step solvers.
    
    Args:
        dim: Dimension of the ODE system (1 for scalar, >1 for high-dim)
    """
    # Test problem: dy/dt = -y, y(0) = 1 (vector of ones for high-dim)
    def f(t: float, y: torch.Tensor) -> torch.Tensor:
        return -y
    
    y0 = torch.ones(dim, dtype=torch.float64)
    t_end = 1.0
    exact_solution = y0 * np.exp(-t_end)  # y(1) = exp(-1)
    
    # Step sizes to test
    step_counts = [10, 20, 50, 100, 200, 500, 1000]
    step_sizes = [t_end / n for n in step_counts]
    
    solvers = [EulerSolver(), MidpointSolver(), RK4Solver()]
    results = {}
    
    for solver in solvers:
        errors = []
        for n_steps in step_counts:
            result = solver.solve(f, y0, t_span=(0.0, t_end), n_steps=n_steps)
            y_final = result.ys[-1]
            # L2 norm of error
            error = torch.norm(y_final - exact_solution).item()
            errors.append(error)
        
        results[solver.name] = {
            "errors": errors,
            "order": solver.order,
        }
        
        # Compute empirical convergence order using log-log slope
        log_h = np.log(step_sizes)
        log_e = np.log(errors)
        # Linear regression for slope
        slope, _ = np.polyfit(log_h[:5], log_e[:5], 1)
        print(f"{solver.name:12s} | Theoretical order: {solver.order} | "
              f"Empirical order: {slope:.2f} | "
              f"NFE (finest): {step_counts[-1] * (1 if solver.name == 'Euler' else 2 if solver.name == 'Midpoint' else 4)}")
    
    return step_sizes, results


def plot_convergence(step_sizes, results, dim: int, save_path: str = None):
    """Generate log-log convergence plot."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    colors = {"Euler": "#e74c3c", "Midpoint": "#3498db", "RK4": "#2ecc71"}
    markers = {"Euler": "o", "Midpoint": "s", "RK4": "D"}
    
    for name, data in results.items():
        ax.loglog(
            step_sizes, data["errors"],
            marker=markers[name], color=colors[name],
            linewidth=2, markersize=8, label=f"{name} (order {data['order']})"
        )
    
    # Reference slopes
    h_ref = np.array(step_sizes)
    for order, style, label in [(1, "--", "O(h)"), (2, "-.", "O(h²)"), (4, ":", "O(h⁴)")]:
        # Normalize to match Euler at first point for visual reference
        ref = h_ref ** order
        ref = ref / ref[0] * results["Euler"]["errors"][0] * (0.1 ** (order - 1))
        ax.loglog(h_ref, ref, style, color="gray", alpha=0.5, label=label)
    
    ax.set_xlabel("Step size h", fontsize=13)
    ax.set_ylabel("Global error ||y(1) - y_exact||", fontsize=13)
    ax.set_title(f"Convergence Order Verification (dim={dim})", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(max(step_sizes) * 1.5, min(step_sizes) * 0.7)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved convergence plot to {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("CONVERGENCE ORDER VERIFICATION")
    print("=" * 60)
    
    os.makedirs("results", exist_ok=True)
    
    # 1D test
    print("\n--- 1D Scalar ODE: dy/dt = -y ---")
    step_sizes, results_1d = run_convergence_test(dim=1)
    plot_convergence(step_sizes, results_1d, dim=1, save_path="results/convergence_1d.png")
    
    # High-dimensional test (simulating latent space)
    print("\n--- 100D Latent Space ODE: dy/dt = -y ---")
    step_sizes, results_hd = run_convergence_test(dim=100)
    plot_convergence(step_sizes, results_hd, dim=100, save_path="results/convergence_100d.png")
    
    # 1000D test
    print("\n--- 1000D Latent Space ODE: dy/dt = -y ---")
    step_sizes, results_vhd = run_convergence_test(dim=1000)
    plot_convergence(step_sizes, results_vhd, dim=1000, save_path="results/convergence_1000d.png")
    
    print("\n[OK] Convergence plots saved to results/")


if __name__ == "__main__":
    main()
