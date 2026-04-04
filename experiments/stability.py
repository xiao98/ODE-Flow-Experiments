"""
Numerical Stability Analysis
==============================
Analyze the stability properties of each ODE solver by:
1. Plotting stability regions in the complex plane
2. Demonstrating instability with stiff test problems
3. Showing how step size affects stability in high-dimensional latent spaces

Key concepts for interviews:
- Stability region: set of h·λ ∈ C where |R(h·λ)| ≤ 1
- For dy/dt = λy with λ<0, method is stable iff h·λ is inside the region
- Stiff problems have eigenvalues with very different magnitudes
- In latent spaces: Jacobian eigenvalues determine local stiffness

Test problem (stiff):
    dy/dt = A·y where A has eigenvalues spread across scales
    This models a high-dimensional latent space with varying dynamics
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


def compute_stability_region(method: str, grid_size: int = 500):
    """Compute the stability region for a given method.
    
    For the test equation dy/dt = λy, the numerical solution is:
        y_{n+1} = R(z) · y_n, where z = h·λ
    
    The stability region is {z ∈ C : |R(z)| ≤ 1}.
    
    Stability functions R(z):
        Euler:    R(z) = 1 + z
        Midpoint: R(z) = 1 + z + z²/2
        RK4:      R(z) = 1 + z + z²/2 + z³/6 + z⁴/24
    """
    # Create grid in complex plane
    x = np.linspace(-5, 2, grid_size)
    y = np.linspace(-3.5, 3.5, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    
    if method == "Euler":
        R = 1 + Z
    elif method == "Midpoint":
        R = 1 + Z + Z**2 / 2
    elif method == "RK4":
        R = 1 + Z + Z**2 / 2 + Z**3 / 6 + Z**4 / 24
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return X, Y, np.abs(R)


def plot_stability_regions(save_path: str = None):
    """Plot stability regions for all three methods."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    methods = ["Euler", "Midpoint", "RK4"]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    
    for ax, method, color in zip(axes, methods, colors):
        X, Y, R_abs = compute_stability_region(method)
        
        # Stability region: |R(z)| ≤ 1
        ax.contour(X, Y, R_abs, levels=[1.0], colors=[color], linewidths=2)
        ax.contourf(X, Y, R_abs, levels=[0, 1.0], colors=[color], alpha=0.2)
        
        ax.set_xlabel("Re(hλ)", fontsize=12)
        ax.set_ylabel("Im(hλ)", fontsize=12)
        ax.set_title(f"{method} Stability Region", fontsize=13)
        ax.set_aspect("equal")
        ax.axhline(y=0, color="k", linewidth=0.5)
        ax.axvline(x=0, color="k", linewidth=0.5)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(-5, 2)
        ax.set_ylim(-3.5, 3.5)
    
    plt.suptitle("Stability Regions of ODE Solvers", fontsize=15, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved stability regions to {save_path}")
    plt.close()


def stability_demo_1d(save_path: str = None):
    """Demonstrate stability vs instability with a stiff 1D problem.
    
    Problem: dy/dt = -15y, y(0) = 1
    Exact: y(t) = exp(-15t)
    
    Euler stability requires h < 2/15 ≈ 0.133
    We'll test h = 0.1 (stable) and h = 0.15 (unstable for Euler)
    """
    lam = -15.0
    
    def f(t, y):
        return lam * y
    
    y0 = torch.tensor([1.0], dtype=torch.float64)
    t_end = 2.0
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Exact solution
    t_exact = np.linspace(0, t_end, 200)
    y_exact = np.exp(lam * t_exact)
    
    for ax, n_steps, title in zip(axes, [20, 12], ["Stable (h=0.1)", "Unstable (h≈0.167)"]):
        ax.plot(t_exact, y_exact, "k-", linewidth=2, label="Exact", zorder=5)
        
        solvers = [EulerSolver(), MidpointSolver(), RK4Solver()]
        colors = ["#e74c3c", "#3498db", "#2ecc71"]
        
        for solver, color in zip(solvers, colors):
            try:
                result = solver.solve(f, y0, t_span=(0.0, t_end), n_steps=n_steps)
                ts = result.ts.numpy()
                ys = result.ys[:, 0].numpy()
                
                # Clip for plotting (unstable can explode)
                ys_clipped = np.clip(ys, -5, 5)
                ax.plot(ts, ys_clipped, "o-", color=color, markersize=3,
                        linewidth=1.5, label=f"{solver.name}", alpha=0.8)
            except Exception as e:
                print(f"{solver.name} failed with h={t_end/n_steps:.3f}: {e}")
        
        h = t_end / n_steps
        ax.set_title(f"{title}, h={h:.3f}\nEuler stability: h·|λ|={h*abs(lam):.2f} {'< 2 [OK]' if h*abs(lam) < 2 else '> 2 ✗'}", fontsize=12)
        ax.set_xlabel("t", fontsize=12)
        ax.set_ylabel("y(t)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-2, 2)
    
    plt.suptitle("Stability Analysis: dy/dt = -15y", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved stability demo to {save_path}")
    plt.close()


def high_dim_stiffness_demo(save_path: str = None):
    """Demonstrate stiffness in a high-dimensional system.
    
    Problem: dy/dt = A·y where A = diag(-1, -10, -100, ..., -1000)
    This simulates a latent space where different dimensions have
    very different dynamics (stiff problem).
    
    Key insight: The step size must satisfy stability for ALL eigenvalues.
    """
    # Create a diagonal system with spread eigenvalues
    eigenvalues = torch.tensor([-1.0, -5.0, -10.0, -50.0, -100.0], dtype=torch.float64)
    dim = len(eigenvalues)
    
    def f(t, y):
        return eigenvalues * y
    
    y0 = torch.ones(dim, dtype=torch.float64)
    t_end = 1.0
    
    step_counts = [5, 10, 20, 50, 100]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    solvers = [EulerSolver(), MidpointSolver(), RK4Solver()]
    colors_solvers = ["#e74c3c", "#3498db", "#2ecc71"]
    
    for ax, solver, color in zip(axes, solvers, colors_solvers):
        errors_per_dim = []
        for n_steps in step_counts:
            result = solver.solve(f, y0, t_span=(0.0, t_end), n_steps=n_steps)
            y_final = result.ys[-1]
            exact = y0 * torch.exp(eigenvalues * t_end)
            per_dim_error = torch.abs(y_final - exact).numpy()
            errors_per_dim.append(per_dim_error)
        
        errors_per_dim = np.array(errors_per_dim)  # (n_step_counts, dim)
        
        for d in range(dim):
            lam = eigenvalues[d].item()
            ax.semilogy(
                step_counts, errors_per_dim[:, d],
                "o-", markersize=5, linewidth=1.5,
                label=f"λ={lam:.0f}"
            )
        
        ax.set_xlabel("Number of steps", fontsize=12)
        ax.set_ylabel("Per-dimension error", fontsize=12)
        ax.set_title(f"{solver.name} (order {solver.order})", fontsize=13, color=color)
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3, which="both")
    
    plt.suptitle("High-Dimensional Stiffness: Different Eigenvalues per Dimension",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved high-dim stiffness demo to {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("NUMERICAL STABILITY ANALYSIS")
    print("=" * 60)
    
    os.makedirs("results", exist_ok=True)
    
    print("\n--- Stability Regions ---")
    plot_stability_regions(save_path="results/stability_regions.png")
    
    print("\n--- 1D Stability Demo ---")
    stability_demo_1d(save_path="results/stability_demo_1d.png")
    
    print("\n--- High-Dimensional Stiffness ---")
    high_dim_stiffness_demo(save_path="results/high_dim_stiffness.png")
    
    print("\n[OK] Stability analysis plots saved to results/")


if __name__ == "__main__":
    main()
