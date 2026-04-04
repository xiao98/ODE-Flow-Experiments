"""
Advanced Analysis Experiments
=============================
1. Wasserstein distance (sliced) for quantitative quality measurement
2. NFE-Quality Pareto frontier
3. Jacobian eigenvalue spectrum along sampling trajectory
4. DOPRI5 adaptive step size distribution
5. Ablation: network size
6. Ablation: training epochs
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

from models.flow_matching import FlowMatchingTrainer, make_moons, make_circles
from solvers import EulerSolver, MidpointSolver, RK4Solver, DormandPrinceSolver


# ============================================================
# Metrics
# ============================================================

def sliced_wasserstein_distance(X, Y, n_projections=200):
    """Sliced Wasserstein distance between two 2D point clouds."""
    X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
    Y_np = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else Y
    d = X_np.shape[1]
    rng = np.random.RandomState(42)
    projections = rng.randn(n_projections, d)
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)

    total = 0.0
    for proj in projections:
        px = X_np @ proj
        py = Y_np @ proj
        total += wasserstein_distance(px, py)
    return total / n_projections


# ============================================================
# 1 & 2. Pareto Frontier: NFE vs Quality
# ============================================================

def run_pareto_analysis(dataset_name="moons", save_dir="results"):
    """Train model, evaluate all solver configs, plot Pareto frontier."""
    print(f"\n=== Pareto Analysis: {dataset_name} ===")

    if dataset_name == "moons":
        data = make_moons(2000, noise=0.05)
    else:
        data = make_circles(2000, noise=0.03)

    trainer = FlowMatchingTrainer(
        data_dim=2, hidden_dim=256, time_dim=64, n_layers=4, lr=1e-3
    )
    trainer.train(data, n_epochs=300, batch_size=256, log_every=100)

    configs = [
        ("Euler", "euler", 10), ("Euler", "euler", 20), ("Euler", "euler", 50),
        ("Euler", "euler", 100), ("Euler", "euler", 200),
        ("Midpoint", "midpoint", 10), ("Midpoint", "midpoint", 20),
        ("Midpoint", "midpoint", 50), ("Midpoint", "midpoint", 100),
        ("RK4", "rk4", 5), ("RK4", "rk4", 10), ("RK4", "rk4", 20),
        ("RK4", "rk4", 50),
    ]

    nfe_mult = {"euler": 1, "midpoint": 2, "rk4": 4}
    colors = {"Euler": "#e74c3c", "Midpoint": "#3498db", "RK4": "#2ecc71", "DOPRI5": "#9b59b6"}

    results = []
    for label, solver_name, n_steps in configs:
        samples = trainer.sample(n_samples=2000, solver_name=solver_name, n_steps=n_steps)
        swd = sliced_wasserstein_distance(samples, data)
        nfe = n_steps * nfe_mult[solver_name]
        results.append((label, nfe, swd, n_steps))
        print(f"  {label} steps={n_steps} NFE={nfe} SWD={swd:.5f}")

    # DOPRI5
    with torch.no_grad():
        dopri = DormandPrinceSolver(atol=1e-5, rtol=1e-5)
        trainer.model.eval()
        z0 = torch.randn(2000, 2)
        def ode_func(t, y):
            t_tensor = torch.full((y.shape[0],), t)
            return trainer.model(y, t_tensor)
        result = dopri.solve(ode_func, z0, t_span=(0.0, 1.0))
        samples_dp = result.ys[-1]
        swd_dp = sliced_wasserstein_distance(samples_dp, data)
        nfe_dp = result.n_fe
        results.append(("DOPRI5", nfe_dp, swd_dp, -1))
        print(f"  DOPRI5 NFE={nfe_dp} SWD={swd_dp:.5f}")

    # Plot Pareto
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in ["Euler", "Midpoint", "RK4", "DOPRI5"]:
        pts = [(nfe, swd) for (l, nfe, swd, _) in results if l == label]
        if pts:
            nfes, swds = zip(*pts)
            ax.plot(nfes, swds, 'o-', color=colors[label], label=label,
                    markersize=7, linewidth=1.5)
            # Annotate step counts
            for (l, nfe, swd, ns) in results:
                if l == label and ns > 0:
                    ax.annotate(f'{ns}', (nfe, swd), textcoords="offset points",
                               xytext=(5, 5), fontsize=7, color=colors[label])

    ax.set_xlabel("Number of Function Evaluations (NFE)", fontsize=12)
    ax.set_ylabel("Sliced Wasserstein Distance", fontsize=12)
    ax.set_title(f"NFE-Quality Pareto Frontier ({dataset_name})", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"pareto_{dataset_name}.png"), dpi=150)
    plt.close()
    print(f"  Saved pareto_{dataset_name}.png")

    return trainer, data, results


# ============================================================
# 3. Jacobian Eigenvalue Spectrum
# ============================================================

def run_jacobian_analysis(trainer, data, save_dir="results"):
    """Compute Jacobian eigenvalues of v_theta along sampling trajectory."""
    print("\n=== Jacobian Eigenvalue Analysis ===")

    trainer.model.eval()
    # Sample a trajectory with RK4
    traj_result = trainer.sample(n_samples=200, solver_name="rk4",
                                 n_steps=50, return_trajectory=True)
    traj = traj_result.ys  # (51, 200, 2)
    ts = traj_result.ts    # (51,)

    # Compute Jacobian at selected time points
    time_indices = list(range(0, len(ts), 5))  # every 5 steps
    eigenvalue_data = []

    for idx in time_indices:
        t_val = ts[idx].item() if isinstance(ts[idx], torch.Tensor) else ts[idx]
        x = traj[idx].clone().detach().requires_grad_(True)  # (200, 2)
        t_tensor = torch.full((x.shape[0],), t_val)

        v = trainer.model(x, t_tensor)  # (200, 2)

        # Compute Jacobian for each sample
        J = torch.zeros(x.shape[0], 2, 2)
        for dim in range(2):
            grad = torch.autograd.grad(v[:, dim].sum(), x, retain_graph=True)[0]
            J[:, dim, :] = grad.detach()

        # Eigenvalues
        eigs = torch.linalg.eigvals(J)  # (200, 2) complex
        eigenvalue_data.append((t_val, eigs.detach()))

    # Plot: eigenvalue real parts vs time
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    times = [t for t, _ in eigenvalue_data]
    real_parts_mean = []
    real_parts_std = []
    imag_parts_mean = []
    cond_numbers = []

    for t_val, eigs in eigenvalue_data:
        re = eigs.real.numpy()  # (200, 2)
        im = eigs.imag.numpy()
        real_parts_mean.append(re.mean(axis=0))
        real_parts_std.append(re.std(axis=0))
        imag_parts_mean.append(np.abs(im).mean(axis=0))
        # Condition number proxy: ratio of max to min eigenvalue magnitude
        mags = np.abs(eigs.numpy())
        cond = mags.max(axis=1) / (mags.min(axis=1) + 1e-8)
        cond_numbers.append((cond.mean(), cond.std()))

    real_parts_mean = np.array(real_parts_mean)
    real_parts_std = np.array(real_parts_std)

    ax = axes[0]
    for dim in range(2):
        ax.plot(times, real_parts_mean[:, dim], linewidth=2,
                label=f'$\\lambda_{dim+1}$ (real part)')
        ax.fill_between(times,
                        real_parts_mean[:, dim] - real_parts_std[:, dim],
                        real_parts_mean[:, dim] + real_parts_std[:, dim],
                        alpha=0.2)
    ax.set_xlabel("Time $t$", fontsize=12)
    ax.set_ylabel("Re($\\lambda$) of Jacobian", fontsize=12)
    ax.set_title("Jacobian Eigenvalues Along Trajectory", fontsize=13)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cond_mean = [c[0] for c in cond_numbers]
    cond_std = [c[1] for c in cond_numbers]
    ax.plot(times, cond_mean, 'o-', color='#e74c3c', linewidth=2, markersize=4)
    ax.fill_between(times,
                    np.array(cond_mean) - np.array(cond_std),
                    np.array(cond_mean) + np.array(cond_std),
                    alpha=0.2, color='#e74c3c')
    ax.set_xlabel("Time $t$", fontsize=12)
    ax.set_ylabel("Condition Number", fontsize=12)
    ax.set_title("Jacobian Condition Number Along Trajectory", fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "jacobian_spectrum.png"), dpi=150)
    plt.close()
    print("  Saved jacobian_spectrum.png")


# ============================================================
# 4. DOPRI5 Step Size Distribution
# ============================================================

def run_dopri5_stepsize_analysis(trainer, save_dir="results"):
    """Analyze how DOPRI5 distributes its step sizes along the trajectory."""
    print("\n=== DOPRI5 Step Size Analysis ===")

    trainer.model.eval()
    with torch.no_grad():
        dopri = DormandPrinceSolver(atol=1e-5, rtol=1e-5)
        z0 = torch.randn(500, 2)

        def ode_func(t, y):
            t_tensor = torch.full((y.shape[0],), t)
            return trainer.model(y, t_tensor)

        result = dopri.solve(ode_func, z0, t_span=(0.0, 1.0))

    ts = result.ts
    if isinstance(ts, torch.Tensor):
        ts = ts.cpu().numpy()
    elif isinstance(ts, list):
        ts = np.array([t.item() if isinstance(t, torch.Tensor) else t for t in ts])

    step_sizes = np.diff(ts)
    step_midpoints = (ts[:-1] + ts[1:]) / 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(step_midpoints, step_sizes, 'o-', color='#9b59b6',
            markersize=3, linewidth=1)
    ax.set_xlabel("Time $t$", fontsize=12)
    ax.set_ylabel("Step size $h$", fontsize=12)
    ax.set_title("DOPRI5 Adaptive Step Size", fontsize=13)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(step_sizes, bins=25, color='#9b59b6', alpha=0.7, edgecolor='white')
    ax.set_xlabel("Step size $h$", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Step Size Distribution (total {len(step_sizes)} steps, {result.n_fe} NFE)",
                 fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dopri5_stepsize.png"), dpi=150)
    plt.close()
    print(f"  Saved dopri5_stepsize.png ({len(step_sizes)} steps)")


# ============================================================
# 5. Ablation: Network Size
# ============================================================

def run_network_ablation(dataset_name="moons", save_dir="results"):
    """How does network capacity affect the quality gap between solvers?"""
    print(f"\n=== Network Size Ablation: {dataset_name} ===")

    if dataset_name == "moons":
        data = make_moons(2000, noise=0.05)
    else:
        data = make_circles(2000, noise=0.03)

    hidden_dims = [64, 128, 256, 512]
    solver_configs = [
        ("Euler-50", "euler", 50),
        ("RK4-20", "rk4", 20),
    ]

    results = {cfg[0]: [] for cfg in solver_configs}

    for hdim in hidden_dims:
        print(f"  Training hidden_dim={hdim}...")
        trainer = FlowMatchingTrainer(
            data_dim=2, hidden_dim=hdim, time_dim=64, n_layers=4, lr=1e-3
        )
        trainer.train(data, n_epochs=300, batch_size=256, log_every=300)

        for label, solver_name, n_steps in solver_configs:
            samples = trainer.sample(n_samples=2000, solver_name=solver_name, n_steps=n_steps)
            swd = sliced_wasserstein_distance(samples, data)
            results[label].append(swd)
            print(f"    {label} SWD={swd:.5f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"Euler-50": "#e74c3c", "RK4-20": "#2ecc71"}
    for label in results:
        ax.plot(hidden_dims, results[label], 'o-', color=colors[label],
                label=f"{label} ({50 if 'Euler' in label else 80} NFE)",
                markersize=7, linewidth=2)

    ax.set_xlabel("Hidden Dimension", fontsize=12)
    ax.set_ylabel("Sliced Wasserstein Distance", fontsize=12)
    ax.set_title(f"Effect of Network Capacity on Solver Performance ({dataset_name})", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"ablation_network_{dataset_name}.png"), dpi=150)
    plt.close()
    print(f"  Saved ablation_network_{dataset_name}.png")


# ============================================================
# 6. Ablation: Training Epochs
# ============================================================

def run_training_ablation(dataset_name="moons", save_dir="results"):
    """How does training duration affect the quality gap between solvers?"""
    print(f"\n=== Training Epochs Ablation: {dataset_name} ===")

    if dataset_name == "moons":
        data = make_moons(2000, noise=0.05)
    else:
        data = make_circles(2000, noise=0.03)

    epoch_checkpoints = [20, 50, 100, 200, 300, 500]
    solver_configs = [
        ("Euler-50", "euler", 50),
        ("RK4-20", "rk4", 20),
    ]

    trainer = FlowMatchingTrainer(
        data_dim=2, hidden_dim=256, time_dim=64, n_layers=4, lr=1e-3
    )

    results = {cfg[0]: [] for cfg in solver_configs}
    data_dev = data.clone()
    trainer.model.train()
    current_epoch = 0

    for target_epoch in epoch_checkpoints:
        # Train incrementally
        extra = target_epoch - current_epoch
        if extra > 0:
            trainer.train(data_dev, n_epochs=extra, batch_size=256, log_every=1000)
            current_epoch = target_epoch

        for label, solver_name, n_steps in solver_configs:
            samples = trainer.sample(n_samples=2000, solver_name=solver_name, n_steps=n_steps)
            swd = sliced_wasserstein_distance(samples, data)
            results[label].append(swd)

        print(f"  Epoch {target_epoch}: Euler-50 SWD={results['Euler-50'][-1]:.5f}, "
              f"RK4-20 SWD={results['RK4-20'][-1]:.5f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {"Euler-50": "#e74c3c", "RK4-20": "#2ecc71"}
    for label in results:
        ax.plot(epoch_checkpoints, results[label], 'o-', color=colors[label],
                label=f"{label}", markersize=7, linewidth=2)

    ax.set_xlabel("Training Epochs", fontsize=12)
    ax.set_ylabel("Sliced Wasserstein Distance", fontsize=12)
    ax.set_title(f"Effect of Training Duration on Solver Performance ({dataset_name})", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"ablation_training_{dataset_name}.png"), dpi=150)
    plt.close()
    print(f"  Saved ablation_training_{dataset_name}.png")


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs("results", exist_ok=True)

    # Pareto analysis (also returns trained model for reuse)
    trainer, data, _ = run_pareto_analysis("moons")

    # Jacobian analysis (reuse trained model)
    run_jacobian_analysis(trainer, data)

    # DOPRI5 step size (reuse trained model)
    run_dopri5_stepsize_analysis(trainer)

    # Pareto for circles
    run_pareto_analysis("circles")

    # Ablations
    run_network_ablation("moons")
    run_training_ablation("moons")

    print("\n=== All advanced analyses complete! ===")


if __name__ == "__main__":
    main()
