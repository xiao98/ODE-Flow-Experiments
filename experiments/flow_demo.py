"""
Flow Matching End-to-End Demo
==============================
Train a Flow Matching model on 2D toy datasets and compare
sampling quality across different ODE solvers.

This demonstrates the practical application of numerical integration
in generative models:
1. Train v_θ(x, t) on a 2D dataset
2. Sample using Euler, Midpoint, RK4 with different step counts
3. Compare: quality vs NFE tradeoff
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.flow_matching import FlowMatchingTrainer, make_moons, make_circles
from solvers import EulerSolver, MidpointSolver, RK4Solver, DormandPrinceSolver


def train_and_sample(
    dataset_name: str = "moons",
    n_data: int = 2000,
    n_epochs: int = 300,
    device: str = "cpu",
    save_dir: str = "results",
):
    """Train Flow Matching model and generate comparison samples."""
    
    print(f"\n{'='*50}")
    print(f"Training on {dataset_name} dataset")
    print(f"{'='*50}")
    
    # Generate training data
    if dataset_name == "moons":
        data = make_moons(n_data, noise=0.05)
    elif dataset_name == "circles":
        data = make_circles(n_data, noise=0.03)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Train the model
    trainer = FlowMatchingTrainer(
        data_dim=2,
        hidden_dim=256,
        time_dim=64,
        n_layers=4,
        lr=1e-3,
        device=device,
    )
    
    history = trainer.train(data, n_epochs=n_epochs, batch_size=256, log_every=50)
    
    # Plot training loss
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(history["loss"], linewidth=1.5, color="#3498db")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("CFM Loss", fontsize=12)
    ax.set_title(f"Training Loss ({dataset_name})", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"training_loss_{dataset_name}.png"), dpi=150)
    plt.close()
    
    # === Sample with different solvers and step counts ===
    solver_configs = [
        ("euler", 10), ("euler", 50), ("euler", 200),
        ("midpoint", 10), ("midpoint", 50), ("midpoint", 100),
        ("rk4", 5), ("rk4", 20), ("rk4", 50),
    ]
    
    n_samples = 1000
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Plot training data
    axes[0, 0].scatter(data[:, 0], data[:, 1], s=3, alpha=0.5, c="#2c3e50")
    axes[0, 0].set_title("Training Data", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlim(-2.5, 3)
    axes[0, 0].set_ylim(-2, 2)
    axes[0, 0].set_aspect("equal")
    
    plot_idx = 1
    for solver_name, n_steps in solver_configs:
        row, col = divmod(plot_idx, 4)
        ax = axes[row, col]
        
        try:
            samples = trainer.sample(
                n_samples=n_samples,
                solver_name=solver_name,
                n_steps=n_steps,
            )
            
            # Compute NFE
            if solver_name == "euler":
                nfe = n_steps
            elif solver_name == "midpoint":
                nfe = n_steps * 2
            elif solver_name == "rk4":
                nfe = n_steps * 4
            
            color_map = {"euler": "#e74c3c", "midpoint": "#3498db", "rk4": "#2ecc71"}
            
            ax.scatter(
                samples[:, 0].cpu(), samples[:, 1].cpu(),
                s=3, alpha=0.5, c=color_map[solver_name]
            )
            ax.set_title(f"{solver_name.upper()} (steps={n_steps}, NFE={nfe})", fontsize=11)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{str(e)[:50]}", transform=ax.transAxes,
                    ha="center", fontsize=9)
            ax.set_title(f"{solver_name} steps={n_steps}", fontsize=11)
        
        ax.set_xlim(-2.5, 3)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        plot_idx += 1
    
    # Adaptive solver
    row, col = divmod(plot_idx, 4)
    ax = axes[row, col]
    try:
        dopri = DormandPrinceSolver(atol=1e-5, rtol=1e-5)
        trainer.model.eval()

        with torch.no_grad():
            z0 = torch.randn(n_samples, 2, device=trainer.device)

            def ode_func(t, y):
                t_tensor = torch.full((y.shape[0],), t, device=y.device)
                return trainer.model(y, t_tensor)

            result = dopri.solve(ode_func, z0, t_span=(0.0, 1.0))
            samples_dp = result.ys[-1]

        ax.scatter(
            samples_dp[:, 0].cpu(), samples_dp[:, 1].cpu(),
            s=3, alpha=0.5, c="#9b59b6"
        )
        ax.set_title(f"DOPRI5 (adaptive, NFE={result.n_fe})", fontsize=11)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error:\n{str(e)[:50]}", transform=ax.transAxes,
                ha="center", fontsize=9)
        ax.set_title("DOPRI5 (adaptive)", fontsize=11)
    
    ax.set_xlim(-2.5, 3)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    
    # Hide unused subplots
    for idx in range(plot_idx + 1, 12):
        row, col = divmod(idx, 4)
        axes[row, col].set_visible(False)
    
    plt.suptitle(f"Flow Matching Samples: Solver Comparison ({dataset_name})",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"solver_comparison_{dataset_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Solver comparison saved")
    
    # === Trajectory visualization ===
    plot_trajectories(trainer, dataset_name, save_dir)
    
    return trainer


def plot_trajectories(trainer, dataset_name, save_dir):
    """Visualize ODE integration trajectories from noise to data."""
    n_traj = 50
    n_steps = 50
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    solver_names = ["euler", "midpoint", "rk4"]
    colors_map = {"euler": "#e74c3c", "midpoint": "#3498db", "rk4": "#2ecc71"}
    
    for ax, solver_name in zip(axes, solver_names):
        result = trainer.sample(
            n_samples=n_traj,
            solver_name=solver_name,
            n_steps=n_steps,
            return_trajectory=True,
        )
        
        traj = result.ys.cpu().numpy()  # (n_steps+1, n_traj, 2)
        
        # Plot trajectories
        for i in range(n_traj):
            ax.plot(traj[:, i, 0], traj[:, i, 1],
                    color=colors_map[solver_name], alpha=0.15, linewidth=0.8)
        
        # Mark start (noise) and end (data)
        ax.scatter(traj[0, :, 0], traj[0, :, 1], 
                   c="gray", s=15, zorder=5, alpha=0.5, label="Noise (t=0)")
        ax.scatter(traj[-1, :, 0], traj[-1, :, 1],
                   c=colors_map[solver_name], s=15, zorder=5, alpha=0.7, label="Generated (t=1)")
        
        ax.set_title(f"{solver_name.upper()} Trajectories", fontsize=13)
        ax.legend(fontsize=9, loc="upper left")
        ax.set_xlim(-3, 3.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
    
    plt.suptitle(f"ODE Integration Trajectories ({dataset_name})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"trajectories_{dataset_name}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Trajectories saved")


def main():
    os.makedirs("results", exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Train on moons dataset
    train_and_sample("moons", n_data=2000, n_epochs=300, device=device)
    
    # Train on circles dataset
    train_and_sample("circles", n_data=2000, n_epochs=300, device=device)
    
    print("\n" + "=" * 50)
    print("[OK] All flow matching experiments complete!")
    print("Results saved to results/")


if __name__ == "__main__":
    main()
