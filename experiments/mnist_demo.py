"""
MNIST Flow Matching Experiment
==============================
Train Flow Matching on MNIST digits (via PCA reduction to 64D),
compare ODE solvers, and visualize generated samples.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from models.flow_matching import FlowMatchingTrainer
from solvers import EulerSolver, MidpointSolver, RK4Solver, DormandPrinceSolver


def load_mnist_pca(n_components=64, n_train=10000):
    """Load MNIST, flatten, apply PCA."""
    import torchvision
    import torchvision.transforms as T

    dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=T.ToTensor()
    )
    # Take subset for speed
    images = dataset.data[:n_train].float() / 255.0  # (N, 28, 28)
    images_flat = images.view(n_train, -1).numpy()     # (N, 784)

    pca = PCA(n_components=n_components)
    latent = pca.fit_transform(images_flat)            # (N, 64)
    # Normalize to roughly unit variance
    latent_std = latent.std()
    latent = latent / latent_std

    return torch.tensor(latent, dtype=torch.float32), pca, latent_std, images_flat


def decode_samples(samples_latent, pca, latent_std):
    """Decode PCA latent samples back to 28x28 images."""
    latent_np = samples_latent.cpu().numpy() * latent_std
    images_flat = pca.inverse_transform(latent_np)
    images_flat = np.clip(images_flat, 0, 1)
    return images_flat.reshape(-1, 28, 28)


def plot_samples_grid(images, title, save_path, nrow=10, ncol=10):
    """Plot a grid of 28x28 images."""
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol, nrow))
    for i in range(nrow):
        for j in range(ncol):
            idx = i * ncol + j
            if idx < len(images):
                axes[i, j].imshow(images[idx], cmap='gray', vmin=0, vmax=1)
            axes[i, j].axis('off')
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_mnist_experiment(save_dir="results"):
    """Full MNIST Flow Matching experiment."""
    print("\n=== MNIST Flow Matching Experiment ===")
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    print("Loading MNIST and fitting PCA (64 components)...")
    data, pca, latent_std, raw_images = load_mnist_pca(n_components=64, n_train=10000)
    print(f"  Data shape: {data.shape}, latent std: {latent_std:.3f}")
    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA explained variance: {explained:.1%}")

    # Train
    print("Training Flow Matching (64D, 500 epochs)...")
    trainer = FlowMatchingTrainer(
        data_dim=64, hidden_dim=512, time_dim=128, n_layers=6, lr=1e-3
    )
    history = trainer.train(data, n_epochs=500, batch_size=256, log_every=100)

    # Plot training loss
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(history["loss"], linewidth=1.5, color="#3498db")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("CFM Loss", fontsize=12)
    ax.set_title("MNIST Training Loss (64D PCA latent)", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mnist_training_loss.png"), dpi=150)
    plt.close()

    # === Solver comparison ===
    from experiments.advanced_analysis import sliced_wasserstein_distance

    solver_configs = [
        ("Euler", "euler", 20),
        ("Euler", "euler", 50),
        ("Euler", "euler", 200),
        ("Midpoint", "midpoint", 20),
        ("Midpoint", "midpoint", 50),
        ("RK4", "rk4", 10),
        ("RK4", "rk4", 20),
        ("RK4", "rk4", 50),
    ]

    nfe_mult = {"euler": 1, "midpoint": 2, "rk4": 4}
    colors = {"Euler": "#e74c3c", "Midpoint": "#3498db", "RK4": "#2ecc71", "DOPRI5": "#9b59b6"}

    results = []
    all_samples = {}

    for label, solver_name, n_steps in solver_configs:
        samples = trainer.sample(n_samples=1000, solver_name=solver_name, n_steps=n_steps)
        nfe = n_steps * nfe_mult[solver_name]
        swd = sliced_wasserstein_distance(samples, data[:1000])
        results.append((label, nfe, swd, n_steps))
        all_samples[(label, n_steps)] = samples
        print(f"  {label} steps={n_steps} NFE={nfe} SWD={swd:.5f}")

    # DOPRI5
    with torch.no_grad():
        dopri = DormandPrinceSolver(atol=1e-5, rtol=1e-5)
        trainer.model.eval()
        z0 = torch.randn(1000, 64)
        def ode_func(t, y):
            t_tensor = torch.full((y.shape[0],), t)
            return trainer.model(y, t_tensor)
        result = dopri.solve(ode_func, z0, t_span=(0.0, 1.0))
        samples_dp = result.ys[-1]
        swd_dp = sliced_wasserstein_distance(samples_dp, data[:1000])
        nfe_dp = result.n_fe
        results.append(("DOPRI5", nfe_dp, swd_dp, -1))
        all_samples[("DOPRI5", -1)] = samples_dp
        print(f"  DOPRI5 NFE={nfe_dp} SWD={swd_dp:.5f}")

    # === Pareto plot ===
    fig, ax = plt.subplots(figsize=(8, 5))
    for label in ["Euler", "Midpoint", "RK4", "DOPRI5"]:
        pts = [(nfe, swd) for (l, nfe, swd, _) in results if l == label]
        if pts:
            nfes, swds = zip(*pts)
            ax.plot(nfes, swds, 'o-', color=colors[label], label=label,
                    markersize=7, linewidth=1.5)
    ax.set_xlabel("NFE", fontsize=12)
    ax.set_ylabel("Sliced Wasserstein Distance", fontsize=12)
    ax.set_title("MNIST: NFE-Quality Pareto Frontier (64D PCA)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mnist_pareto.png"), dpi=150)
    plt.close()

    # === Sample grids ===
    # Best config per solver
    show_configs = [
        ("Euler", 50), ("RK4", 20), ("DOPRI5", -1)
    ]
    fig, axes_rows = plt.subplots(3, 10, figsize=(12, 4))
    for row_idx, (label, ns) in enumerate(show_configs):
        samples = all_samples[(label, ns)]
        images = decode_samples(samples[:10], pca, latent_std)
        nfe = ns * nfe_mult.get(label.lower(), 1) if ns > 0 else nfe_dp
        for col in range(10):
            axes_rows[row_idx, col].imshow(images[col], cmap='gray', vmin=0, vmax=1)
            axes_rows[row_idx, col].axis('off')
        axes_rows[row_idx, 0].set_ylabel(
            f"{label}\n{'adpt' if ns < 0 else str(ns)+'s'} ({nfe} NFE)",
            fontsize=9, rotation=0, labelpad=55, va='center')

    plt.suptitle("MNIST Samples: Solver Comparison", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mnist_solver_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Large grid with best solver
    best_key = ("RK4", 50)
    images = decode_samples(all_samples[best_key][:100], pca, latent_std)
    plot_samples_grid(images, "MNIST Generated Samples (RK4, 50 steps)",
                      os.path.join(save_dir, "mnist_samples_grid.png"))

    print("\n=== MNIST experiment complete! ===")


if __name__ == "__main__":
    run_mnist_experiment()
