"""
Flow Matching Training Pipeline
================================
Implementation of Conditional Flow Matching (CFM) with Optimal Transport
probability paths for training generative models.

Theory:
    Flow Matching trains a neural network v_θ(x, t) to approximate
    a target vector field u_t(x) that generates a probability path
    from a prior distribution p_0 (e.g., Gaussian noise) to the
    data distribution p_1.
    
    Conditional Flow Matching (CFM):
        Instead of matching the marginal vector field (hard to compute),
        we match conditional vector fields u_t(x | x_1) which are
        analytically known for chosen probability paths.
    
    Optimal Transport (OT) Path:
        x_t = (1 - t) · x_0 + t · x_1
        u_t(x | x_1) = x_1 - x_0
    
    Loss:
        L(θ) = E_{t, x_0, x_1} [ ||v_θ(x_t, t) - (x_1 - x_0)||² ]

References:
    - Lipman et al., "Flow Matching for Generative Modeling" (2023)
    - Tong et al., "Conditional Flow Matching" (ICML 2024)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict

from .vector_field import VectorFieldMLP


def make_moons(n_samples: int = 1000, noise: float = 0.05) -> torch.Tensor:
    """Generate 2D moons dataset (like sklearn but pure torch)."""
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    outer_circ_x = torch.cos(torch.linspace(0, torch.pi, n_samples_out))
    outer_circ_y = torch.sin(torch.linspace(0, torch.pi, n_samples_out))
    inner_circ_x = 1 - torch.cos(torch.linspace(0, torch.pi, n_samples_in))
    inner_circ_y = 1 - torch.sin(torch.linspace(0, torch.pi, n_samples_in)) - 0.5
    
    X = torch.cat([
        torch.stack([outer_circ_x, outer_circ_y], dim=1),
        torch.stack([inner_circ_x, inner_circ_y], dim=1),
    ], dim=0)
    
    X += torch.randn_like(X) * noise
    return X


def make_circles(n_samples: int = 1000, noise: float = 0.05) -> torch.Tensor:
    """Generate 2D concentric circles dataset."""
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out
    
    theta_out = torch.linspace(0, 2 * torch.pi, n_samples_out + 1)[:-1]
    theta_in = torch.linspace(0, 2 * torch.pi, n_samples_in + 1)[:-1]
    
    outer = torch.stack([torch.cos(theta_out), torch.sin(theta_out)], dim=1)
    inner = 0.5 * torch.stack([torch.cos(theta_in), torch.sin(theta_in)], dim=1)
    
    X = torch.cat([outer, inner], dim=0)
    X += torch.randn_like(X) * noise
    return X


def make_checkerboard(n_samples: int = 1000) -> torch.Tensor:
    """Generate 2D checkerboard dataset."""
    x1 = torch.rand(n_samples) * 4 - 2
    x2 = torch.rand(n_samples) * 4 - 2
    
    mask = ((torch.floor(x1) + torch.floor(x2)) % 2 == 0).float()
    # Resample rejected points
    while mask.sum() < n_samples // 2:
        idx = mask == 0
        x1[idx] = torch.rand(idx.sum()) * 4 - 2
        x2[idx] = torch.rand(idx.sum()) * 4 - 2
        mask = ((torch.floor(x1) + torch.floor(x2)) % 2 == 0).float()
    
    X = torch.stack([x1[mask == 1], x2[mask == 1]], dim=1)
    # Normalize
    X = X[:n_samples]
    if len(X) < n_samples:
        X = torch.cat([X, X[:n_samples - len(X)]], dim=0)
    return X


class FlowMatchingTrainer:
    """Conditional Flow Matching trainer with OT paths.
    
    This class handles:
    1. Sampling from the training data distribution p_1
    2. Sampling noise from prior p_0 = N(0, I)
    3. Constructing interpolated samples x_t along OT paths
    4. Computing the CFM loss
    5. Training loop with logging
    
    After training, the learned vector field can be integrated with any
    ODE solver to generate new samples: z ~ N(0,I) → solve ODE → x_generated
    
    Usage:
        trainer = FlowMatchingTrainer(data_dim=2)
        trainer.train(data, n_epochs=100)
        samples = trainer.sample(n_samples=500, solver='rk4')
    """
    
    def __init__(
        self,
        data_dim: int = 2,
        hidden_dim: int = 256,
        time_dim: int = 64,
        n_layers: int = 4,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.data_dim = data_dim
        
        # Create vector field network
        self.model = VectorFieldMLP(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            n_layers=n_layers,
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_history = []
    
    def compute_loss(
        self, x1: torch.Tensor, batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Compute the Conditional Flow Matching loss.
        
        L(θ) = E_{t, x_0, x_1} [ ||v_θ(x_t, t) - u_t||² ]
        
        where:
            x_0 ~ N(0, I)               (noise / prior)
            x_1 ~ p_data                (training data)
            t   ~ U(0, 1)               (uniform time)
            x_t = (1-t)·x_0 + t·x_1     (OT interpolation)
            u_t = x_1 - x_0             (target velocity)
        
        Args:
            x1: Training data batch, shape (batch_size, data_dim)
        Returns:
            Scalar loss value
        """
        if batch_size is None:
            batch_size = x1.shape[0]
        
        # Sample noise from prior p_0 = N(0, I)
        x0 = torch.randn(batch_size, self.data_dim, device=self.device)
        
        # Sample random time t ~ U(0, 1)
        t = torch.rand(batch_size, device=self.device)
        
        # Construct x_t along optimal transport path
        # x_t = (1 - t) * x0 + t * x1
        t_expand = t.unsqueeze(-1)  # (B, 1) for broadcasting
        xt = (1 - t_expand) * x0 + t_expand * x1
        
        # Target velocity field (analytical for OT path)
        ut = x1 - x0  # (B, data_dim)
        
        # Predict velocity with neural network
        vt = self.model(xt, t)  # (B, data_dim)
        
        # MSE loss
        loss = torch.mean((vt - ut) ** 2)
        return loss
    
    def train(
        self,
        data: torch.Tensor,
        n_epochs: int = 200,
        batch_size: int = 256,
        log_every: int = 20,
    ) -> Dict[str, list]:
        """Train the flow matching model.
        
        Args:
            data: Training data, shape (n_samples, data_dim)
            n_epochs: Number of training epochs
            batch_size: Batch size
            log_every: Print loss every N epochs
            
        Returns:
            Dictionary with training history
        """
        data = data.to(self.device)
        n_samples = data.shape[0]
        self.model.train()
        
        for epoch in range(n_epochs):
            # Random batch sampling
            perm = torch.randperm(n_samples, device=self.device)
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                idx = perm[i : i + batch_size]
                x1_batch = data[idx]
                
                loss = self.compute_loss(x1_batch)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)
            
            if (epoch + 1) % log_every == 0:
                print(f"Epoch {epoch+1:4d}/{n_epochs} | Loss: {avg_loss:.6f}")
        
        return {"loss": self.loss_history}
    
    @torch.no_grad()
    def sample(
        self,
        n_samples: int = 500,
        solver_name: str = "rk4",
        n_steps: int = 100,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """Generate samples by integrating the learned vector field.
        
        Sample z_0 ~ N(0, I) and solve the ODE:
            dz/dt = v_θ(z, t),  t ∈ [0, 1]
        
        The solution z(1) ≈ samples from the data distribution.
        
        Args:
            n_samples: Number of samples to generate
            solver_name: 'euler', 'midpoint', 'rk4', or 'dopri5'
            n_steps: Number of integration steps (fixed-step solvers)
            return_trajectory: If True, return full trajectory
            
        Returns:
            Generated samples, shape (n_samples, data_dim)
        """
        from solvers import EulerSolver, MidpointSolver, RK4Solver, DormandPrinceSolver
        
        # Select solver
        solver_map = {
            "euler": EulerSolver,
            "midpoint": MidpointSolver,
            "rk4": RK4Solver,
            "dopri5": lambda: DormandPrinceSolver(atol=1e-5, rtol=1e-5),
        }
        
        if solver_name == "dopri5":
            solver = DormandPrinceSolver(atol=1e-5, rtol=1e-5)
        else:
            solver = solver_map[solver_name]()
        
        # Sample initial noise
        z0 = torch.randn(n_samples, self.data_dim, device=self.device)
        
        # Define the ODE RHS using the trained model
        self.model.eval()
        
        def ode_func(t: float, y: torch.Tensor) -> torch.Tensor:
            """Wraps the neural network for the ODE solver."""
            t_tensor = torch.full((y.shape[0],), t, device=y.device)
            return self.model(y, t_tensor)
        
        # Integrate from t=0 (noise) to t=1 (data)
        result = solver.solve(ode_func, z0, t_span=(0.0, 1.0), n_steps=n_steps)
        
        if return_trajectory:
            return result
        
        # Return final state (generated samples)
        return result.ys[-1]
