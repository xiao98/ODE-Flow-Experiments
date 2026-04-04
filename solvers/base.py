"""
ODE Solver Base Class
=====================
Abstract base class for all ODE solvers. Defines the common interface
and trajectory recording logic.

Mathematical formulation:
    Given dy/dt = f(t, y), y(t0) = y0
    Find y(t1) by numerical integration.
"""

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple


@dataclass
class ODESolveResult:
    """Result container for ODE solve operations.
    
    Attributes:
        ts: Time points evaluated at, shape (n_steps+1,)
        ys: Solution trajectory, shape (n_steps+1, *y_shape)
        n_fe: Number of function evaluations (NFE)
        errors: Per-step local error estimates (if available)
    """
    ts: torch.Tensor
    ys: torch.Tensor
    n_fe: int = 0
    errors: Optional[List[float]] = None


class ODESolver(ABC):
    """Abstract base class for ODE solvers.
    
    All solvers implement the Initial Value Problem (IVP):
        dy/dt = f(t, y),  y(t0) = y0
    
    where y can be a tensor of arbitrary shape (enabling high-dimensional
    latent space integration for generative models).
    
    Usage:
        solver = EulerSolver()
        result = solver.solve(f, y0, t_span=(0.0, 1.0), n_steps=100)
    """
    
    @property
    @abstractmethod
    def order(self) -> int:
        """Theoretical convergence order of the solver.
        
        For a method of order p, the local truncation error is O(h^{p+1})
        and the global error is O(h^p).
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the solver."""
        pass
    
    @abstractmethod
    def solve(
        self,
        f: Callable[[float, torch.Tensor], torch.Tensor],
        y0: torch.Tensor,
        t_span: Tuple[float, float],
        n_steps: int = 100,
        **kwargs,
    ) -> ODESolveResult:
        """Solve the ODE initial value problem.
        
        Args:
            f: Right-hand side function dy/dt = f(t, y).
            y0: Initial condition, shape (*y_shape).
            t_span: (t_start, t_end) integration interval.
            n_steps: Number of integration steps (for fixed-step solvers).
            **kwargs: Additional solver-specific parameters.
            
        Returns:
            ODESolveResult with the full trajectory.
        """
        pass
    
    def _make_time_grid(
        self, t_span: Tuple[float, float], n_steps: int, device: torch.device
    ) -> torch.Tensor:
        """Create uniform time grid for fixed-step solvers."""
        t0, t1 = t_span
        return torch.linspace(t0, t1, n_steps + 1, device=device)
    
    def __repr__(self) -> str:
        return f"{self.name}(order={self.order})"
