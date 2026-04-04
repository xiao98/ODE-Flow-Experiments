"""
Fixed-Step ODE Solvers
======================
Hand-written implementations of classical fixed-step methods.
Each method is derived from Taylor expansion of the solution.

Key concepts for interviews:
- Euler: 1st-order, simplest possible method, LTE = O(h²)
- Midpoint: 2nd-order, evaluates slope at midpoint, LTE = O(h³)  
- RK4: 4th-order, the "workhorse" of numerical integration, LTE = O(h⁵)

Notation:
    h  = step size
    f  = dy/dt = f(t, y)
    LTE = Local Truncation Error
    GTE = Global Truncation Error (accumulated over all steps)
"""

import torch
from typing import Callable, Tuple
from .base import ODESolver, ODESolveResult


class EulerSolver(ODESolver):
    """Forward Euler Method (Explicit Euler).
    
    The simplest numerical integrator. Derived from the first-order
    Taylor expansion:
        y(t + h) = y(t) + h·y'(t) + O(h²)
                 = y(t) + h·f(t, y(t)) + O(h²)
    
    Update rule:
        y_{n+1} = y_n + h · f(t_n, y_n)
    
    Properties:
        - Order: 1 (GTE = O(h), LTE = O(h²))
        - 1 function evaluation per step
        - Stability region: circle of radius 1 centered at (-1, 0)
        - Simple but requires small h for accuracy
    
    In generative models:
        DDIM sampling with η=0 is essentially Euler integration
        of the probability flow ODE.
    """
    
    @property
    def order(self) -> int:
        return 1
    
    @property
    def name(self) -> str:
        return "Euler"
    
    def _step(
        self,
        f: Callable[[float, torch.Tensor], torch.Tensor],
        t: float,
        y: torch.Tensor,
        h: float,
    ) -> Tuple[torch.Tensor, int]:
        """Single Euler step.
        
        Args:
            f: RHS function
            t: Current time
            y: Current state
            h: Step size
            
        Returns:
            (y_next, n_function_evals)
        """
        # k1 = f(t_n, y_n) — slope at current point
        k1 = f(t, y)
        y_next = y + h * k1
        return y_next, 1
    
    def solve(
        self,
        f: Callable[[float, torch.Tensor], torch.Tensor],
        y0: torch.Tensor,
        t_span: Tuple[float, float] = (0.0, 1.0),
        n_steps: int = 100,
        **kwargs,
    ) -> ODESolveResult:
        ts = self._make_time_grid(t_span, n_steps, y0.device)
        h = (t_span[1] - t_span[0]) / n_steps
        
        ys = [y0.clone()]
        y = y0.clone()
        n_fe = 0
        
        for i in range(n_steps):
            y, fe = self._step(f, ts[i].item(), y, h)
            n_fe += fe
            ys.append(y.clone())
        
        return ODESolveResult(
            ts=ts,
            ys=torch.stack(ys),
            n_fe=n_fe,
        )


class MidpointSolver(ODESolver):
    """Explicit Midpoint Method (Modified Euler / RK2).
    
    Uses the slope at the midpoint of the interval for better accuracy.
    Derived from second-order Taylor expansion.
    
    Step 1: k1 = f(t_n, y_n)                    — slope at start
    Step 2: k2 = f(t_n + h/2, y_n + h/2 · k1)  — slope at midpoint
    Update: y_{n+1} = y_n + h · k2
    
    Butcher tableau:
        0   |
        1/2 | 1/2
        ----|------
            | 0   1
    
    Properties:
        - Order: 2 (GTE = O(h²), LTE = O(h³))
        - 2 function evaluations per step
        - Larger stability region than Euler
    """
    
    @property
    def order(self) -> int:
        return 2
    
    @property
    def name(self) -> str:
        return "Midpoint"
    
    def _step(
        self,
        f: Callable[[float, torch.Tensor], torch.Tensor],
        t: float,
        y: torch.Tensor,
        h: float,
    ) -> Tuple[torch.Tensor, int]:
        """Single Midpoint step."""
        k1 = f(t, y)
        k2 = f(t + h / 2, y + (h / 2) * k1)
        y_next = y + h * k2
        return y_next, 2
    
    def solve(
        self,
        f: Callable[[float, torch.Tensor], torch.Tensor],
        y0: torch.Tensor,
        t_span: Tuple[float, float] = (0.0, 1.0),
        n_steps: int = 100,
        **kwargs,
    ) -> ODESolveResult:
        ts = self._make_time_grid(t_span, n_steps, y0.device)
        h = (t_span[1] - t_span[0]) / n_steps
        
        ys = [y0.clone()]
        y = y0.clone()
        n_fe = 0
        
        for i in range(n_steps):
            y, fe = self._step(f, ts[i].item(), y, h)
            n_fe += fe
            ys.append(y.clone())
        
        return ODESolveResult(
            ts=ts,
            ys=torch.stack(ys),
            n_fe=n_fe,
        )


class RK4Solver(ODESolver):
    """Classical 4th-Order Runge-Kutta Method.
    
    The "gold standard" of fixed-step ODE solvers. Achieves 4th-order
    accuracy with only 4 function evaluations per step.
    
    k1 = f(t_n,       y_n)
    k2 = f(t_n + h/2, y_n + h/2 · k1)
    k3 = f(t_n + h/2, y_n + h/2 · k2)
    k4 = f(t_n + h,   y_n + h   · k3)
    
    y_{n+1} = y_n + (h/6) · (k1 + 2·k2 + 2·k3 + k4)
    
    Butcher tableau:
        0   |
        1/2 | 1/2
        1/2 | 0    1/2
        1   | 0    0    1
        ----|------------------
            | 1/6  1/3  1/3  1/6
    
    Properties:
        - Order: 4 (GTE = O(h⁴), LTE = O(h⁵))
        - 4 function evaluations per step
        - Excellent accuracy-to-cost ratio
        - Widely used in practice for non-stiff problems
    
    In generative models:
        RK4 provides high-quality ODE sampling with moderate NFE.
        For a typical 1000-dim latent space, RK4 with 50 steps
        often matches adaptive solvers that use ~100 NFE.
    """
    
    @property
    def order(self) -> int:
        return 4
    
    @property
    def name(self) -> str:
        return "RK4"
    
    def _step(
        self,
        f: Callable[[float, torch.Tensor], torch.Tensor],
        t: float,
        y: torch.Tensor,
        h: float,
    ) -> Tuple[torch.Tensor, int]:
        """Single RK4 step."""
        k1 = f(t, y)
        k2 = f(t + h / 2, y + (h / 2) * k1)
        k3 = f(t + h / 2, y + (h / 2) * k2)
        k4 = f(t + h, y + h * k3)
        
        y_next = y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y_next, 4
    
    def solve(
        self,
        f: Callable[[float, torch.Tensor], torch.Tensor],
        y0: torch.Tensor,
        t_span: Tuple[float, float] = (0.0, 1.0),
        n_steps: int = 100,
        **kwargs,
    ) -> ODESolveResult:
        ts = self._make_time_grid(t_span, n_steps, y0.device)
        h = (t_span[1] - t_span[0]) / n_steps
        
        ys = [y0.clone()]
        y = y0.clone()
        n_fe = 0
        
        for i in range(n_steps):
            y, fe = self._step(f, ts[i].item(), y, h)
            n_fe += fe
            ys.append(y.clone())
        
        return ODESolveResult(
            ts=ts,
            ys=torch.stack(ys),
            n_fe=n_fe,
        )
