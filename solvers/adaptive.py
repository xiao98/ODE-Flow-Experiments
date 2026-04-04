"""
Adaptive-Step ODE Solver: Dormand-Prince 5(4)
==============================================
An embedded Runge-Kutta method that provides automatic step size control
by comparing a 5th-order and 4th-order solution to estimate local error.

This is the method behind MATLAB's ode45 and scipy's RK45.

Key interview points:
1. Why adaptive? — High-dim latent spaces have varying stiffness across
   dimensions; fixed-step either wastes NFE or loses accuracy.
2. Embedded pair: The 5th-order solution advances the state, while the
   difference with the 4th-order solution estimates the local error.
3. Step size control: h_new = h * min(max_factor, max(min_factor, 
   safety * (tol/err)^(1/(p+1))))
"""

import torch
from typing import Callable, Tuple, Optional
from .base import ODESolver, ODESolveResult


class DormandPrinceSolver(ODESolver):
    """Dormand-Prince 5(4) Adaptive Step Size Solver (DOPRI5).
    
    An embedded Runge-Kutta pair that uses 7 stages (with FSAL property)
    to produce both a 5th-order and 4th-order solution. The difference
    gives a local error estimate for step size control.
    
    FSAL (First Same As Last):
        The last stage k7 of step n equals k1 of step n+1,
        so effectively only 6 new evaluations per step.
    
    Butcher tableau coefficients from Dormand & Prince (1980).
    
    Properties:
        - Order: 5 (with 4th-order embedded error estimator)
        - 6 effective function evaluations per step (FSAL)
        - Automatic step size control with error tolerance
        - Most efficient general-purpose solver for non-stiff problems
    """
    
    # --- Dormand-Prince coefficients ---
    # Node positions (c_i)
    C2, C3, C4, C5, C6, C7 = 1/5, 3/10, 4/5, 8/9, 1.0, 1.0
    
    # Runge-Kutta matrix (a_ij) — lower triangular
    A21 = 1/5
    A31, A32 = 3/40, 9/40
    A41, A42, A43 = 44/45, -56/15, 32/9
    A51, A52, A53, A54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    A61, A62, A63, A64, A65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
    A71, A72, A73, A74, A75, A76 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84
    
    # 5th-order weights (b_i) — same as A7x (FSAL)
    B1, B2, B3, B4, B5, B6, B7 = 35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0
    
    # 4th-order weights for error estimation (b*_i)
    B1S, B2S, B3S, B4S, B5S, B6S, B7S = (
        5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40
    )
    
    def __init__(
        self,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        max_steps: int = 10000,
        safety: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
    ):
        """
        Args:
            atol: Absolute error tolerance.
            rtol: Relative error tolerance.
            max_steps: Maximum number of steps to prevent infinite loops.
            safety: Safety factor for step size control (< 1).
            min_factor: Minimum step size reduction factor.
            max_factor: Maximum step size growth factor.
        """
        self.atol = atol
        self.rtol = rtol
        self.max_steps = max_steps
        self.safety = safety
        self.min_factor = min_factor
        self.max_factor = max_factor
    
    @property
    def order(self) -> int:
        return 5
    
    @property
    def name(self) -> str:
        return "DormandPrince5(4)"
    
    def _estimate_initial_step(
        self,
        f: Callable,
        t0: float,
        y0: torch.Tensor,
        t_end: float,
    ) -> float:
        """Estimate a good initial step size using the algorithm from
        Hairer, Nørsett & Wanner, "Solving Ordinary Differential
        Equations I", Section II.4."""
        scale = self.atol + torch.abs(y0) * self.rtol
        f0 = f(t0, y0)
        
        d0 = torch.sqrt(torch.mean((y0 / scale) ** 2)).item()
        d1 = torch.sqrt(torch.mean((f0 / scale) ** 2)).item()
        
        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1
        
        # Ensure h0 doesn't exceed interval
        h0 = min(h0, abs(t_end - t0))
        return h0
    
    def _step(
        self,
        f: Callable,
        t: float,
        y: torch.Tensor,
        h: float,
        k1: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        """Single Dormand-Prince step with error estimation.
        
        Returns:
            (y5, y_err, n_fe, k7_for_fsal)
        """
        if k1 is None:
            k1 = f(t, y)
            n_fe = 1
        else:
            n_fe = 0
        
        k2 = f(t + self.C2 * h, y + h * self.A21 * k1)
        k3 = f(t + self.C3 * h, y + h * (self.A31 * k1 + self.A32 * k2))
        k4 = f(t + self.C4 * h, y + h * (self.A41 * k1 + self.A42 * k2 + self.A43 * k3))
        k5 = f(t + self.C5 * h, y + h * (self.A51 * k1 + self.A52 * k2 + self.A53 * k3 + self.A54 * k4))
        k6 = f(t + self.C6 * h, y + h * (self.A61 * k1 + self.A62 * k2 + self.A63 * k3 + self.A64 * k4 + self.A65 * k5))
        n_fe += 5
        
        # 5th-order solution
        y5 = y + h * (self.B1 * k1 + self.B3 * k3 + self.B4 * k4 + self.B5 * k5 + self.B6 * k6)
        
        # k7 for FSAL (also evaluates f at the new point)
        k7 = f(t + h, y5)
        n_fe += 1
        
        # Error estimate = y5 - y4 (difference between 5th and 4th order solutions)
        y_err = h * (
            (self.B1 - self.B1S) * k1
            + (self.B3 - self.B3S) * k3
            + (self.B4 - self.B4S) * k4
            + (self.B5 - self.B5S) * k5
            + (self.B6 - self.B6S) * k6
            + (self.B7 - self.B7S) * k7
        )
        
        return y5, y_err, n_fe, k7
    
    def _compute_error_norm(
        self, y_err: torch.Tensor, y: torch.Tensor, y_new: torch.Tensor
    ) -> float:
        """Compute a scaled error norm for step size control.
        
        Uses mixed absolute/relative tolerance:
            scale_i = atol + max(|y_i|, |y_new_i|) * rtol
            err_norm = sqrt(mean((y_err / scale)²))
        """
        scale = self.atol + torch.max(torch.abs(y), torch.abs(y_new)) * self.rtol
        return torch.sqrt(torch.mean((y_err / scale) ** 2)).item()
    
    def solve(
        self,
        f: Callable[[float, torch.Tensor], torch.Tensor],
        y0: torch.Tensor,
        t_span: Tuple[float, float] = (0.0, 1.0),
        n_steps: int = 100,  # ignored for adaptive solver
        **kwargs,
    ) -> ODESolveResult:
        """Solve with adaptive step size control.
        
        The n_steps argument is ignored; the solver automatically
        determines the number of steps based on error tolerances.
        """
        t0, t_end = t_span
        t = t0
        y = y0.clone()
        h = self._estimate_initial_step(f, t0, y0, t_end)
        
        ts_list = [t]
        ys_list = [y0.clone()]
        errors_list = []
        total_nfe = 0
        
        k1 = None  # FSAL: reuse last stage
        
        for step in range(self.max_steps):
            if t >= t_end - 1e-12:
                break
            
            # Don't overshoot
            h = min(h, t_end - t)
            
            # Attempt a step
            y_new, y_err, nfe, k7 = self._step(f, t, y, h, k1)
            total_nfe += nfe
            
            # Compute error norm
            err_norm = self._compute_error_norm(y_err, y, y_new)
            
            if err_norm <= 1.0:
                # Step accepted
                t = t + h
                y = y_new
                k1 = k7  # FSAL reuse
                
                ts_list.append(t)
                ys_list.append(y.clone())
                errors_list.append(err_norm)
                
                # Compute new step size
                if err_norm == 0:
                    factor = self.max_factor
                else:
                    factor = min(
                        self.max_factor,
                        max(
                            self.min_factor,
                            self.safety * err_norm ** (-1.0 / (self.order + 1)),
                        ),
                    )
                h = h * factor
            else:
                # Step rejected — reduce h and retry
                factor = max(
                    self.min_factor,
                    self.safety * err_norm ** (-1.0 / (self.order + 1)),
                )
                h = h * factor
                k1 = None  # Cannot reuse FSAL after rejection
        
        return ODESolveResult(
            ts=torch.tensor(ts_list, device=y0.device),
            ys=torch.stack(ys_list),
            n_fe=total_nfe,
            errors=errors_list,
        )
