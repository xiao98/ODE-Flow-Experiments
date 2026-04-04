"""
Unit Tests for ODE Solvers
===========================
Validate correctness and convergence order of all solvers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pytest
from solvers import EulerSolver, MidpointSolver, RK4Solver, DormandPrinceSolver


# === Test Problem: dy/dt = -y, y(0) = 1, y(t) = exp(-t) ===

def exact_exp_decay(t: float) -> float:
    return np.exp(-t)


def f_exp_decay(t: float, y: torch.Tensor) -> torch.Tensor:
    return -y


# === Test Problem 2: dy/dt = cos(t), y(0) = 0, y(t) = sin(t) ===

def f_cos(t: float, y: torch.Tensor) -> torch.Tensor:
    return torch.cos(torch.tensor(t)) * torch.ones_like(y)


class TestEulerSolver:
    def test_basic_solve(self):
        solver = EulerSolver()
        y0 = torch.tensor([1.0])
        result = solver.solve(f_exp_decay, y0, t_span=(0.0, 1.0), n_steps=1000)
        
        assert result.ys.shape == (1001, 1)
        assert result.ts.shape == (1001,)
        assert abs(result.ys[-1].item() - exact_exp_decay(1.0)) < 0.001
    
    def test_order_1(self):
        """Verify Euler has convergence order 1."""
        solver = EulerSolver()
        y0 = torch.tensor([1.0], dtype=torch.float64)
        
        errors = []
        step_counts = [100, 200, 400, 800]
        for n in step_counts:
            result = solver.solve(f_exp_decay, y0, t_span=(0.0, 1.0), n_steps=n)
            error = abs(result.ys[-1].item() - exact_exp_decay(1.0))
            errors.append(error)
        
        # Compute ratios: if order=p, ratio ≈ 2^p
        ratios = [errors[i] / errors[i+1] for i in range(len(errors)-1)]
        for r in ratios:
            assert 1.8 < r < 2.2, f"Euler ratio {r:.3f} not close to 2^1"
    
    def test_nfe(self):
        solver = EulerSolver()
        y0 = torch.tensor([1.0])
        result = solver.solve(f_exp_decay, y0, t_span=(0.0, 1.0), n_steps=50)
        assert result.n_fe == 50
    
    def test_high_dim(self):
        solver = EulerSolver()
        y0 = torch.ones(100, dtype=torch.float64)
        result = solver.solve(f_exp_decay, y0, t_span=(0.0, 1.0), n_steps=500)
        exact = torch.ones(100, dtype=torch.float64) * exact_exp_decay(1.0)
        assert torch.allclose(result.ys[-1], exact, atol=0.01)


class TestMidpointSolver:
    def test_basic_solve(self):
        solver = MidpointSolver()
        y0 = torch.tensor([1.0])
        result = solver.solve(f_exp_decay, y0, t_span=(0.0, 1.0), n_steps=100)
        assert abs(result.ys[-1].item() - exact_exp_decay(1.0)) < 0.001
    
    def test_order_2(self):
        """Verify Midpoint has convergence order 2."""
        solver = MidpointSolver()
        y0 = torch.tensor([1.0], dtype=torch.float64)
        
        errors = []
        for n in [50, 100, 200, 400]:
            result = solver.solve(f_exp_decay, y0, t_span=(0.0, 1.0), n_steps=n)
            error = abs(result.ys[-1].item() - exact_exp_decay(1.0))
            errors.append(error)
        
        ratios = [errors[i] / errors[i+1] for i in range(len(errors)-1)]
        for r in ratios:
            assert 3.5 < r < 4.5, f"Midpoint ratio {r:.3f} not close to 2^2=4"
    
    def test_nfe(self):
        solver = MidpointSolver()
        y0 = torch.tensor([1.0])
        result = solver.solve(f_exp_decay, y0, t_span=(0.0, 1.0), n_steps=50)
        assert result.n_fe == 100  # 2 evals per step


class TestRK4Solver:
    def test_basic_solve(self):
        solver = RK4Solver()
        y0 = torch.tensor([1.0])
        result = solver.solve(f_exp_decay, y0, t_span=(0.0, 1.0), n_steps=10)
        # RK4 should be very accurate even with few steps
        assert abs(result.ys[-1].item() - exact_exp_decay(1.0)) < 1e-6
    
    def test_order_4(self):
        """Verify RK4 has convergence order 4."""
        solver = RK4Solver()
        y0 = torch.tensor([1.0], dtype=torch.float64)
        
        errors = []
        for n in [10, 20, 40, 80]:
            result = solver.solve(f_exp_decay, y0, t_span=(0.0, 1.0), n_steps=n)
            error = abs(result.ys[-1].item() - exact_exp_decay(1.0))
            errors.append(error)
        
        ratios = [errors[i] / errors[i+1] for i in range(len(errors)-1)]
        for r in ratios:
            assert 14 < r < 18, f"RK4 ratio {r:.3f} not close to 2^4=16"
    
    def test_nfe(self):
        solver = RK4Solver()
        y0 = torch.tensor([1.0])
        result = solver.solve(f_exp_decay, y0, t_span=(0.0, 1.0), n_steps=50)
        assert result.n_fe == 200  # 4 evals per step


class TestDormandPrinceSolver:
    def test_basic_solve(self):
        solver = DormandPrinceSolver(atol=1e-8, rtol=1e-8)
        y0 = torch.tensor([1.0], dtype=torch.float64)
        result = solver.solve(f_exp_decay, y0, t_span=(0.0, 1.0))
        
        assert abs(result.ys[-1].item() - exact_exp_decay(1.0)) < 1e-7
    
    def test_adaptive_steps(self):
        """Adaptive solver should use fewer steps for smooth problems."""
        solver = DormandPrinceSolver(atol=1e-6, rtol=1e-6)
        y0 = torch.tensor([1.0], dtype=torch.float64)
        result = solver.solve(f_exp_decay, y0, t_span=(0.0, 1.0))
        
        # For this smooth problem, should need relatively few steps
        n_steps = len(result.ts) - 1
        assert n_steps < 100, f"Too many steps: {n_steps}"
        assert n_steps > 1, f"Too few steps: {n_steps}"
    
    def test_high_dim(self):
        solver = DormandPrinceSolver(atol=1e-6, rtol=1e-6)
        y0 = torch.ones(50, dtype=torch.float64)
        result = solver.solve(f_exp_decay, y0, t_span=(0.0, 1.0))
        exact = torch.ones(50, dtype=torch.float64) * exact_exp_decay(1.0)
        assert torch.allclose(result.ys[-1], exact, atol=1e-5)
    
    def test_tolerance_affects_accuracy(self):
        """Tighter tolerance should give more accurate results."""
        y0 = torch.tensor([1.0], dtype=torch.float64)
        
        solver_loose = DormandPrinceSolver(atol=1e-3, rtol=1e-3)
        solver_tight = DormandPrinceSolver(atol=1e-8, rtol=1e-8)
        
        result_loose = solver_loose.solve(f_exp_decay, y0, t_span=(0.0, 1.0))
        result_tight = solver_tight.solve(f_exp_decay, y0, t_span=(0.0, 1.0))
        
        err_loose = abs(result_loose.ys[-1].item() - exact_exp_decay(1.0))
        err_tight = abs(result_tight.ys[-1].item() - exact_exp_decay(1.0))
        
        assert err_tight < err_loose


class TestSolverProperties:
    def test_order_values(self):
        assert EulerSolver().order == 1
        assert MidpointSolver().order == 2
        assert RK4Solver().order == 4
        assert DormandPrinceSolver().order == 5
    
    def test_names(self):
        assert EulerSolver().name == "Euler"
        assert MidpointSolver().name == "Midpoint"
        assert RK4Solver().name == "RK4"
        assert "Dormand" in DormandPrinceSolver().name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
