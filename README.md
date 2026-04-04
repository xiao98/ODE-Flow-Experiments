# ODE-Flow-Experiments

**From Euler to Dormand-Prince: ODE Solvers for Flow Matching Generative Models**

From-scratch PyTorch implementations of four ODE solvers (Euler, Midpoint, RK4, Dormand-Prince 5(4)) applied to Conditional Flow Matching. Includes convergence verification, stability analysis, Jacobian spectrum measurements, MNIST generation, and quantitative NFE-quality benchmarks.

An accompanying paper is available in the [`paper/`](paper/) directory.

## Project Structure

```
ODE-Flow-Experiments/
├── solvers/
│   ├── base.py            # Abstract ODE solver base class
│   ├── fixed_step.py      # Euler, Midpoint, RK4 (from scratch)
│   └── adaptive.py        # Dormand-Prince 5(4) adaptive step size
├── models/
│   ├── vector_field.py    # Time-conditioned MLP v_θ(x, t)
│   └── flow_matching.py   # Conditional Flow Matching trainer
├── experiments/
│   ├── convergence.py     # Convergence order verification
│   ├── stability.py       # Numerical stability analysis
│   └── flow_demo.py       # End-to-end Flow Matching demo
├── tests/
│   └── test_solvers.py    # Unit tests
├── paper/
│   ├── main.tex           # LaTeX source
│   ├── main.pdf           # Compiled paper (21 pages)
│   ├── references.bib     # Bibliography
│   └── figures/           # Paper figures
├── report/
│   └── technical_report.md # Detailed technical report
└── results/               # Generated experiment figures
```

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run tests
```bash
python -m pytest tests/ -v
```

### Run experiments
```bash
# Convergence order verification
python experiments/convergence.py

# Numerical stability analysis
python experiments/stability.py

# End-to-end Flow Matching demo (2D)
python experiments/flow_demo.py

# Quantitative analysis: Pareto frontier, Jacobian spectrum, ablations
python experiments/advanced_analysis.py

# MNIST Flow Matching (64D PCA latent)
python experiments/mnist_demo.py
```

## ODE Solvers

| Method | Order | NFE/step | Local Truncation Error | Stability |
|--------|-------|----------|----------------------|-----------|
| Euler | 1 | 1 | O(h^2) | Small region |
| Midpoint (RK2) | 2 | 2 | O(h^3) | Moderate |
| RK4 | 4 | 4 | O(h^5) | Large |
| Dormand-Prince 5(4) | 5 | ~6 | Adaptive | Auto-controlled |

All solvers are implemented from scratch in PyTorch (~1,800 lines), without relying on `scipy` or `torchdiffeq`.

## Conditional Flow Matching

The project implements Conditional Flow Matching (CFM) with Optimal Transport probability paths:

- **OT path**: `x_t = (1-t) * x_0 + t * x_1`, where `x_0 ~ N(0,I)`, `x_1 ~ p_data`
- **Target velocity**: `u_t = x_1 - x_0`
- **Training loss**: `L(θ) = E[||v_θ(x_t, t) - u_t||^2]`
- **Sampling**: Solve `dz/dt = v_θ(z, t)` from `t=0` to `t=1` using any ODE solver

## Key Results

- **NFE efficiency**: RK4 at 80 NFE matches Euler at 200 NFE (2.5x cost reduction), confirmed by sliced Wasserstein distance on the Pareto frontier.
- **Adaptive stepping**: DOPRI5 concentrates steps near t=1 where the Jacobian condition number spikes — confirmed by eigenvalue measurements.
- **Model maturity matters**: The solver quality gap is *widest* for undertrained/small models, and narrows as the model converges.
- **MNIST**: Same solver ordering holds on 64D PCA-compressed MNIST digits.

### Generated Figures

After running all experiments, `results/` contains:
- `pareto_moons.png`, `pareto_circles.png` — NFE vs quality Pareto frontiers
- `jacobian_spectrum.png` — Jacobian eigenvalue spectrum along trajectory
- `dopri5_stepsize.png` — Adaptive step size distribution
- `ablation_network_moons.png` — Network capacity ablation
- `ablation_training_moons.png` — Training duration ablation
- `mnist_pareto.png`, `mnist_solver_comparison.png` — MNIST results
- `convergence_*.png`, `stability_*.png`, `solver_comparison_*.png`, `trajectories_*.png` — Core experiments

## Paper

The full paper is available at [`paper/main.pdf`](paper/main.pdf). To recompile:
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## References

- Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
- Tong et al., "Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport" (TMLR 2024)
- Chen et al., "Neural Ordinary Differential Equations" (NeurIPS 2018)
- Dormand & Prince, "A Family of Embedded Runge-Kutta Formulae" (1980)
- Hairer, Norsett & Wanner, "Solving Ordinary Differential Equations I" (Springer, 1993)

## License

This project is released for educational and research purposes.
