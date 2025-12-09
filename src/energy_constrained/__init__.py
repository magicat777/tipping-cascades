"""
Energy-Constrained Tipping Cascades Module

This module extends the PyCascades framework with thermodynamic constraints
to investigate how energy flow and entropy production affect tipping cascade
dynamics.

Key components:
- elements: Energy-tracking tipping elements (EnergyConstrainedCusp)
- couplings: Energy-based coupling functions
- network: Dissipative network with energy balance tracking
- solvers: Energy-constrained SDE integrators
- analysis: Entropy production and energy flow analysis tools
- dask_utils: Distributed computing support for large-scale experiments
"""

from .elements import EnergyConstrainedElement, EnergyConstrainedCusp
from .couplings import EnergyCoupling, GradientDrivenCoupling, AsymmetricEnergyCoupling
from .network import EnergyConstrainedNetwork
from .solvers import energy_constrained_euler_maruyama, run_ensemble
from .analysis import EnergyAnalyzer

# Dask utilities (optional, graceful degradation if Dask not installed)
try:
    from .dask_utils import (
        get_dask_client,
        run_ensemble_parallel,
        analyze_ensemble_parallel,
        run_parameter_sweep_parallel,
        results_to_solver_results
    )
    DASK_SUPPORT = True
except ImportError:
    DASK_SUPPORT = False
    # Provide stub functions that fall back to serial
    def get_dask_client(*args, **kwargs):
        return None

    def run_ensemble_parallel(network, n_runs, duration, dt, sigma, alpha=2.0, **kwargs):
        from .solvers import run_ensemble
        results = run_ensemble(network, n_runs, duration, dt, sigma, alpha, **kwargs)
        return [
            {'run_idx': i, 't': r.t, 'x': r.x, 'E': r.E, 'y': r.y,
             'diagnostics': r.diagnostics}
            for i, r in enumerate(results)
        ]

    analyze_ensemble_parallel = None
    run_parameter_sweep_parallel = None
    results_to_solver_results = None

__all__ = [
    # Core classes
    'EnergyConstrainedElement',
    'EnergyConstrainedCusp',
    'EnergyCoupling',
    'GradientDrivenCoupling',
    'AsymmetricEnergyCoupling',
    'EnergyConstrainedNetwork',
    # Solvers
    'energy_constrained_euler_maruyama',
    'run_ensemble',
    # Analysis
    'EnergyAnalyzer',
    # Dask utilities
    'get_dask_client',
    'run_ensemble_parallel',
    'analyze_ensemble_parallel',
    'run_parameter_sweep_parallel',
    'results_to_solver_results',
    'DASK_SUPPORT',
]

__version__ = '0.1.0'
