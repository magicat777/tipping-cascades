"""
Energy-Constrained SDE Solvers

Extends PyCascades' semi-implicit Euler-Maruyama solver with energy
tracking and optional energy constraints.

Key Features:
-------------
1. Extended state integration: simultaneously evolves x and E
2. Energy bounds enforcement: prevents unphysical energy values
3. Conservation monitoring: tracks energy budget closure
4. Lévy stable noise support: inherited from PyCascades

Solver Types:
-------------
1. energy_constrained_euler_maruyama: Basic extended state solver
2. energy_conserving_euler_maruyama: Enforces energy conservation
3. adaptive_energy_solver: Adjusts dt to maintain energy accuracy

Mathematical Framework:
-----------------------
Extended state: y = [x_0, ..., x_{n-1}, E_0, ..., E_{n-1}]

Dynamics: dy/dt = f_extended(y, t) + noise (on x only)

Semi-implicit update:
    y_pred = y + f(y, t) * dt + σ * dW
    y_new = y_pred (with optional projection to constraints)

References:
-----------
- PyCascades: evolve_sde.semi_impl_euler_maruyama_alphastable_sde
- Kloeden & Platen: Numerical SDEs
- Milstein methods for improved accuracy
"""

import numpy as np
from typing import Callable, Tuple, Optional, List, Dict, Any
from dataclasses import dataclass, field
import warnings

# Try to import numba for JIT compilation
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        """Dummy decorator when numba not available."""
        def decorator(func):
            return func
        return decorator

# Try to import scipy for Lévy stable distributions
try:
    from scipy.stats import levy_stable
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class SolverResult:
    """
    Container for solver output.

    Attributes
    ----------
    t : ndarray
        Time points
    y : ndarray
        Extended state trajectory (n_times, 2*n_elements)
    x : ndarray
        State trajectory (n_times, n_elements)
    E : ndarray
        Energy trajectory (n_times, n_elements)
    energy_violations : list
        List of energy constraint violations (if checking enabled)
    diagnostics : dict
        Additional solver diagnostics
    """
    t: np.ndarray
    y: np.ndarray
    x: np.ndarray
    E: np.ndarray
    energy_violations: List[Tuple[float, float]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def generate_gaussian_noise(
    dt: float,
    n_elements: int,
    sigma: np.ndarray
) -> np.ndarray:
    """
    Generate Gaussian noise increments.

    Parameters
    ----------
    dt : float
        Time step
    n_elements : int
        Number of state variables
    sigma : array-like
        Noise amplitudes for each element

    Returns
    -------
    dW : ndarray
        Noise increments
    """
    dW = np.random.randn(n_elements)
    return sigma * dW * np.sqrt(dt)


def generate_levy_noise(
    dt: float,
    n_elements: int,
    sigma: np.ndarray,
    alpha: np.ndarray,
    beta: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate Lévy stable noise increments.

    Parameters
    ----------
    dt : float
        Time step
    n_elements : int
        Number of state variables
    sigma : array-like
        Scale parameters
    alpha : array-like
        Stability parameters (0 < α ≤ 2)
    beta : array-like, optional
        Skewness parameters (default 0)

    Returns
    -------
    dL : ndarray
        Lévy noise increments
    """
    if not SCIPY_AVAILABLE:
        warnings.warn("scipy not available, using Gaussian noise")
        return generate_gaussian_noise(dt, n_elements, sigma)

    if beta is None:
        beta = np.zeros(n_elements)

    dL = np.zeros(n_elements)
    for i in range(n_elements):
        # Lévy stable distribution
        dL[i] = levy_stable.rvs(alpha[i], beta[i], scale=sigma[i])

    # Scale by dt^(1/α) for proper time scaling
    for i in range(n_elements):
        dL[i] *= dt ** (1.0 / alpha[i])

    return dL


def energy_constrained_euler_maruyama(
    f_extended: Callable[[np.ndarray, float], np.ndarray],
    y0: np.ndarray,
    t_span: Tuple[float, float],
    dt: float,
    sigma: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    energy_bounds: Optional[Tuple[float, float]] = None,
    conservation_tol: Optional[float] = None,
    callback: Optional[Callable[[np.ndarray, float], None]] = None
) -> SolverResult:
    """
    Semi-implicit Euler-Maruyama with energy tracking.

    Integrates the extended state y = [x, E] where noise is applied
    only to state variables x, not to energies E.

    Parameters
    ----------
    f_extended : callable
        Extended dynamics function: f(y, t) -> dy/dt
    y0 : array-like
        Initial extended state [x_0, ..., E_0, ...]
    t_span : tuple
        (t_start, t_end)
    dt : float
        Time step
    sigma : array-like
        Noise amplitudes for state variables
    alpha : array-like, optional
        Lévy stability parameters (default: Gaussian, α=2)
    energy_bounds : tuple, optional
        (E_min, E_max) bounds for energy clipping
    conservation_tol : float, optional
        Tolerance for energy conservation checking
    callback : callable, optional
        Function called each timestep: callback(y, t)

    Returns
    -------
    SolverResult
        Solver output with trajectories and diagnostics
    """
    n_total = len(y0)
    n_elements = n_total // 2

    # Ensure sigma has correct shape
    sigma = np.atleast_1d(sigma)
    if len(sigma) == 1:
        sigma = np.repeat(sigma, n_elements)

    # Default to Gaussian noise
    if alpha is None:
        alpha = 2.0 * np.ones(n_elements)
    else:
        alpha = np.atleast_1d(alpha)
        if len(alpha) == 1:
            alpha = np.repeat(alpha, n_elements)

    # Time array
    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)

    # Output arrays
    y = np.zeros((n_steps, n_total))
    y[0] = y0

    # Tracking
    energy_violations = []

    # Main integration loop
    for i in range(1, n_steps):
        # Current state
        y_curr = y[i-1]

        # Deterministic step
        dydt = f_extended(y_curr, t[i-1])
        y_det = y_curr + dydt * dt

        # Stochastic step (only on state variables, not energy)
        use_levy = np.any(alpha < 2.0)
        if use_levy:
            noise = generate_levy_noise(dt, n_elements, sigma, alpha)
        else:
            noise = generate_gaussian_noise(dt, n_elements, sigma)

        # Apply noise only to state variables (first n_elements)
        y_pred = y_det.copy()
        y_pred[:n_elements] += noise

        # State bounds to prevent numerical overflow (Lévy noise can produce extreme jumps)
        y_pred[:n_elements] = np.clip(y_pred[:n_elements], -10.0, 10.0)

        # Energy constraint enforcement (default bounds if not specified)
        if energy_bounds is not None:
            E_min, E_max = energy_bounds
        else:
            # Default: keep energy in reasonable range
            E_min, E_max = -100.0, 100.0
        y_pred[n_elements:] = np.clip(y_pred[n_elements:], E_min, E_max)

        # Conservation check
        if conservation_tol is not None:
            E_before = y_curr[n_elements:].sum()
            E_after = y_pred[n_elements:].sum()
            delta_E = E_after - E_before
            if abs(delta_E) > conservation_tol:
                energy_violations.append((t[i], delta_E))

        # Store result
        y[i] = y_pred

        # Callback
        if callback is not None:
            callback(y_pred, t[i])

    # Package results
    x = y[:, :n_elements]
    E = y[:, n_elements:]

    diagnostics = {
        'n_steps': n_steps,
        'dt': dt,
        'noise_type': 'levy' if use_levy else 'gaussian',
        'n_energy_violations': len(energy_violations)
    }

    return SolverResult(
        t=t, y=y, x=x, E=E,
        energy_violations=energy_violations,
        diagnostics=diagnostics
    )


def energy_conserving_euler_maruyama(
    f_extended: Callable[[np.ndarray, float], np.ndarray],
    y0: np.ndarray,
    t_span: Tuple[float, float],
    dt: float,
    sigma: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    external_power: Optional[Callable[[float], float]] = None,
    dissipation_func: Optional[Callable[[np.ndarray], float]] = None
) -> SolverResult:
    """
    Energy-conserving Euler-Maruyama solver.

    Adjusts energy evolution to maintain energy budget:
        dE_total/dt = P_external - D_total

    Parameters
    ----------
    f_extended : callable
        Extended dynamics function
    y0 : array-like
        Initial state
    t_span : tuple
        (t_start, t_end)
    dt : float
        Time step
    sigma : array-like
        Noise amplitudes
    alpha : array-like, optional
        Lévy stability parameters
    external_power : callable, optional
        P(t) -> external power input
    dissipation_func : callable, optional
        D(y) -> total dissipation rate

    Returns
    -------
    SolverResult
        Solver output
    """
    n_total = len(y0)
    n_elements = n_total // 2

    sigma = np.atleast_1d(sigma)
    if len(sigma) == 1:
        sigma = np.repeat(sigma, n_elements)

    if alpha is None:
        alpha = 2.0 * np.ones(n_elements)
    else:
        alpha = np.atleast_1d(alpha)
        if len(alpha) == 1:
            alpha = np.repeat(alpha, n_elements)

    t_start, t_end = t_span
    t = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t)

    y = np.zeros((n_steps, n_total))
    y[0] = y0

    energy_balance = []  # Track energy budget

    for i in range(1, n_steps):
        y_curr = y[i-1]

        # Standard dynamics
        dydt = f_extended(y_curr, t[i-1])
        y_det = y_curr + dydt * dt

        # Noise on state only
        use_levy = np.any(alpha < 2.0)
        if use_levy:
            noise = generate_levy_noise(dt, n_elements, sigma, alpha)
        else:
            noise = generate_gaussian_noise(dt, n_elements, sigma)

        y_pred = y_det.copy()
        y_pred[:n_elements] += noise

        # Energy conservation correction
        E_target = y_curr[n_elements:].sum()  # Start with previous total

        # Add external power
        if external_power is not None:
            E_target += external_power(t[i]) * dt

        # Subtract dissipation
        if dissipation_func is not None:
            E_target -= dissipation_func(y_curr) * dt

        # Current predicted total
        E_pred = y_pred[n_elements:].sum()

        # Correction factor
        if E_pred != 0:
            correction = E_target / E_pred
            y_pred[n_elements:] *= correction

        energy_balance.append({
            't': t[i],
            'E_target': E_target,
            'E_uncorrected': E_pred,
            'correction': correction if E_pred != 0 else 1.0
        })

        y[i] = y_pred

    x = y[:, :n_elements]
    E = y[:, n_elements:]

    diagnostics = {
        'n_steps': n_steps,
        'dt': dt,
        'energy_balance': energy_balance,
        'max_correction': max(abs(eb['correction'] - 1.0) for eb in energy_balance)
    }

    return SolverResult(t=t, y=y, x=x, E=E, diagnostics=diagnostics)


def run_ensemble(
    network,
    n_runs: int,
    duration: float,
    dt: float,
    sigma: float,
    alpha: float = 2.0,
    x0: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    progress: bool = True
) -> List[SolverResult]:
    """
    Run ensemble of simulations with different random seeds.

    Parameters
    ----------
    network : EnergyConstrainedNetwork
        Network to simulate
    n_runs : int
        Number of ensemble members
    duration : float
        Simulation duration
    dt : float
        Time step
    sigma : float
        Noise amplitude
    alpha : float
        Lévy stability parameter
    x0 : array-like, optional
        Initial states
    seed : int, optional
        Base random seed
    progress : bool
        Show progress indicator

    Returns
    -------
    list of SolverResult
        Ensemble results
    """
    results = []

    n_elements = network.n_elements
    sigma_arr = sigma * np.ones(n_elements)
    alpha_arr = alpha * np.ones(n_elements)

    y0 = network.get_initial_state(x0)

    for i in range(n_runs):
        if seed is not None:
            np.random.seed(seed + i)

        if progress and (i + 1) % 10 == 0:
            print(f"  Run {i+1}/{n_runs}")

        result = energy_constrained_euler_maruyama(
            f_extended=network.f_extended,
            y0=y0,
            t_span=(0, duration),
            dt=dt,
            sigma=sigma_arr,
            alpha=alpha_arr
        )
        results.append(result)

    return results


def compute_ensemble_statistics(
    results: List[SolverResult],
    burn_in: float = 0.1
) -> Dict[str, Any]:
    """
    Compute statistics across ensemble.

    Parameters
    ----------
    results : list of SolverResult
        Ensemble results
    burn_in : float
        Fraction of trajectory to discard

    Returns
    -------
    dict
        Ensemble statistics
    """
    n_runs = len(results)
    n_times = len(results[0].t)
    n_elements = results[0].x.shape[1]

    burn_idx = int(burn_in * n_times)

    # Extract post-burn-in data
    x_all = np.array([r.x[burn_idx:] for r in results])  # (n_runs, n_times, n_elem)
    E_all = np.array([r.E[burn_idx:] for r in results])

    # State statistics
    x_mean = np.mean(x_all, axis=(0, 1))  # Mean state per element
    x_std = np.std(x_all, axis=(0, 1))
    time_tipped = np.mean(x_all > 0, axis=(0, 1))  # Fraction of time tipped

    # Energy statistics
    E_mean = np.mean(E_all, axis=(0, 1))
    E_std = np.std(E_all, axis=(0, 1))
    E_total_mean = np.mean(np.sum(E_all, axis=2))

    # Cross-correlations
    correlations = {}
    for i in range(n_elements):
        for j in range(i+1, n_elements):
            corr_runs = []
            for r in range(n_runs):
                corr = np.corrcoef(x_all[r, :, i], x_all[r, :, j])[0, 1]
                if not np.isnan(corr):
                    corr_runs.append(corr)
            if corr_runs:
                correlations[f'{i}-{j}'] = np.mean(corr_runs)

    return {
        'x_mean': x_mean,
        'x_std': x_std,
        'time_tipped': time_tipped,
        'E_mean': E_mean,
        'E_std': E_std,
        'E_total_mean': E_total_mean,
        'correlations': correlations,
        'n_runs': n_runs
    }
