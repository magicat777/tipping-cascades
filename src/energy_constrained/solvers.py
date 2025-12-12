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

        # State bounds with soft reflection to prevent boundary oscillation
        # The cusp potential has stable equilibria at x ≈ ±1, so we use bounds
        # slightly outside this range. When states exceed bounds, we reflect
        # them back rather than hard clamping, which prevents numerical instability.
        x_bound = 2.0  # Just outside bistable region (equilibria at ±1)
        x_states = y_pred[:n_elements]

        # Soft reflection: if |x| > bound, reflect back into valid range
        # This prevents the boundary oscillation that occurs with hard clamping
        for j in range(n_elements):
            if x_states[j] > x_bound:
                # Reflect back: overshoot beyond bound gets folded back
                x_states[j] = x_bound - (x_states[j] - x_bound) * 0.5
            elif x_states[j] < -x_bound:
                x_states[j] = -x_bound - (x_states[j] + x_bound) * 0.5

        # Final hard clamp as safety net (should rarely be needed)
        y_pred[:n_elements] = np.clip(x_states, -3.0, 3.0)

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


@dataclass
class TwoPhaseResult:
    """
    Container for two-phase cascade-then-recovery experiment results.

    Attributes
    ----------
    cascade_result : SolverResult
        Results from cascade phase
    recovery_result : SolverResult
        Results from recovery phase
    t_full : ndarray
        Combined time array
    x_full : ndarray
        Combined state trajectory
    E_full : ndarray
        Combined energy trajectory
    cascade_end_state : ndarray
        State at end of cascade phase
    tipped_at_cascade_end : ndarray
        Boolean array of which elements were tipped at cascade end
    recovered : ndarray
        Boolean array of which tipped elements recovered
    recovery_times : dict
        Time to recovery for each element (NaN if didn't recover)
    metrics : dict
        Summary metrics
    """
    cascade_result: SolverResult
    recovery_result: SolverResult
    t_full: np.ndarray
    x_full: np.ndarray
    E_full: np.ndarray
    cascade_end_state: np.ndarray
    tipped_at_cascade_end: np.ndarray
    recovered: np.ndarray
    recovery_times: Dict[int, float]
    metrics: Dict[str, Any]


def run_two_phase_experiment(
    network,
    cascade_duration: float = 200,
    recovery_duration: float = 800,
    dt: float = 0.5,
    cascade_sigma: float = 0.06,
    cascade_alpha: float = 1.5,
    recovery_sigma: float = 0.02,
    recovery_alpha: float = 2.0,
    recovery_forcing: float = 0.0,
    x0: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> TwoPhaseResult:
    """
    Run two-phase cascade-then-recovery experiment.

    Phase 1 (Cascade): Apply Lévy noise to trigger cascades
    Phase 2 (Recovery): Switch to gentle Gaussian noise, observe recovery

    Parameters
    ----------
    network : EnergyConstrainedNetwork
        Network to simulate
    cascade_duration : float
        Duration of cascade phase (default 200)
    recovery_duration : float
        Duration of recovery phase (default 800)
    dt : float
        Time step (default 0.5)
    cascade_sigma : float
        Noise amplitude during cascade phase (default 0.06)
    cascade_alpha : float
        Lévy stability parameter during cascade (default 1.5)
    recovery_sigma : float
        Noise amplitude during recovery phase (default 0.02)
    recovery_alpha : float
        Stability parameter during recovery - 2.0 = Gaussian (default 2.0)
    recovery_forcing : float
        Constant forcing during recovery phase (default 0.0).
        Negative values push toward stable state (x < 0), simulating
        active restoration intervention. Typical range: -0.1 to -0.5.
    x0 : array-like, optional
        Initial states
    seed : int, optional
        Random seed

    Returns
    -------
    TwoPhaseResult
        Container with results and recovery metrics
    """
    n_elements = network.n_elements

    if seed is not None:
        np.random.seed(seed)

    # Prepare initial state
    y0 = network.get_initial_state(x0)

    # === Phase 1: Cascade ===
    cascade_result = energy_constrained_euler_maruyama(
        f_extended=network.f_extended,
        y0=y0,
        t_span=(0, cascade_duration),
        dt=dt,
        sigma=cascade_sigma * np.ones(n_elements),
        alpha=cascade_alpha * np.ones(n_elements)
    )

    # Record state at cascade end
    cascade_end_state = cascade_result.x[-1].copy()
    cascade_end_y = cascade_result.y[-1].copy()
    tipped_at_cascade_end = cascade_end_state > 0

    # === Phase 2: Recovery ===
    # Continue from cascade end state
    # If recovery_forcing != 0, wrap f_extended to add constant forcing
    if recovery_forcing != 0.0:
        # Create modified dynamics with forcing
        original_f = network.f_extended
        def f_with_forcing(t, y):
            dydt = original_f(t, y)
            # Add forcing to state variables (first n_elements), not energy
            dydt[:n_elements] += recovery_forcing
            return dydt
        recovery_f = f_with_forcing
    else:
        recovery_f = network.f_extended

    recovery_result = energy_constrained_euler_maruyama(
        f_extended=recovery_f,
        y0=cascade_end_y,
        t_span=(0, recovery_duration),
        dt=dt,
        sigma=recovery_sigma * np.ones(n_elements),
        alpha=recovery_alpha * np.ones(n_elements)
    )

    # Combine trajectories
    # Shift recovery time to continue from cascade end
    t_full = np.concatenate([
        cascade_result.t,
        cascade_result.t[-1] + recovery_result.t[1:]  # Skip t=0 of recovery
    ])
    x_full = np.concatenate([
        cascade_result.x,
        recovery_result.x[1:]  # Skip first row (same as cascade end)
    ], axis=0)
    E_full = np.concatenate([
        cascade_result.E,
        recovery_result.E[1:]
    ], axis=0)

    # === Analyze Recovery ===
    recovered = np.zeros(n_elements, dtype=bool)
    recovery_times = {}

    for i in range(n_elements):
        if tipped_at_cascade_end[i]:
            # Find first time this element crossed back to stable (x < 0)
            recovery_traj = recovery_result.x[:, i]
            recovery_crossings = np.where(recovery_traj < 0)[0]

            if len(recovery_crossings) > 0:
                # Find first crossing that persists (to avoid brief excursions)
                for cross_idx in recovery_crossings:
                    # Check if it stays stable for at least 10 time steps
                    if cross_idx + 10 < len(recovery_traj):
                        if np.all(recovery_traj[cross_idx:cross_idx+10] < 0):
                            recovered[i] = True
                            recovery_times[i] = recovery_result.t[cross_idx]
                            break
                    else:
                        # Near end, just check remaining
                        if np.all(recovery_traj[cross_idx:] < 0):
                            recovered[i] = True
                            recovery_times[i] = recovery_result.t[cross_idx]
                            break

            if i not in recovery_times:
                recovery_times[i] = np.nan
        else:
            recovery_times[i] = 0.0  # Wasn't tipped, no recovery needed

    # === Compute Metrics ===
    n_tipped = np.sum(tipped_at_cascade_end)
    n_recovered = np.sum(recovered)

    metrics = {
        'cascade_duration': cascade_duration,
        'recovery_duration': recovery_duration,
        'cascade_sigma': cascade_sigma,
        'cascade_alpha': cascade_alpha,
        'recovery_sigma': recovery_sigma,
        'recovery_alpha': recovery_alpha,
        'recovery_forcing': recovery_forcing,
        'n_elements': n_elements,
        'n_tipped_at_cascade_end': int(n_tipped),
        'pct_tipped_at_cascade_end': float(n_tipped / n_elements * 100),
        'n_recovered': int(n_recovered),
        'recovery_fraction': float(n_recovered / n_tipped) if n_tipped > 0 else np.nan,
        'mean_recovery_time': float(np.nanmean(list(recovery_times.values()))),
        'n_permanent_tips': int(n_tipped - n_recovered),
        'final_pct_tipped': float(np.sum(recovery_result.x[-1] > 0) / n_elements * 100)
    }

    return TwoPhaseResult(
        cascade_result=cascade_result,
        recovery_result=recovery_result,
        t_full=t_full,
        x_full=x_full,
        E_full=E_full,
        cascade_end_state=cascade_end_state,
        tipped_at_cascade_end=tipped_at_cascade_end,
        recovered=recovered,
        recovery_times=recovery_times,
        metrics=metrics
    )


def run_two_phase_ensemble(
    network,
    n_runs: int,
    cascade_duration: float = 200,
    recovery_duration: float = 800,
    dt: float = 0.5,
    cascade_sigma: float = 0.06,
    cascade_alpha: float = 1.5,
    recovery_sigma: float = 0.02,
    recovery_alpha: float = 2.0,
    x0: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    progress: bool = True
) -> List[TwoPhaseResult]:
    """
    Run ensemble of two-phase experiments.

    Parameters
    ----------
    network : EnergyConstrainedNetwork
        Network to simulate
    n_runs : int
        Number of ensemble members
    cascade_duration : float
        Duration of cascade phase
    recovery_duration : float
        Duration of recovery phase
    dt : float
        Time step
    cascade_sigma : float
        Noise amplitude during cascade phase
    cascade_alpha : float
        Lévy stability parameter during cascade
    recovery_sigma : float
        Noise amplitude during recovery phase
    recovery_alpha : float
        Stability parameter during recovery (2.0 = Gaussian)
    x0 : array-like, optional
        Initial states
    seed : int, optional
        Base random seed
    progress : bool
        Show progress indicator

    Returns
    -------
    list of TwoPhaseResult
        Ensemble results
    """
    results = []

    for i in range(n_runs):
        run_seed = seed + i if seed is not None else None

        if progress and (i + 1) % 5 == 0:
            print(f"  Run {i+1}/{n_runs}")

        result = run_two_phase_experiment(
            network=network,
            cascade_duration=cascade_duration,
            recovery_duration=recovery_duration,
            dt=dt,
            cascade_sigma=cascade_sigma,
            cascade_alpha=cascade_alpha,
            recovery_sigma=recovery_sigma,
            recovery_alpha=recovery_alpha,
            x0=x0,
            seed=run_seed
        )
        results.append(result)

    return results


def aggregate_two_phase_results(results: List[TwoPhaseResult]) -> Dict[str, Any]:
    """
    Aggregate metrics across two-phase ensemble.

    Parameters
    ----------
    results : list of TwoPhaseResult
        Ensemble results

    Returns
    -------
    dict
        Aggregated statistics
    """
    n_runs = len(results)

    # Collect metrics across runs
    metrics_list = [r.metrics for r in results]

    # Aggregate
    aggregated = {
        'n_runs': n_runs,
        'mean_pct_tipped': np.mean([m['pct_tipped_at_cascade_end'] for m in metrics_list]),
        'std_pct_tipped': np.std([m['pct_tipped_at_cascade_end'] for m in metrics_list]),
        'mean_recovery_fraction': np.nanmean([m['recovery_fraction'] for m in metrics_list]),
        'std_recovery_fraction': np.nanstd([m['recovery_fraction'] for m in metrics_list]),
        'mean_recovery_time': np.nanmean([m['mean_recovery_time'] for m in metrics_list]),
        'mean_permanent_tips': np.mean([m['n_permanent_tips'] for m in metrics_list]),
        'mean_final_pct_tipped': np.mean([m['final_pct_tipped'] for m in metrics_list]),
    }

    # Element-level recovery statistics
    n_elements = results[0].metrics['n_elements']
    element_recovery_rates = np.zeros(n_elements)

    for i in range(n_elements):
        tipped_count = 0
        recovered_count = 0
        for r in results:
            if r.tipped_at_cascade_end[i]:
                tipped_count += 1
                if r.recovered[i]:
                    recovered_count += 1
        if tipped_count > 0:
            element_recovery_rates[i] = recovered_count / tipped_count
        else:
            element_recovery_rates[i] = np.nan

    aggregated['element_recovery_rates'] = element_recovery_rates
    aggregated['most_vulnerable_elements'] = np.argsort(element_recovery_rates)[:5]

    return aggregated
