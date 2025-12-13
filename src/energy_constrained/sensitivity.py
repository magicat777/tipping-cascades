"""
Sobol Sensitivity Analysis for Energy-Constrained Tipping Cascades

Implements global sensitivity analysis to identify which parameters most
influence cascade outcomes. This addresses Objective 5 from the research
project: "Conduct parameter sensitivity analysis to identify energy flow
parameters with greatest influence on cascade risk."

The analysis uses Sobol' indices to decompose variance in model outputs:
- First-order indices (S1): Direct contribution of each parameter
- Total indices (ST): Total contribution including interactions
- Second-order indices (S2): Pairwise interaction effects

Key Parameters Analyzed:
------------------------
Energy parameters:
- E_nominal: Normal operating energy
- E_critical: Critical energy threshold
- energy_sensitivity: How threshold depends on energy
- decay_rate: Energy loss rate when supply disrupted

Coupling parameters:
- base_strength: Coupling strength coefficient
- saturation_type: Energy saturation function

Network parameters:
- n_elements: Number of tipping elements
- connectivity: Network edge density
- inter_layer_coupling: Cross-layer coupling strength

References:
-----------
- Sobol (2001): Global sensitivity indices for nonlinear mathematical models
- SALib documentation: https://salib.readthedocs.io/
- Research project Phase 4: Analysis and Validation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
import warnings

# Try to import SALib
try:
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    warnings.warn("SALib not available. Install with: pip install SALib")


# ========== Problem Definitions ==========

def define_energy_problem() -> Dict[str, Any]:
    """
    Define the parameter space for energy-constrained cascade sensitivity.

    Returns
    -------
    dict
        SALib problem definition with parameter names and bounds
    """
    return {
        'num_vars': 8,
        'names': [
            'E_nominal',         # Normal energy supply
            'E_critical',        # Critical energy threshold
            'energy_sensitivity', # Threshold-energy relationship
            'decay_rate',        # Energy decay rate
            'barrier_height',    # Tipping barrier
            'coupling_strength', # Inter-element coupling
            'noise_amplitude',   # Stochastic forcing
            'alpha_levy'         # Lévy noise stability parameter
        ],
        'bounds': [
            [0.5, 2.0],      # E_nominal
            [0.05, 0.3],     # E_critical (fraction of nominal)
            [0.5, 2.0],      # energy_sensitivity
            [0.01, 0.2],     # decay_rate
            [0.1, 0.5],      # barrier_height
            [0.05, 0.3],     # coupling_strength
            [0.02, 0.1],     # noise_amplitude
            [1.3, 2.0]       # alpha_levy (1.3=heavy Lévy, 2.0=Gaussian)
        ]
    }


def define_three_layer_problem() -> Dict[str, Any]:
    """
    Define parameter space for three-layer network sensitivity.

    Returns
    -------
    dict
        SALib problem definition
    """
    return {
        'num_vars': 12,
        'names': [
            # Energy parameters
            'E_nominal_climate',
            'E_nominal_human',
            'energy_sensitivity',
            'decay_rate',
            # Coupling parameters
            'intra_layer_coupling',
            'climate_biosphere_coupling',
            'biosphere_human_coupling',
            'human_biosphere_feedback',
            # Network parameters
            'n_climate_elements',
            'n_human_settlements',
            # Forcing parameters
            'noise_amplitude',
            'climate_forcing_rate'
        ],
        'bounds': [
            [0.5, 2.0],      # E_nominal_climate
            [0.5, 2.0],      # E_nominal_human
            [0.5, 2.0],      # energy_sensitivity
            [0.01, 0.2],     # decay_rate
            [0.02, 0.15],    # intra_layer_coupling
            [0.02, 0.15],    # climate_biosphere_coupling
            [0.02, 0.15],    # biosphere_human_coupling
            [0.01, 0.1],     # human_biosphere_feedback
            [2, 8],          # n_climate_elements (discrete)
            [3, 10],         # n_human_settlements (discrete)
            [0.02, 0.1],     # noise_amplitude
            [0.0, 0.01]      # climate_forcing_rate
        ]
    }


# ========== Sampling Functions ==========

def generate_samples(
    problem: Dict[str, Any],
    n_samples: int = 1024,
    calc_second_order: bool = True
) -> np.ndarray:
    """
    Generate Saltelli samples for Sobol analysis.

    Parameters
    ----------
    problem : dict
        SALib problem definition
    n_samples : int
        Number of base samples (total = n_samples * (2D + 2) for Sobol)
    calc_second_order : bool
        Whether to calculate second-order indices

    Returns
    -------
    ndarray
        Sample matrix of shape (n_total, n_vars)
    """
    if not SALIB_AVAILABLE:
        raise ImportError("SALib required for Sobol sampling")

    return saltelli.sample(
        problem,
        n_samples,
        calc_second_order=calc_second_order
    )


# ========== Model Evaluation Functions ==========

def evaluate_cascade_model(
    params: np.ndarray,
    problem: Dict[str, Any],
    duration: float = 500.0,
    dt: float = 0.5,
    n_runs: int = 10
) -> Dict[str, float]:
    """
    Evaluate cascade model for a single parameter set.

    Parameters
    ----------
    params : array-like
        Parameter values (one row from sample matrix)
    problem : dict
        Problem definition with parameter names
    duration : float
        Simulation duration
    dt : float
        Time step
    n_runs : int
        Number of ensemble runs to average

    Returns
    -------
    dict
        Model outputs (metrics)
    """
    from .elements import EnergyDependentCusp
    from .network import EnergyConstrainedNetwork
    from .couplings import EnergyMediatedCoupling
    from .solvers import run_two_phase_experiment

    # Extract parameters
    param_dict = {
        name: params[i]
        for i, name in enumerate(problem['names'])
    }

    # Build network with these parameters
    net = EnergyConstrainedNetwork()

    # Add 4 elements (standard Earth system)
    for i in range(4):
        element = EnergyDependentCusp(
            E_nominal=param_dict.get('E_nominal', 1.0),
            E_critical=param_dict.get('E_critical', 0.1),
            energy_sensitivity=param_dict.get('energy_sensitivity', 1.0),
            decay_rate=param_dict.get('decay_rate', 0.1),
            barrier_height=param_dict.get('barrier_height', 0.2)
        )
        net.add_element(f'elem_{i}', element)

    # Add couplings
    coupling_strength = param_dict.get('coupling_strength', 0.1)
    for i in range(4):
        for j in range(4):
            if i != j:
                coupling = EnergyMediatedCoupling(
                    base_strength=coupling_strength,
                    E_reference=param_dict.get('E_nominal', 1.0)
                )
                net.add_coupling(f'elem_{i}', f'elem_{j}', coupling)

    # Run ensemble
    results = {
        'cascade_fraction': [],
        'recovery_fraction': [],
        'total_entropy': [],
        'final_tipped_fraction': []
    }

    sigma = param_dict.get('noise_amplitude', 0.05)
    alpha = param_dict.get('alpha_levy', 1.5)

    for run in range(n_runs):
        try:
            result = run_two_phase_experiment(
                net,
                cascade_duration=duration / 2,
                recovery_duration=duration / 2,
                dt=dt,
                cascade_sigma=sigma,
                recovery_sigma=sigma * 0.8,
                cascade_alpha=alpha,
                recovery_alpha=2.0,  # Gaussian for recovery
                seed=42 + run
            )

            # Extract metrics
            n_elements = net.n_elements
            x_final = result.x[-1, :n_elements]

            results['cascade_fraction'].append(result.metrics['max_tipped'])
            results['recovery_fraction'].append(result.metrics['recovery_fraction'])
            results['total_entropy'].append(result.metrics['total_entropy'])
            results['final_tipped_fraction'].append(np.mean(x_final > 0))

        except Exception as e:
            # If simulation fails, record NaN
            results['cascade_fraction'].append(np.nan)
            results['recovery_fraction'].append(np.nan)
            results['total_entropy'].append(np.nan)
            results['final_tipped_fraction'].append(np.nan)

    # Average over ensemble
    return {
        key: np.nanmean(vals)
        for key, vals in results.items()
    }


def run_sobol_analysis(
    samples: np.ndarray,
    problem: Dict[str, Any],
    output_name: str = 'cascade_fraction',
    parallel: bool = False,
    n_workers: int = 4,
    **eval_kwargs
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Run full Sobol sensitivity analysis.

    Parameters
    ----------
    samples : ndarray
        Sample matrix from generate_samples()
    problem : dict
        Problem definition
    output_name : str
        Which output metric to analyze
    parallel : bool
        Whether to use parallel evaluation
    n_workers : int
        Number of parallel workers
    **eval_kwargs
        Additional arguments for evaluate_cascade_model

    Returns
    -------
    outputs : ndarray
        Model outputs for each sample
    indices : dict
        Sobol indices (S1, S1_conf, ST, ST_conf, S2, S2_conf)
    """
    if not SALIB_AVAILABLE:
        raise ImportError("SALib required for Sobol analysis")

    n_samples = len(samples)
    outputs = np.zeros(n_samples)

    if parallel:
        # Try to use Dask for parallel evaluation
        try:
            from dask import delayed, compute
            from dask.distributed import Client

            @delayed
            def evaluate_sample(i):
                result = evaluate_cascade_model(
                    samples[i],
                    problem,
                    **eval_kwargs
                )
                return result[output_name]

            # Create delayed tasks
            tasks = [evaluate_sample(i) for i in range(n_samples)]

            # Execute in parallel
            results = compute(*tasks, scheduler='threads', num_workers=n_workers)
            outputs = np.array(results)

        except ImportError:
            parallel = False

    if not parallel:
        # Serial evaluation with progress
        for i in range(n_samples):
            if i % 100 == 0:
                print(f"Evaluating sample {i}/{n_samples}")

            result = evaluate_cascade_model(samples[i], problem, **eval_kwargs)
            outputs[i] = result[output_name]

    # Handle NaN values
    valid_mask = ~np.isnan(outputs)
    if np.sum(~valid_mask) > 0:
        warnings.warn(f"{np.sum(~valid_mask)} samples produced NaN outputs")
        # Replace NaN with mean for Sobol analysis
        outputs = np.where(np.isnan(outputs), np.nanmean(outputs), outputs)

    # Compute Sobol indices
    indices = sobol.analyze(
        problem,
        outputs,
        calc_second_order=True,
        print_to_console=False
    )

    return outputs, indices


# ========== Results Visualization ==========

def plot_sobol_indices(
    indices: Dict[str, np.ndarray],
    problem: Dict[str, Any],
    output_name: str = 'Output',
    figsize: Tuple[float, float] = (12, 5)
) -> 'matplotlib.figure.Figure':
    """
    Create visualization of Sobol indices.

    Parameters
    ----------
    indices : dict
        Sobol indices from run_sobol_analysis
    problem : dict
        Problem definition with parameter names
    output_name : str
        Name of output for plot title
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    names = problem['names']
    n_vars = len(names)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # First-order indices
    ax1 = axes[0]
    x = np.arange(n_vars)
    width = 0.35

    bars1 = ax1.bar(
        x - width/2,
        indices['S1'],
        width,
        yerr=indices['S1_conf'],
        label='First-order (S1)',
        color='steelblue',
        alpha=0.8,
        capsize=3
    )
    bars2 = ax1.bar(
        x + width/2,
        indices['ST'],
        width,
        yerr=indices['ST_conf'],
        label='Total (ST)',
        color='darkorange',
        alpha=0.8,
        capsize=3
    )

    ax1.set_xlabel('Parameter')
    ax1.set_ylabel('Sobol Index')
    ax1.set_title(f'Sobol Indices for {output_name}')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.legend()
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax1.set_ylim(bottom=-0.1)

    # Second-order indices (heatmap)
    ax2 = axes[1]

    if 'S2' in indices and indices['S2'] is not None:
        S2 = indices['S2']
        # S2 is upper triangular, mirror it
        S2_full = S2 + S2.T
        np.fill_diagonal(S2_full, 0)

        im = ax2.imshow(S2_full, cmap='RdBu_r', aspect='auto')
        ax2.set_xticks(range(n_vars))
        ax2.set_yticks(range(n_vars))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_yticklabels(names)
        ax2.set_title('Second-order Indices (S2)')
        plt.colorbar(im, ax=ax2)
    else:
        ax2.text(0.5, 0.5, 'S2 not computed', ha='center', va='center')
        ax2.set_title('Second-order Indices')

    plt.tight_layout()
    return fig


def summarize_sobol_results(
    indices: Dict[str, np.ndarray],
    problem: Dict[str, Any],
    threshold: float = 0.05
) -> str:
    """
    Generate text summary of Sobol analysis results.

    Parameters
    ----------
    indices : dict
        Sobol indices
    problem : dict
        Problem definition
    threshold : float
        Minimum index to consider significant

    Returns
    -------
    str
        Formatted summary text
    """
    names = problem['names']

    # Sort by total index
    sorted_idx = np.argsort(indices['ST'])[::-1]

    lines = [
        "=" * 60,
        "SOBOL SENSITIVITY ANALYSIS RESULTS",
        "=" * 60,
        "",
        "Parameters ranked by Total Index (ST):",
        "-" * 40
    ]

    for i, idx in enumerate(sorted_idx):
        st = indices['ST'][idx]
        s1 = indices['S1'][idx]
        interaction = st - s1

        significance = "***" if st > 0.2 else "**" if st > 0.1 else "*" if st > threshold else ""

        lines.append(
            f"{i+1:2d}. {names[idx]:25s} ST={st:6.3f} S1={s1:6.3f} "
            f"Int={interaction:6.3f} {significance}"
        )

    # Find significant interactions
    if 'S2' in indices and indices['S2'] is not None:
        S2 = indices['S2']
        lines.extend([
            "",
            "Significant Interactions (S2 > threshold):",
            "-" * 40
        ])

        for i in range(len(names)):
            for j in range(i+1, len(names)):
                if S2[i, j] > threshold:
                    lines.append(
                        f"  {names[i]} x {names[j]}: {S2[i, j]:.3f}"
                    )

    # Key findings
    lines.extend([
        "",
        "KEY FINDINGS:",
        "-" * 40,
        f"Most influential parameter: {names[sorted_idx[0]]} (ST={indices['ST'][sorted_idx[0]]:.3f})",
        f"Parameters with ST > 0.1: {sum(indices['ST'] > 0.1)}",
        f"Total variance explained by S1: {sum(indices['S1']):.1%}",
        f"Interaction effects: {sum(indices['ST'] - indices['S1']):.1%}",
        ""
    ])

    return "\n".join(lines)


# ========== Quick Analysis Functions ==========

def quick_sensitivity_scan(
    n_samples: int = 256,
    n_runs_per_sample: int = 5,
    output_metrics: List[str] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run a quick sensitivity scan with default settings.

    Good for initial exploration before full analysis.

    Parameters
    ----------
    n_samples : int
        Number of base samples (total ~2500 for 8 params)
    n_runs_per_sample : int
        Ensemble size per parameter set
    output_metrics : list
        Which metrics to analyze

    Returns
    -------
    dict
        Sobol indices for each metric
    """
    if output_metrics is None:
        output_metrics = ['cascade_fraction', 'recovery_fraction']

    problem = define_energy_problem()
    samples = generate_samples(problem, n_samples, calc_second_order=True)

    results = {}
    for metric in output_metrics:
        print(f"\nAnalyzing: {metric}")
        outputs, indices = run_sobol_analysis(
            samples, problem,
            output_name=metric,
            n_runs=n_runs_per_sample
        )
        results[metric] = {
            'outputs': outputs,
            'indices': indices
        }

        # Print summary
        print(summarize_sobol_results(indices, problem))

    return results


# ========== Export Interface ==========

__all__ = [
    'define_energy_problem',
    'define_three_layer_problem',
    'generate_samples',
    'evaluate_cascade_model',
    'run_sobol_analysis',
    'plot_sobol_indices',
    'summarize_sobol_results',
    'quick_sensitivity_scan',
    'SALIB_AVAILABLE'
]
