"""
Dask Utilities for Parallel Energy-Constrained Simulations

Provides distributed computing support for large-scale experiments:
1. Parallel ensemble runs
2. Distributed parameter sweeps
3. Parallel analysis across ensemble members

Usage:
------
>>> from energy_constrained.dask_utils import get_dask_client, run_ensemble_parallel
>>> client = get_dask_client()  # Connects to k3s Dask cluster if available
>>> results = run_ensemble_parallel(network, n_runs=100, ...)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import warnings
import pickle

# Conditional Dask import
try:
    from dask.distributed import Client, as_completed, get_client
    from dask import delayed
    import dask.bag as db
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    warnings.warn("Dask not available. Install with: pip install dask distributed")


# Default Dask scheduler address for k3s cluster
K3S_DASK_SCHEDULER = "tcp://cascades-dask-scheduler:8786"


def _init_worker_path():
    """Initialize Python path on worker for energy_constrained module."""
    import sys
    research_path = '/opt/research-local/src'
    if research_path not in sys.path:
        sys.path.insert(0, research_path)
    return True


def get_dask_client(
    scheduler_address: Optional[str] = None,
    fallback_local: bool = True,
    n_workers_local: int = 4
) -> Optional['Client']:
    """
    Get or create a Dask client.

    Attempts to connect to the k3s Dask cluster first, falls back to
    local cluster if unavailable.

    Parameters
    ----------
    scheduler_address : str, optional
        Dask scheduler address. If None, tries k3s cluster first.
    fallback_local : bool
        If True, create local cluster if remote unavailable
    n_workers_local : int
        Number of workers for local fallback cluster

    Returns
    -------
    Client or None
        Dask client, or None if unavailable
    """
    if not DASK_AVAILABLE:
        warnings.warn("Dask not installed. Running in serial mode.")
        return None

    # Try to get existing client
    try:
        client = get_client()
        # Verify client is connected to k3s (not local) and healthy
        try:
            scheduler_addr = client.scheduler.address if hasattr(client, 'scheduler') else ''
            n_workers = len(client.scheduler_info()['workers'])

            # Check if connected to k3s cluster (not local)
            is_k3s = 'cascades-dask-scheduler' in scheduler_addr or scheduler_addr == K3S_DASK_SCHEDULER

            if n_workers > 0 and is_k3s:
                print(f"Using existing Dask client: {client.dashboard_link}")
                print(f"  Scheduler: {scheduler_addr}")
                print(f"  Workers: {n_workers}")
                # Ensure workers have the path initialized
                client.run(_init_worker_path)
                return client
            else:
                # Connected to local cluster, close and reconnect to k3s
                print(f"Existing client is local ({scheduler_addr}), reconnecting to k3s...")
                try:
                    client.close()
                except Exception:
                    pass
        except Exception as e:
            # Existing client is stale, close it and reconnect
            print(f"Existing client unhealthy: {e}")
            try:
                client.close()
            except Exception:
                pass
    except ValueError:
        pass  # No existing client

    # Try k3s cluster
    if scheduler_address is None:
        scheduler_address = K3S_DASK_SCHEDULER

    try:
        client = Client(scheduler_address, timeout='10s')
        n_workers = len(client.scheduler_info()['workers'])
        print(f"Connected to Dask cluster at {scheduler_address}")
        print(f"  Workers: {n_workers}")
        print(f"  Dashboard: {client.dashboard_link}")

        # Initialize Python path on all workers
        print("  Initializing worker paths...")
        client.run(_init_worker_path)
        print("  Workers ready.")

        return client
    except Exception as e:
        print(f"Could not connect to {scheduler_address}: {e}")

        if fallback_local:
            print(f"Falling back to local cluster with {n_workers_local} workers")
            from dask.distributed import LocalCluster
            cluster = LocalCluster(n_workers=n_workers_local, threads_per_worker=1)
            client = Client(cluster)
            print(f"  Dashboard: {client.dashboard_link}")
            return client
        else:
            return None


def _run_single_simulation(
    network_bytes: bytes,
    run_idx: int,
    duration: float,
    dt: float,
    sigma: float,
    alpha: float,
    seed: int,
    x0: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Worker function for running a single simulation.

    Network is passed as pickled bytes to avoid serialization issues.
    Returns dict instead of SolverResult for easier Dask handling.
    """
    import sys
    import numpy as np
    import pickle

    # Add research code path for workers (k3s environment)
    research_path = '/opt/research-local/src'
    if research_path not in sys.path:
        sys.path.insert(0, research_path)

    # Reconstruct network
    network = pickle.loads(network_bytes)

    # Set random seed
    np.random.seed(seed + run_idx)

    # Import here to avoid circular imports on workers
    from energy_constrained.solvers import energy_constrained_euler_maruyama

    n_elements = network.n_elements
    sigma_arr = sigma * np.ones(n_elements)
    alpha_arr = alpha * np.ones(n_elements)

    y0 = network.get_initial_state(x0)

    result = energy_constrained_euler_maruyama(
        f_extended=network.f_extended,
        y0=y0,
        t_span=(0, duration),
        dt=dt,
        sigma=sigma_arr,
        alpha=alpha_arr
    )

    # Return as dict for easier serialization
    return {
        'run_idx': run_idx,
        't': result.t,
        'x': result.x,
        'E': result.E,
        'y': result.y,
        'diagnostics': result.diagnostics
    }


def run_ensemble_parallel(
    network,
    n_runs: int,
    duration: float,
    dt: float,
    sigma: float,
    alpha: float = 2.0,
    x0: Optional[np.ndarray] = None,
    seed: int = 42,
    client: Optional['Client'] = None,
    batch_size: int = 10
) -> List[Dict[str, Any]]:
    """
    Run ensemble of simulations in parallel using Dask.

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
        Lévy stability parameter (2.0 = Gaussian)
    x0 : array-like, optional
        Initial states
    seed : int
        Base random seed
    client : Client, optional
        Dask client (auto-detected if None)
    batch_size : int
        Number of runs to submit at once

    Returns
    -------
    list of dict
        Ensemble results (same structure as SolverResult)
    """
    if client is None:
        client = get_dask_client()

    if client is None:
        # Fall back to serial execution
        print("No Dask client available, running serially...")
        from .solvers import run_ensemble, SolverResult
        results = run_ensemble(
            network, n_runs, duration, dt, sigma, alpha, x0, seed
        )
        # Convert to dict format
        return [
            {'run_idx': i, 't': r.t, 'x': r.x, 'E': r.E, 'y': r.y,
             'diagnostics': r.diagnostics}
            for i, r in enumerate(results)
        ]

    # Serialize network once and SCATTER to workers
    # This sends data ONCE to all workers instead of with every task
    network_bytes = pickle.dumps(network)
    network_future = client.scatter(network_bytes, broadcast=True)
    print(f"Network scattered to workers ({len(network_bytes) / 1024:.1f} KB)")

    print(f"Submitting {n_runs} simulations to Dask cluster...")

    # Submit all jobs using scattered data reference
    futures = []
    for i in range(n_runs):
        future = client.submit(
            _run_single_simulation,
            network_future, i, duration, dt, sigma, alpha, seed, x0,
            key=f"sim_{seed}_{i}",
            pure=False  # These are stochastic simulations
        )
        futures.append(future)

    # Gather results with progress
    results = []
    for i, future in enumerate(as_completed(futures)):
        result = future.result()
        results.append(result)
        if (i + 1) % batch_size == 0:
            print(f"  Completed {i+1}/{n_runs}")

    # Sort by run_idx
    results.sort(key=lambda r: r['run_idx'])

    print(f"All {n_runs} simulations complete.")
    return results


def _analyze_single_result(
    result_dict: Dict[str, Any],
    network_bytes: bytes,
    T: float = 1.0
) -> Dict[str, Any]:
    """
    Worker function for analyzing a single simulation result.
    """
    import sys
    import numpy as np
    import pickle

    # Add research code path for workers (k3s environment)
    research_path = '/opt/research-local/src'
    if research_path not in sys.path:
        sys.path.insert(0, research_path)

    network = pickle.loads(network_bytes)

    # Reconstruct SolverResult
    from energy_constrained.solvers import SolverResult
    result = SolverResult(
        t=result_dict['t'],
        x=result_dict['x'],
        E=result_dict['E'],
        y=result_dict['y'],
        diagnostics=result_dict.get('diagnostics', {})
    )

    # Run analysis
    from energy_constrained.analysis import EnergyAnalyzer
    analyzer = EnergyAnalyzer(network, result)

    # Compute metrics
    budget = analyzer.compute_energy_budget(T)
    events = analyzer.identify_tipping_events()

    tip_events = [e for e in events if e.direction == 'tip']
    recover_events = [e for e in events if e.direction == 'recover']

    # Time in tipped state per element
    time_tipped = np.mean(result.x > 0, axis=0)

    return {
        'run_idx': result_dict['run_idx'],
        'total_entropy': np.trapz(budget.entropy_production, budget.t),
        'mean_dissipation': np.mean(budget.dissipation_rate),
        'n_tip_events': len(tip_events),
        'n_recover_events': len(recover_events),
        'time_tipped': time_tipped,
        'final_state': result.x[-1],
        'final_energy': result.E[-1]
    }


def analyze_ensemble_parallel(
    results: List[Dict[str, Any]],
    network,
    T: float = 1.0,
    client: Optional['Client'] = None
) -> Dict[str, Any]:
    """
    Analyze ensemble results in parallel using Dask.

    Parameters
    ----------
    results : list of dict
        Ensemble results from run_ensemble_parallel
    network : EnergyConstrainedNetwork
        Network that was simulated
    T : float
        Reference temperature for entropy calculations
    client : Client, optional
        Dask client (auto-detected if None)

    Returns
    -------
    dict
        Aggregated analysis results
    """
    if client is None:
        client = get_dask_client()

    n_runs = len(results)
    network_bytes = pickle.dumps(network)

    if client is None:
        # Serial fallback
        print("No Dask client available, analyzing serially...")
        analyses = [
            _analyze_single_result(r, network_bytes, T)
            for r in results
        ]
    else:
        print(f"Analyzing {n_runs} results in parallel...")
        futures = [
            client.submit(_analyze_single_result, r, network_bytes, T)
            for r in results
        ]
        analyses = client.gather(futures)
        print("Analysis complete.")

    # Aggregate results
    total_entropy = [a['total_entropy'] for a in analyses]
    mean_dissipation = [a['mean_dissipation'] for a in analyses]
    n_tip_events = [a['n_tip_events'] for a in analyses]
    n_recover_events = [a['n_recover_events'] for a in analyses]
    time_tipped = np.array([a['time_tipped'] for a in analyses])

    return {
        'total_entropy': {
            'mean': np.mean(total_entropy),
            'std': np.std(total_entropy),
            'values': total_entropy
        },
        'mean_dissipation': {
            'mean': np.mean(mean_dissipation),
            'std': np.std(mean_dissipation),
            'values': mean_dissipation
        },
        'n_tip_events': {
            'mean': np.mean(n_tip_events),
            'std': np.std(n_tip_events),
            'values': n_tip_events
        },
        'n_recover_events': {
            'mean': np.mean(n_recover_events),
            'std': np.std(n_recover_events),
            'values': n_recover_events
        },
        'time_tipped': {
            'mean': np.mean(time_tipped, axis=0),
            'std': np.std(time_tipped, axis=0)
        },
        'n_runs': n_runs,
        'individual_analyses': analyses
    }


def run_parameter_sweep_parallel(
    network,
    param_grid: Dict[str, List],
    duration: float,
    dt: float,
    n_runs_per_config: int = 10,
    seed: int = 42,
    client: Optional['Client'] = None
) -> List[Dict[str, Any]]:
    """
    Run parameter sweep in parallel.

    Parameters
    ----------
    network : EnergyConstrainedNetwork
        Base network
    param_grid : dict
        Parameter grid, e.g. {'sigma': [0.02, 0.05, 0.08], 'alpha': [1.5, 2.0]}
    duration : float
        Simulation duration
    dt : float
        Time step
    n_runs_per_config : int
        Ensemble size per parameter combination
    seed : int
        Base random seed
    client : Client, optional
        Dask client

    Returns
    -------
    list of dict
        Results for each parameter combination
    """
    import itertools

    if client is None:
        client = get_dask_client()

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    print(f"Running parameter sweep: {len(combinations)} configurations")
    print(f"  Parameters: {param_names}")
    print(f"  Runs per config: {n_runs_per_config}")
    print(f"  Total simulations: {len(combinations) * n_runs_per_config}")

    sweep_results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        print(f"\nConfig {i+1}/{len(combinations)}: {params}")

        # Extract sigma and alpha, use defaults for others
        sigma = params.get('sigma', 0.05)
        alpha = params.get('alpha', 2.0)

        # Run ensemble for this configuration
        results = run_ensemble_parallel(
            network,
            n_runs=n_runs_per_config,
            duration=duration,
            dt=dt,
            sigma=sigma,
            alpha=alpha,
            seed=seed + i * 1000,  # Different seed range per config
            client=client
        )

        # Analyze
        analysis = analyze_ensemble_parallel(results, network, client=client)

        sweep_results.append({
            'params': params,
            'analysis': analysis
        })

    return sweep_results


def results_to_solver_results(results: List[Dict[str, Any]]) -> List:
    """
    Convert dict results back to SolverResult objects.

    Useful when you need the original SolverResult format.
    """
    from .solvers import SolverResult

    return [
        SolverResult(
            t=r['t'],
            x=r['x'],
            E=r['E'],
            y=r['y'],
            diagnostics=r.get('diagnostics', {})
        )
        for r in results
    ]


def submit_experiment_batch(
    client: 'Client',
    worker_func: Callable,
    network_bytes: bytes,
    tasks: List[Dict[str, Any]],
    scatter_network: bool = True
) -> List:
    """
    Submit a batch of experiments with optimized data transfer.

    This function pre-scatters the network data to all workers ONCE,
    then submits all tasks using references to the scattered data.
    This eliminates the O(n_tasks * network_size) serialization overhead.

    Parameters
    ----------
    client : Client
        Dask client
    worker_func : callable
        Function to execute on workers. Should accept (network_bytes, **task_kwargs)
    network_bytes : bytes
        Pickled network data
    tasks : list of dict
        List of task configurations. Each dict contains kwargs for worker_func.
        Must include 'key' for task naming.
    scatter_network : bool
        If True, scatter network to workers. Set False if already scattered.

    Returns
    -------
    list of Future
        Futures for all submitted tasks

    Example
    -------
    >>> tasks = [
    ...     {'recovery_alpha': 1.5, 'seed': 42, 'key': 'alpha_1.5_run_0'},
    ...     {'recovery_alpha': 1.5, 'seed': 43, 'key': 'alpha_1.5_run_1'},
    ... ]
    >>> futures = submit_experiment_batch(client, worker_func, network_bytes, tasks)
    >>> for future in as_completed(futures):
    ...     result = future.result()
    """
    # Scatter network data ONCE to all workers
    if scatter_network:
        network_future = client.scatter(network_bytes, broadcast=True)
        print(f"  Network scattered to workers ({len(network_bytes) / 1024:.1f} KB)")
    else:
        network_future = network_bytes  # Assume already a Future

    # Submit all tasks at once - no waiting between submissions
    futures = []
    for task in tasks:
        key = task.pop('key', None)
        future = client.submit(
            worker_func,
            network_bytes=network_future,
            **task,
            key=key,
            pure=False  # Stochastic simulations
        )
        futures.append(future)

    return futures


def run_alpha_sweep_optimized(
    client: 'Client',
    network,
    alpha_values: List[float],
    n_runs_per_alpha: int,
    config: Dict[str, Any],
    worker_func: Callable,
    base_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Run alpha-sweep experiment with optimized Dask performance.

    Designed to maximize worker utilization by:
    1. Scattering network data ONCE to all workers
    2. Submitting ALL tasks upfront (not waiting for completions)
    3. Using as_completed for streaming results

    Parameters
    ----------
    client : Client
        Dask client
    network : EnergyConstrainedNetwork
        Network to simulate
    alpha_values : list of float
        Alpha values to sweep
    n_runs_per_alpha : int
        Number of runs per alpha
    config : dict
        Experiment configuration passed to worker_func
    worker_func : callable
        Worker function that accepts (network_bytes, recovery_alpha, config, seed)
    base_seed : int
        Base random seed

    Returns
    -------
    list of dict
        All results
    """
    import time

    # Serialize and scatter network
    network_bytes = pickle.dumps(network)
    network_future = client.scatter(network_bytes, broadcast=True)
    print(f"Network scattered to workers ({len(network_bytes) / 1024:.1f} KB)")

    # Build all tasks upfront
    total_tasks = len(alpha_values) * n_runs_per_alpha
    print(f"Submitting {total_tasks} tasks...")

    start_submit = time.time()
    futures = []
    task_info = {}  # Map future key to (alpha, run_idx)

    for i, alpha in enumerate(alpha_values):
        for run_idx in range(n_runs_per_alpha):
            seed = base_seed + i * 1000 + run_idx
            key = f"alpha_{alpha:.1f}_run_{run_idx}"

            future = client.submit(
                worker_func,
                network_bytes=network_future,
                recovery_alpha=float(alpha),
                config=config,
                seed=seed,
                key=key,
                pure=False
            )
            futures.append(future)
            task_info[key] = (alpha, run_idx)

    submit_time = time.time() - start_submit
    print(f"All {total_tasks} tasks submitted in {submit_time:.1f}s")

    # Collect results with progress
    all_results = []
    completed_by_alpha = {float(a): 0 for a in alpha_values}

    print("Processing results...")
    for future in as_completed(futures):
        result = future.result()
        all_results.append(result)

        # Track progress
        alpha = result.get('recovery_alpha', 0)
        if alpha in completed_by_alpha:
            completed_by_alpha[alpha] += 1

        # Print progress
        total_completed = len(all_results)
        if total_completed % n_runs_per_alpha == 0:
            print(f"  Completed {total_completed}/{total_tasks}")

    return all_results
