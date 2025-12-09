"""
Energy Flow and Entropy Production Analysis

Tools for analyzing thermodynamic properties of tipping cascade
simulations, including:

1. Energy budget tracking
2. Entropy production rates
3. Cascade energy costs
4. Thermodynamic barrier identification

These tools help test Phase 2 hypotheses:
- Does asymmetric coupling correlate with entropy production patterns?
- Is protective coupling thermodynamically favorable?
- What's the energy "cost" of cascade propagation?

References:
-----------
- Phase 2 findings on asymmetric coupling
- Kramers escape rate theory
- Wang (2015): Landscape-flux theory
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings


@dataclass
class TippingEvent:
    """Information about a single tipping event."""
    time: float
    element_idx: int
    element_name: str
    direction: str  # 'tip' or 'recover'
    x_before: float
    x_after: float
    E_before: float
    E_after: float
    delta_E: float


@dataclass
class EnergyBudget:
    """Energy budget over time."""
    t: np.ndarray
    E_total: np.ndarray
    E_elements: np.ndarray
    dissipation_rate: np.ndarray
    entropy_production: np.ndarray
    energy_flows: Optional[np.ndarray] = None


class EnergyAnalyzer:
    """
    Tools for analyzing energy flow and entropy production.

    This class provides methods for post-processing simulation results
    to extract thermodynamic information.

    Parameters
    ----------
    network : EnergyConstrainedNetwork
        The network that was simulated
    result : SolverResult
        Simulation output

    Examples
    --------
    >>> analyzer = EnergyAnalyzer(network, result)
    >>> budget = analyzer.compute_energy_budget()
    >>> events = analyzer.identify_tipping_events()
    >>> costs = analyzer.compute_cascade_energy_costs(events)
    """

    def __init__(self, network, result):
        """
        Initialize analyzer.

        Parameters
        ----------
        network : EnergyConstrainedNetwork
            The simulated network
        result : SolverResult
            Simulation results
        """
        self.net = network
        self.t = result.t
        self.x = result.x
        self.E = result.E
        self.y = result.y

        self.n_times = len(self.t)
        self.n_elements = self.x.shape[1]

    def compute_energy_budget(
        self,
        T: float = 1.0
    ) -> EnergyBudget:
        """
        Compute energy budget components over time.

        Parameters
        ----------
        T : float
            Reference temperature for entropy calculations

        Returns
        -------
        EnergyBudget
            Energy budget over time
        """
        E_total = np.sum(self.E, axis=1)
        E_elements = self.E.copy()

        dissipation_rate = np.zeros(self.n_times)
        entropy_production = np.zeros(self.n_times)

        for i in range(1, self.n_times):
            dt = self.t[i] - self.t[i-1]
            if dt > 0:
                dxdt = (self.x[i] - self.x[i-1]) / dt
                dissipation_rate[i] = np.sum(
                    self.net.compute_dissipation_rates(self.x[i], dxdt)
                )
                entropy_production[i] = self.net.compute_entropy_production(
                    self.x[i], dxdt, T
                )

        return EnergyBudget(
            t=self.t,
            E_total=E_total,
            E_elements=E_elements,
            dissipation_rate=dissipation_rate,
            entropy_production=entropy_production
        )

    def compute_entropy_production_rate(self, T: float = 1.0) -> np.ndarray:
        """
        Compute instantaneous entropy production rate.

        Parameters
        ----------
        T : float
            Reference temperature

        Returns
        -------
        sigma : ndarray
            Entropy production rate over time
        """
        budget = self.compute_energy_budget(T)
        return budget.entropy_production

    def compute_total_entropy_produced(
        self,
        T: float = 1.0,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None
    ) -> float:
        """
        Compute total entropy produced over time interval.

        Parameters
        ----------
        T : float
            Reference temperature
        t_start, t_end : float, optional
            Time interval (defaults to full simulation)

        Returns
        -------
        float
            Total entropy produced
        """
        sigma = self.compute_entropy_production_rate(T)

        # Time indices
        if t_start is None:
            i_start = 0
        else:
            i_start = np.searchsorted(self.t, t_start)

        if t_end is None:
            i_end = self.n_times
        else:
            i_end = np.searchsorted(self.t, t_end)

        # Integrate using trapezoidal rule
        return np.trapz(sigma[i_start:i_end], self.t[i_start:i_end])

    def identify_tipping_events(
        self,
        threshold: float = 0.0,
        min_duration: float = 0.0
    ) -> List[TippingEvent]:
        """
        Identify tipping events in trajectory.

        A tipping event is a crossing of the threshold that persists
        for at least min_duration.

        Parameters
        ----------
        threshold : float
            State threshold for tipping
        min_duration : float
            Minimum duration to count as event

        Returns
        -------
        list of TippingEvent
            Identified tipping events
        """
        events = []

        for i in range(self.n_elements):
            x_series = self.x[:, i]
            E_series = self.E[:, i]

            # Find threshold crossings
            tipped = x_series > threshold
            crossings = np.where(np.diff(tipped.astype(int)) != 0)[0]

            name = self.net.node_list[i] if hasattr(self.net, 'node_list') else f'element_{i}'

            for idx in crossings:
                if idx + 1 >= self.n_times:
                    continue

                direction = 'tip' if tipped[idx + 1] else 'recover'

                # Check duration (if min_duration > 0)
                if min_duration > 0 and direction == 'tip':
                    # Find when it crosses back
                    future_tipped = tipped[idx+1:]
                    if not future_tipped.all():
                        recovery_idx = np.where(~future_tipped)[0][0]
                        duration = self.t[idx + 1 + recovery_idx] - self.t[idx + 1]
                        if duration < min_duration:
                            continue

                events.append(TippingEvent(
                    time=self.t[idx],
                    element_idx=i,
                    element_name=name,
                    direction=direction,
                    x_before=x_series[idx],
                    x_after=x_series[idx + 1],
                    E_before=E_series[idx],
                    E_after=E_series[idx + 1],
                    delta_E=E_series[idx + 1] - E_series[idx]
                ))

        # Sort by time
        events.sort(key=lambda e: e.time)
        return events

    def compute_cascade_energy_costs(
        self,
        events: Optional[List[TippingEvent]] = None,
        window: float = 10.0
    ) -> Dict[str, Any]:
        """
        Compute energy cost associated with each tipping event.

        The "cost" includes:
        - Direct energy change (delta_E)
        - Entropy produced during transition
        - Net energy flow to/from other elements

        Parameters
        ----------
        events : list of TippingEvent, optional
            Events to analyze (auto-detected if None)
        window : float
            Time window around event for analysis

        Returns
        -------
        dict
            Energy costs for each event
        """
        if events is None:
            events = self.identify_tipping_events()

        costs = []

        for event in events:
            # Time window around event
            t_start = max(0, event.time - window/2)
            t_end = min(self.t[-1], event.time + window/2)

            i_start = np.searchsorted(self.t, t_start)
            i_end = np.searchsorted(self.t, t_end)

            if i_end <= i_start:
                continue

            # Total entropy produced in window
            entropy_cost = self.compute_total_entropy_produced(
                t_start=t_start, t_end=t_end
            )

            # Energy change of the tipping element
            elem_idx = event.element_idx
            E_change = self.E[i_end-1, elem_idx] - self.E[i_start, elem_idx]

            # System energy change
            E_system_change = (
                np.sum(self.E[i_end-1]) - np.sum(self.E[i_start])
            )

            costs.append({
                'event': event,
                'element_E_change': E_change,
                'system_E_change': E_system_change,
                'entropy_produced': entropy_cost,
                'thermodynamic_cost': entropy_cost,  # σ = "cost"
            })

        # Summary statistics
        if costs:
            avg_entropy = np.mean([c['entropy_produced'] for c in costs])
            avg_E_change = np.mean([c['element_E_change'] for c in costs])

            # Split by direction
            tip_costs = [c for c in costs if c['event'].direction == 'tip']
            recover_costs = [c for c in costs if c['event'].direction == 'recover']
        else:
            avg_entropy = 0
            avg_E_change = 0
            tip_costs = []
            recover_costs = []

        return {
            'individual_costs': costs,
            'average_entropy_per_event': avg_entropy,
            'average_E_change_per_event': avg_E_change,
            'n_tip_events': len(tip_costs),
            'n_recover_events': len(recover_costs),
            'tip_entropy_avg': np.mean([c['entropy_produced'] for c in tip_costs]) if tip_costs else 0,
            'recover_entropy_avg': np.mean([c['entropy_produced'] for c in recover_costs]) if recover_costs else 0,
        }

    def find_thermodynamic_barriers(
        self,
        n_samples: int = 100
    ) -> Dict[str, float]:
        """
        Estimate effective thermodynamic barriers for each element.

        Uses trajectory statistics to estimate barrier heights via
        modified Kramers analysis.

        Parameters
        ----------
        n_samples : int
            Number of time windows for estimation

        Returns
        -------
        dict
            Estimated barrier heights per element
        """
        barriers = {}

        for i in range(self.n_elements):
            name = self.net.node_list[i] if hasattr(self.net, 'node_list') else f'element_{i}'

            x_series = self.x[:, i]
            E_series = self.E[:, i]

            # Estimate barrier from energy fluctuations
            # Barrier ~ max(E) - min(E) in stable state
            stable_mask = x_series < 0

            if np.sum(stable_mask) > 10:
                E_stable = E_series[stable_mask]
                E_range = np.percentile(E_stable, 95) - np.percentile(E_stable, 5)
                barriers[name] = E_range
            else:
                # Not enough stable data
                barriers[name] = np.nan

        return barriers

    def compare_coupling_configurations(
        self,
        other_analyzer: 'EnergyAnalyzer',
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare thermodynamic properties between two configurations.

        Useful for testing hypothesis that asymmetric coupling
        minimizes entropy production.

        Parameters
        ----------
        other_analyzer : EnergyAnalyzer
            Analyzer for comparison configuration
        metrics : list of str, optional
            Metrics to compare (default: all)

        Returns
        -------
        dict
            Comparison results
        """
        if metrics is None:
            metrics = ['total_entropy', 'mean_dissipation', 'n_tip_events']

        results = {}

        # Total entropy produced
        if 'total_entropy' in metrics:
            s1 = self.compute_total_entropy_produced()
            s2 = other_analyzer.compute_total_entropy_produced()
            results['total_entropy'] = {
                'config1': s1,
                'config2': s2,
                'ratio': s1/s2 if s2 > 0 else float('inf'),
                'difference': s1 - s2
            }

        # Mean dissipation rate
        if 'mean_dissipation' in metrics:
            b1 = self.compute_energy_budget()
            b2 = other_analyzer.compute_energy_budget()
            d1 = np.mean(b1.dissipation_rate)
            d2 = np.mean(b2.dissipation_rate)
            results['mean_dissipation'] = {
                'config1': d1,
                'config2': d2,
                'ratio': d1/d2 if d2 > 0 else float('inf')
            }

        # Number of tipping events
        if 'n_tip_events' in metrics:
            e1 = len([e for e in self.identify_tipping_events() if e.direction == 'tip'])
            e2 = len([e for e in other_analyzer.identify_tipping_events() if e.direction == 'tip'])
            results['n_tip_events'] = {
                'config1': e1,
                'config2': e2,
                'difference': e1 - e2
            }

        return results


def analyze_ensemble_thermodynamics(
    network,
    results_list: List,
    burn_in: float = 0.1
) -> Dict[str, Any]:
    """
    Analyze thermodynamic properties across ensemble.

    Parameters
    ----------
    network : EnergyConstrainedNetwork
        The simulated network
    results_list : list of SolverResult
        Ensemble results
    burn_in : float
        Fraction of trajectory to discard

    Returns
    -------
    dict
        Ensemble thermodynamic statistics
    """
    n_runs = len(results_list)

    total_entropy = []
    mean_dissipation = []
    n_tip_events = []
    element_time_tipped = []

    for result in results_list:
        analyzer = EnergyAnalyzer(network, result)

        # Skip burn-in
        burn_idx = int(burn_in * len(result.t))
        t_start = result.t[burn_idx]

        # Total entropy
        s = analyzer.compute_total_entropy_produced(t_start=t_start)
        total_entropy.append(s)

        # Mean dissipation
        budget = analyzer.compute_energy_budget()
        mean_d = np.mean(budget.dissipation_rate[burn_idx:])
        mean_dissipation.append(mean_d)

        # Tipping events
        events = analyzer.identify_tipping_events()
        n_tip = len([e for e in events if e.direction == 'tip' and e.time >= t_start])
        n_tip_events.append(n_tip)

        # Time tipped per element
        time_tipped = np.mean(result.x[burn_idx:] > 0, axis=0)
        element_time_tipped.append(time_tipped)

    element_time_tipped = np.array(element_time_tipped)

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
        'element_time_tipped': {
            'mean': np.mean(element_time_tipped, axis=0),
            'std': np.std(element_time_tipped, axis=0),
        },
        'n_runs': n_runs
    }
