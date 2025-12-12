"""
Energy-Constrained Tipping Network

Extends PyCascades' tipping_network with system-wide energy tracking
and entropy production monitoring.

Key Features:
-------------
1. Extended state space: y = [x_0, ..., x_{n-1}, E_0, ..., E_{n-1}]
2. Energy flow tracking between coupled elements
3. System-wide energy balance: dE_total/dt = P_in - D_total
4. Entropy production rate monitoring

Mathematical Framework:
-----------------------
For N elements, the network tracks:

State dynamics (standard PyCascades):
    dx_i/dt = f_i(x_i) + Σ_j c_ij(x_j, x_i)

Energy dynamics (new):
    dE_i/dt = dE_i/dt_internal - D_i + Σ_j (Φ_ji - Φ_ij)
            = C_i * dx_i/dt - γ_i * (dx_i/dt)² + net energy flow

where:
    C_i = heat capacity
    γ_i = dissipation rate
    Φ_ij = energy flow from i to j

References:
-----------
- Phase 2 findings on asymmetric coupling
- Lucarini et al. (2014): Thermodynamic climate framework
- Wang (2015): Landscape-flux theory
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
import warnings

# Import PyCascades base class
try:
    from pycascades.core.tipping_network import tipping_network
    import networkx as nx
    PYCASCADES_AVAILABLE = True
except ImportError:
    import networkx as nx
    PYCASCADES_AVAILABLE = False

    class tipping_network(nx.DiGraph):
        """Stub base class for development."""
        def __init__(self):
            super().__init__()
            self._f = None
            self._jac = None

        def f(self, x, t):
            """Compute dx/dt for all elements."""
            raise NotImplementedError

        def jac(self, x, t):
            """Compute Jacobian matrix."""
            raise NotImplementedError


from .elements import EnergyConstrainedElement, EnergyConstrainedCusp
from .couplings import EnergyCoupling
import copy


class EnergyConstrainedNetwork(tipping_network):
    """
    Network of tipping elements with energy balance tracking.

    This class extends PyCascades' tipping_network to track energy
    dynamics alongside state dynamics. It maintains an extended state
    vector y = [x_0, ..., x_{n-1}, E_0, ..., E_{n-1}] and computes
    both state and energy evolution.

    Parameters
    ----------
    track_energy_history : bool
        Whether to store energy values at each timestep (default True)
    energy_conservation_check : bool
        Whether to verify energy budget closure (default False)

    Attributes
    ----------
    energy_history : list
        Time series of energy values if tracking enabled
    total_dissipation : float
        Cumulative energy dissipated
    total_energy_in : float
        Cumulative energy input from forcing

    Examples
    --------
    >>> net = EnergyConstrainedNetwork()
    >>> net.add_element('GIS', EnergyConstrainedCusp(E_stable=0, E_tipped=1))
    >>> net.add_element('THC', EnergyConstrainedCusp(E_stable=0, E_tipped=0.5))
    >>> net.add_coupling('GIS', 'THC', GradientDrivenCoupling(0.1))
    """

    def __init__(
        self,
        track_energy_history: bool = True,
        energy_conservation_check: bool = False
    ):
        super().__init__()
        self.track_energy_history = track_energy_history
        self.energy_conservation_check = energy_conservation_check

        # Energy tracking state
        self.energy_history: List[Dict[str, float]] = []
        self.total_dissipation = 0.0
        self.total_energy_in = 0.0

        # Cache for efficiency
        self._node_list: Optional[List[str]] = None
        self._n_elements: Optional[int] = None

    def add_element(self, name: str, element: EnergyConstrainedElement) -> None:
        """
        Add an energy-constrained tipping element to the network.

        Parameters
        ----------
        name : str
            Unique identifier for the element
        element : EnergyConstrainedElement
            The tipping element object
        """
        self.add_node(name, data=element)
        self._invalidate_cache()

    def add_coupling(
        self,
        from_element: str,
        to_element: str,
        coupling: EnergyCoupling
    ) -> None:
        """
        Add an energy-aware coupling between elements.

        Parameters
        ----------
        from_element : str
            Name of source element
        to_element : str
            Name of destination element
        coupling : EnergyCoupling
            The coupling object
        """
        self.add_edge(from_element, to_element, data=coupling)

    def _invalidate_cache(self) -> None:
        """Clear cached node list."""
        self._node_list = None
        self._n_elements = None

    @property
    def node_list(self) -> List[str]:
        """Ordered list of node names."""
        if self._node_list is None:
            self._node_list = list(self.nodes())
        return self._node_list

    @property
    def n_elements(self) -> int:
        """Number of tipping elements."""
        if self._n_elements is None:
            self._n_elements = self.number_of_nodes()
        return self._n_elements

    def get_element(self, name: str) -> EnergyConstrainedElement:
        """Get element by name."""
        return self.nodes[name]['data']

    def get_coupling(self, from_elem: str, to_elem: str) -> EnergyCoupling:
        """Get coupling between elements."""
        return self.edges[from_elem, to_elem]['data']

    # ========== Energy Computation Methods ==========

    def compute_element_energies(self, x: np.ndarray) -> np.ndarray:
        """
        Compute energy of each element from state vector.

        Parameters
        ----------
        x : array-like, shape (n_elements,)
            State variables

        Returns
        -------
        E : ndarray, shape (n_elements,)
            Element energies
        """
        E = np.zeros(self.n_elements)
        for i, name in enumerate(self.node_list):
            element = self.get_element(name)
            # Clip state to prevent overflow in energy calculations
            x_clipped = np.clip(x[i], -10.0, 10.0)
            if hasattr(element, 'energy'):
                E[i] = element.energy(x_clipped)
            else:
                # Fallback: use state as proxy for energy
                E[i] = x_clipped
        # Clip energy values to reasonable range
        return np.clip(E, -100.0, 100.0)

    def compute_total_energy(self, x: np.ndarray) -> float:
        """
        Compute total system energy.

        Parameters
        ----------
        x : array-like
            State variables

        Returns
        -------
        float
            Sum of element energies
        """
        return np.sum(self.compute_element_energies(x))

    def compute_energy_flows(
        self,
        x: np.ndarray,
        E: np.ndarray
    ) -> np.ndarray:
        """
        Compute matrix of energy flow rates between elements.

        Parameters
        ----------
        x : array-like, shape (n_elements,)
            State variables
        E : array-like, shape (n_elements,)
            Element energies

        Returns
        -------
        flows : ndarray, shape (n_elements, n_elements)
            Flow matrix where flows[i,j] = energy flow from i to j
        """
        n = self.n_elements
        flows = np.zeros((n, n))

        for i, from_name in enumerate(self.node_list):
            for j, to_name in enumerate(self.node_list):
                if self.has_edge(from_name, to_name):
                    coupling = self.get_coupling(from_name, to_name)
                    if hasattr(coupling, 'energy_transfer_rate'):
                        flows[i, j] = coupling.energy_transfer_rate(
                            x[i], x[j], E[i], E[j]
                        )

        return flows

    def compute_dissipation_rates(
        self,
        x: np.ndarray,
        dxdt: np.ndarray
    ) -> np.ndarray:
        """
        Compute dissipation rate for each element.

        Parameters
        ----------
        x : array-like, shape (n_elements,)
            State variables
        dxdt : array-like, shape (n_elements,)
            State derivatives

        Returns
        -------
        D : ndarray, shape (n_elements,)
            Dissipation rates (always >= 0)
        """
        D = np.zeros(self.n_elements)
        for i, name in enumerate(self.node_list):
            element = self.get_element(name)
            if hasattr(element, 'dissipation'):
                D[i] = element.dissipation(x[i], dxdt[i])
        return D

    def compute_coupling_dissipation(
        self,
        x: np.ndarray,
        E: np.ndarray
    ) -> float:
        """
        Compute total energy dissipated in coupling transfers.

        Parameters
        ----------
        x, E : array-like
            States and energies

        Returns
        -------
        float
            Total coupling dissipation
        """
        D_total = 0.0
        for i, from_name in enumerate(self.node_list):
            for j, to_name in enumerate(self.node_list):
                if self.has_edge(from_name, to_name):
                    coupling = self.get_coupling(from_name, to_name)
                    if hasattr(coupling, 'dissipation_in_transfer'):
                        D_total += coupling.dissipation_in_transfer(
                            x[i], x[j], E[i], E[j]
                        )
        return D_total

    def compute_entropy_production(
        self,
        x: np.ndarray,
        dxdt: np.ndarray,
        T: float = 1.0
    ) -> float:
        """
        Compute total entropy production rate.

        σ = Σ_i D_i / T_i + coupling entropy

        Parameters
        ----------
        x : array-like
            State variables
        dxdt : array-like
            State derivatives
        T : float
            Reference temperature

        Returns
        -------
        float
            Total entropy production rate
        """
        sigma = 0.0

        # Element dissipation
        for i, name in enumerate(self.node_list):
            element = self.get_element(name)
            if hasattr(element, 'entropy_production'):
                sigma += element.entropy_production(x[i], dxdt[i], T)

        # Coupling dissipation
        E = self.compute_element_energies(x)
        D_coupling = self.compute_coupling_dissipation(x, E)
        sigma += D_coupling / T

        return sigma

    # ========== Dynamics Methods ==========

    def f_state(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Compute state dynamics dx/dt (standard PyCascades dynamics).

        Parameters
        ----------
        x : array-like, shape (n_elements,)
            State variables
        t : float
            Time

        Returns
        -------
        dxdt : ndarray, shape (n_elements,)
            State derivatives
        """
        n = self.n_elements
        dxdt = np.zeros(n)

        # Diagonal (intrinsic) dynamics
        for i, name in enumerate(self.node_list):
            element = self.get_element(name)
            # Clip state to prevent overflow in power calculations
            x_clipped = np.clip(x[i], -10.0, 10.0)
            # PyCascades dxdt_diag() returns a lambda (t, x) -> dx/dt
            if hasattr(element, 'eval_dxdt'):
                dxdt[i] = element.eval_dxdt(x_clipped, t)
            else:
                dxdt[i] = element.dxdt_diag()(t, x_clipped)

        # Coupling contributions
        E = self.compute_element_energies(x)
        for i, from_name in enumerate(self.node_list):
            for j, to_name in enumerate(self.node_list):
                if self.has_edge(from_name, to_name):
                    coupling = self.get_coupling(from_name, to_name)
                    # Coupling affects the 'to' element
                    if hasattr(coupling, 'dxdt_cpl'):
                        try:
                            # Try energy-aware coupling
                            dxdt[j] += coupling.dxdt_cpl(
                                x[i], x[j], E[i], E[j]
                            )
                        except TypeError:
                            # Fall back to standard coupling
                            dxdt[j] += coupling.dxdt_cpl(x[i], x[j])

        return dxdt

    def f_energy(
        self,
        x: np.ndarray,
        dxdt: np.ndarray,
        E: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Compute energy dynamics dE/dt.

        dE_i/dt = internal change - dissipation + net flow

        Parameters
        ----------
        x : array-like
            State variables
        dxdt : array-like
            State derivatives (already computed)
        E : array-like
            Current energies
        t : float
            Time

        Returns
        -------
        dEdt : ndarray
            Energy derivatives
        """
        n = self.n_elements
        dEdt = np.zeros(n)

        # Internal energy change from state evolution
        for i, name in enumerate(self.node_list):
            element = self.get_element(name)
            if hasattr(element, 'dEdt_internal'):
                dEdt[i] += element.dEdt_internal(x[i], dxdt[i])

        # Dissipation (energy loss)
        D = self.compute_dissipation_rates(x, dxdt)
        dEdt -= D

        # Energy flow between elements
        flows = self.compute_energy_flows(x, E)
        for i in range(n):
            # Net flow: incoming - outgoing
            dEdt[i] += flows[:, i].sum() - flows[i, :].sum()

        return dEdt

    def f_extended(self, y: np.ndarray, t: float) -> np.ndarray:
        """
        Extended dynamics for state + energy.

        Parameters
        ----------
        y : array-like, shape (2 * n_elements,)
            Extended state: [x_0, ..., x_{n-1}, E_0, ..., E_{n-1}]
        t : float
            Time

        Returns
        -------
        dydt : ndarray, shape (2 * n_elements,)
            Extended derivatives
        """
        n = self.n_elements
        x = y[:n]
        E = y[n:]

        # State dynamics
        dxdt = self.f_state(x, t)

        # Energy dynamics
        dEdt = self.f_energy(x, dxdt, E, t)

        return np.concatenate([dxdt, dEdt])

    # Override parent's f method if needed
    def f(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Standard dynamics (for compatibility with PyCascades solvers).

        Parameters
        ----------
        x : array-like
            State variables
        t : float
            Time

        Returns
        -------
        dxdt : ndarray
            State derivatives
        """
        return self.f_state(x, t)

    # ========== Initialization ==========

    def get_initial_state(
        self,
        x0: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get initial extended state vector.

        Parameters
        ----------
        x0 : array-like, optional
            Initial states (defaults to stable states)

        Returns
        -------
        y0 : ndarray, shape (2 * n_elements,)
            Initial extended state [x, E]
        """
        n = self.n_elements

        if x0 is None:
            # Default: stable state (x = -1 for cusp)
            x0 = -np.ones(n)

        # Compute initial energies from states
        E0 = self.compute_element_energies(x0)

        return np.concatenate([x0, E0])

    # ========== Analysis Methods ==========

    def get_energy_budget(
        self,
        y_trajectory: np.ndarray,
        t: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute energy budget from trajectory.

        Parameters
        ----------
        y_trajectory : array-like, shape (n_times, 2 * n_elements)
            Time series of extended state
        t : array-like, shape (n_times,)
            Time points

        Returns
        -------
        dict
            Energy budget components over time:
            - 'E_total': total energy
            - 'E_elements': element energies (n_times, n_elements)
            - 'dissipation_rate': instantaneous dissipation
            - 'entropy_production': instantaneous entropy rate
        """
        n_times = len(t)
        n = self.n_elements

        E_total = np.zeros(n_times)
        E_elements = np.zeros((n_times, n))
        dissipation_rate = np.zeros(n_times)
        entropy_prod = np.zeros(n_times)

        for i in range(n_times):
            x = y_trajectory[i, :n]
            E = y_trajectory[i, n:]

            E_total[i] = np.sum(E)
            E_elements[i] = E

            if i > 0:
                dt = t[i] - t[i-1]
                x_prev = y_trajectory[i-1, :n]
                dxdt = (x - x_prev) / dt

                D = self.compute_dissipation_rates(x, dxdt)
                dissipation_rate[i] = np.sum(D)
                entropy_prod[i] = self.compute_entropy_production(x, dxdt)

        return {
            'E_total': E_total,
            'E_elements': E_elements,
            'dissipation_rate': dissipation_rate,
            'entropy_production': entropy_prod
        }

    def identify_tipping_events(
        self,
        y_trajectory: np.ndarray,
        t: np.ndarray,
        threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Identify tipping events in trajectory.

        Parameters
        ----------
        y_trajectory : array-like
            Extended state trajectory
        t : array-like
            Time points
        threshold : float
            State threshold for tipping (default 0)

        Returns
        -------
        list of dict
            Tipping events with time, element, and energy change
        """
        n = self.n_elements
        events = []

        for i, name in enumerate(self.node_list):
            x_series = y_trajectory[:, i]
            E_series = y_trajectory[:, n + i]

            # Find threshold crossings
            tipped = x_series > threshold
            crossings = np.where(np.diff(tipped.astype(int)) != 0)[0]

            for idx in crossings:
                direction = 'tip' if tipped[idx + 1] else 'recover'
                events.append({
                    'time': t[idx],
                    'element': name,
                    'direction': direction,
                    'E_before': E_series[idx],
                    'E_after': E_series[idx + 1],
                    'delta_E': E_series[idx + 1] - E_series[idx]
                })

        # Sort by time
        events.sort(key=lambda e: e['time'])
        return events


# Convenience function for creating standard Earth system network
def create_earth_system_network(
    coupling_type: str = 'gradient',
    **coupling_kwargs
) -> EnergyConstrainedNetwork:
    """
    Create the 4-element Earth system network with energy tracking.

    Elements: GIS, THC, WAIS, AMAZ
    Couplings: Based on Wunderling et al. with specified type

    Parameters
    ----------
    coupling_type : str
        Type of energy coupling ('gradient', 'asymmetric', etc.)
    **coupling_kwargs
        Additional coupling parameters

    Returns
    -------
    EnergyConstrainedNetwork
        Configured network ready for simulation
    """
    from .elements import create_earth_system_element, EARTH_SYSTEM_ELEMENTS
    from .couplings import create_coupling_matrix

    net = EnergyConstrainedNetwork()

    # Add elements
    for name, params in EARTH_SYSTEM_ELEMENTS.items():
        element = create_earth_system_element(name, **params)
        net.add_element(name, element)

    # Wunderling coupling matrix
    coupling_strengths = np.array([
        [0.000, 0.312, 0.271, 0.000],  # GIS ->
        [0.578, 0.000, 0.132, 0.372],  # THC ->
        [0.193, 0.120, 0.000, 0.000],  # WAIS ->
        [0.000, 0.000, 0.000, 0.000],  # AMAZ ->
    ])

    # Create coupling factory
    coupling_factory = create_coupling_matrix(
        coupling_type,
        coupling_strengths,
        **coupling_kwargs
    )

    # Add couplings
    names = list(EARTH_SYSTEM_ELEMENTS.keys())
    for i, from_name in enumerate(names):
        for j, to_name in enumerate(names):
            if coupling_strengths[i, j] > 0:
                coupling = coupling_factory(i, j)
                net.add_coupling(from_name, to_name, coupling)

    return net


# ========== Network Fragmentation Functions ==========

def fragment_network(
    network: EnergyConstrainedNetwork,
    retention_fraction: float,
    method: str = 'random',
    seed: Optional[int] = None
) -> EnergyConstrainedNetwork:
    """
    Create a fragmented copy of a network by removing edges.

    This simulates deforestation or network degradation scenarios by
    progressively removing connections while preserving node structure.

    Parameters
    ----------
    network : EnergyConstrainedNetwork
        Original network to fragment
    retention_fraction : float
        Fraction of edges to retain (0.0 to 1.0)
    method : str
        Edge removal strategy:
        - 'random': Remove edges uniformly at random
        - 'low_flow_first': Remove weakest edges first
        - 'high_betweenness_first': Remove highest betweenness edges first
        - 'low_betweenness_first': Remove lowest betweenness edges first
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    EnergyConstrainedNetwork
        New network with reduced connectivity

    Examples
    --------
    >>> fragmented = fragment_network(original, retention_fraction=0.5, method='random')
    >>> print(f"Edges: {original.number_of_edges()} -> {fragmented.number_of_edges()}")
    """
    if not 0.0 <= retention_fraction <= 1.0:
        raise ValueError("retention_fraction must be between 0.0 and 1.0")

    if seed is not None:
        np.random.seed(seed)

    # Create a deep copy of the network
    fragmented = EnergyConstrainedNetwork(
        track_energy_history=network.track_energy_history,
        energy_conservation_check=network.energy_conservation_check
    )

    # Copy all nodes with their data
    for node in network.nodes():
        element = network.get_element(node)
        fragmented.add_element(node, copy.deepcopy(element))

    # Get all edges
    edges = list(network.edges())
    n_edges = len(edges)
    n_retain = int(n_edges * retention_fraction)

    if n_retain == 0 and retention_fraction > 0:
        n_retain = 1  # Keep at least one edge if retention > 0

    # Determine which edges to keep based on method
    if method == 'random':
        edges_to_keep = _select_random_edges(edges, n_retain)

    elif method == 'low_flow_first':
        # Remove weak edges first (keep strong ones)
        edges_to_keep = _select_by_flow(network, edges, n_retain, keep_strong=True)

    elif method == 'high_flow_first':
        # Remove strong edges first (keep weak ones)
        edges_to_keep = _select_by_flow(network, edges, n_retain, keep_strong=False)

    elif method == 'high_betweenness_first':
        # Remove high-betweenness edges first (targeted attack)
        edges_to_keep = _select_by_betweenness(network, edges, n_retain, keep_high=False)

    elif method == 'low_betweenness_first':
        # Remove low-betweenness edges first (keep critical bridges)
        edges_to_keep = _select_by_betweenness(network, edges, n_retain, keep_high=True)

    else:
        raise ValueError(f"Unknown fragmentation method: {method}")

    # Add retained edges
    for from_node, to_node in edges_to_keep:
        coupling = network.get_coupling(from_node, to_node)
        fragmented.add_coupling(from_node, to_node, copy.deepcopy(coupling))

    return fragmented


def _select_random_edges(
    edges: List[Tuple[str, str]],
    n_retain: int
) -> List[Tuple[str, str]]:
    """Select edges uniformly at random."""
    indices = np.random.choice(len(edges), size=n_retain, replace=False)
    return [edges[i] for i in indices]


def _select_by_flow(
    network: EnergyConstrainedNetwork,
    edges: List[Tuple[str, str]],
    n_retain: int,
    keep_strong: bool = True
) -> List[Tuple[str, str]]:
    """Select edges by coupling strength."""
    # Get coupling strengths
    strengths = []
    for from_node, to_node in edges:
        coupling = network.get_coupling(from_node, to_node)
        if hasattr(coupling, 'strength'):
            strengths.append(coupling.strength)
        elif hasattr(coupling, 'base_strength'):
            strengths.append(coupling.base_strength)
        else:
            strengths.append(1.0)  # Default strength

    # Sort by strength
    sorted_indices = np.argsort(strengths)

    if keep_strong:
        # Keep the strongest edges (remove weak first)
        indices = sorted_indices[-n_retain:]
    else:
        # Keep the weakest edges (remove strong first)
        indices = sorted_indices[:n_retain]

    return [edges[i] for i in indices]


def _select_by_betweenness(
    network: EnergyConstrainedNetwork,
    edges: List[Tuple[str, str]],
    n_retain: int,
    keep_high: bool = True
) -> List[Tuple[str, str]]:
    """Select edges by edge betweenness centrality."""
    # Compute edge betweenness
    betweenness = nx.edge_betweenness_centrality(network)

    # Get betweenness for each edge
    edge_bc = []
    for edge in edges:
        bc = betweenness.get(edge, 0.0)
        edge_bc.append(bc)

    # Sort by betweenness
    sorted_indices = np.argsort(edge_bc)

    if keep_high:
        # Keep high-betweenness edges (remove peripheral first)
        indices = sorted_indices[-n_retain:]
    else:
        # Keep low-betweenness edges (remove critical bridges first)
        indices = sorted_indices[:n_retain]

    return [edges[i] for i in indices]


def compute_network_metrics(network: EnergyConstrainedNetwork) -> Dict[str, float]:
    """
    Compute network connectivity metrics.

    Parameters
    ----------
    network : EnergyConstrainedNetwork
        Network to analyze

    Returns
    -------
    dict
        Network metrics including:
        - n_nodes: Number of nodes
        - n_edges: Number of edges
        - density: Edge density
        - avg_degree: Average node degree
        - clustering: Average clustering coefficient
        - n_components: Number of weakly connected components
        - largest_component_fraction: Fraction of nodes in largest component
    """
    n_nodes = network.number_of_nodes()
    n_edges = network.number_of_edges()

    # Density
    max_edges = n_nodes * (n_nodes - 1)  # Directed graph
    density = n_edges / max_edges if max_edges > 0 else 0.0

    # Average degree
    degrees = [d for n, d in network.degree()]
    avg_degree = np.mean(degrees) if degrees else 0.0

    # Clustering (convert to undirected for this)
    try:
        undirected = network.to_undirected()
        clustering = nx.average_clustering(undirected)
    except:
        clustering = 0.0

    # Connected components
    try:
        components = list(nx.weakly_connected_components(network))
        n_components = len(components)
        largest_component = max(len(c) for c in components) if components else 0
        largest_component_fraction = largest_component / n_nodes if n_nodes > 0 else 0.0
    except:
        n_components = 1
        largest_component_fraction = 1.0

    return {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'density': density,
        'avg_degree': avg_degree,
        'clustering': clustering,
        'n_components': n_components,
        'largest_component_fraction': largest_component_fraction
    }


def run_fragmentation_sweep(
    network: EnergyConstrainedNetwork,
    retention_fractions: List[float],
    method: str = 'random',
    n_replicates: int = 5,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Run a sweep across fragmentation levels.

    Parameters
    ----------
    network : EnergyConstrainedNetwork
        Original network
    retention_fractions : list of float
        Fractions of edges to retain
    method : str
        Fragmentation method
    n_replicates : int
        Number of random fragmentation replicates per level
    seed : int, optional
        Base random seed

    Returns
    -------
    list of dict
        Results for each fragmentation level and replicate
    """
    results = []

    for i, retention in enumerate(retention_fractions):
        for rep in range(n_replicates):
            rep_seed = (seed + i * n_replicates + rep) if seed is not None else None

            fragmented = fragment_network(
                network,
                retention_fraction=retention,
                method=method,
                seed=rep_seed
            )

            metrics = compute_network_metrics(fragmented)
            metrics['retention_fraction'] = retention
            metrics['replicate'] = rep
            metrics['method'] = method

            results.append({
                'network': fragmented,
                'metrics': metrics
            })

    return results
