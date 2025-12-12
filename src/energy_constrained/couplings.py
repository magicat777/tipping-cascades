"""
Energy-Based Coupling Functions

Extends PyCascades coupling mechanisms with thermodynamically-motivated
energy transfer between tipping elements.

Key insight from Phase 2: Coupling asymmetry stabilizes cascades.
This module explores whether energy-based constraints naturally produce
such asymmetry.

Coupling Types:
---------------
1. GradientDrivenCoupling: Energy flows from high to low potential
   - Strength proportional to energy gradient
   - Naturally asymmetric in non-equilibrium states

2. AsymmetricEnergyCoupling: Different rates for uphill/downhill transfer
   - Thermodynamic favorability determines direction
   - Models activation energy for unfavorable transitions

3. DissipativeCoupling: Energy transfer with losses
   - Some transferred energy is dissipated
   - Entropy production from inter-element transfer

Mathematical Framework:
-----------------------
Standard linear coupling: dx_i/dt += k * (x_j - x_i)

Energy-based coupling:
    dx_i/dt += f(E_j - E_i, x_j, x_i)
    dE_i/dt += Φ_ji - Φ_ij - D_coupling

where Φ_ij is the energy flow rate from i to j.

References:
-----------
- Phase 2 findings: Asymmetric coupling protects GIS from cascading
- Amazon moisture recycling: Coupling can be protective
- Network dissipation principles (Pereira et al. 2015)
"""

import numpy as np
from typing import Optional, Callable, Tuple

# Import PyCascades base class
try:
    from pycascades.core.coupling import coupling, linear_coupling
except ImportError:
    # Fallback for development/testing
    class coupling:
        """Stub base class for development."""
        def dxdt_cpl(self, x_from, x_to):
            raise NotImplementedError
        def jac_cpl(self, x_from, x_to):
            raise NotImplementedError
        def jac_diag(self, x_from, x_to):
            raise NotImplementedError

    class linear_coupling(coupling):
        """Stub linear coupling for development."""
        def __init__(self, strength=1.0):
            self.strength = strength

        def dxdt_cpl(self, x_from, x_to):
            return self.strength * (x_from - x_to)

        def jac_cpl(self, x_from, x_to):
            return self.strength

        def jac_diag(self, x_from, x_to):
            return -self.strength


class EnergyCoupling(coupling):
    """
    Base class for energy-aware couplings.

    Extends the standard coupling interface with methods for
    computing energy transfer rates between elements.

    Subclasses must implement:
    - energy_transfer_rate(): Rate of energy flow between elements
    - dxdt_cpl(): How energy gradient affects state dynamics
    """

    def energy_transfer_rate(
        self,
        x_from: float,
        x_to: float,
        E_from: float,
        E_to: float
    ) -> float:
        """
        Compute rate of energy transfer between elements.

        Positive value means energy flows FROM the 'from' element
        TO the 'to' element.

        Parameters
        ----------
        x_from : float
            State of source element
        x_to : float
            State of destination element
        E_from : float
            Energy of source element
        E_to : float
            Energy of destination element

        Returns
        -------
        float
            Energy transfer rate (positive = flow to destination)
        """
        raise NotImplementedError

    def dissipation_in_transfer(
        self,
        x_from: float,
        x_to: float,
        E_from: float,
        E_to: float
    ) -> float:
        """
        Energy dissipated during transfer (default: none).

        Subclasses can override to model lossy coupling.

        Returns
        -------
        float
            Energy dissipation rate (always >= 0)
        """
        return 0.0


class GradientDrivenCoupling(EnergyCoupling):
    """
    Coupling strength proportional to energy gradient.

    Energy naturally flows from high to low potential, analogous to
    heat conduction. The coupling strength is modulated by the
    energy difference between elements.

    This creates natural asymmetry: a tipped element (higher energy)
    more strongly influences an untipped neighbor than vice versa.

    Parameters
    ----------
    conductivity : float
        Base coupling strength (energy transfer coefficient)
    state_coupling : float
        Additional direct state coupling (default 0)

    Notes
    -----
    Energy flow: Φ = k * (E_from - E_to)
    State effect: dx_to/dt += f(Φ) where f depends on implementation

    For simple case: dx/dt += (k/C) * (E_from - E_to)
    where C is heat capacity of receiving element.
    """

    def __init__(
        self,
        conductivity: float = 0.1,
        state_coupling: float = 0.0
    ):
        self.k = conductivity
        self.k_state = state_coupling

    def energy_transfer_rate(
        self,
        x_from: float,
        x_to: float,
        E_from: float,
        E_to: float
    ) -> float:
        """
        Energy flows from high to low potential.

        Φ = k * (E_from - E_to)

        Parameters
        ----------
        x_from, x_to : float
            State variables (unused in gradient-only model)
        E_from, E_to : float
            Element energies

        Returns
        -------
        float
            Energy transfer rate to destination
        """
        # Handle NaN/Inf values
        if not (np.isfinite(E_from) and np.isfinite(E_to)):
            return 0.0

        delta_E = E_from - E_to
        return self.k * delta_E

    def dxdt_cpl(
        self,
        x_from: float,
        x_to: float,
        E_from: Optional[float] = None,
        E_to: Optional[float] = None
    ) -> float:
        """
        State change driven by energy gradient.

        If energies are provided, coupling is gradient-driven.
        Otherwise, falls back to simple linear coupling.

        Parameters
        ----------
        x_from, x_to : float
            State variables
        E_from, E_to : float, optional
            Element energies

        Returns
        -------
        float
            Rate of change contribution to x_to
        """
        # Clip inputs to prevent overflow
        x_from = np.clip(x_from, -10.0, 10.0)
        x_to = np.clip(x_to, -10.0, 10.0)

        # Direct state coupling (like standard linear)
        state_effect = self.k_state * (x_from - x_to)

        # Energy gradient effect
        if E_from is not None and E_to is not None:
            # Handle NaN/Inf
            if np.isfinite(E_from) and np.isfinite(E_to):
                energy_effect = self.k * (E_from - E_to)
            else:
                energy_effect = 0.0
        else:
            # Fallback: use state difference as proxy for energy
            energy_effect = self.k * (x_from - x_to)

        return state_effect + energy_effect

    def jac_cpl(self, x_from: float, x_to: float) -> float:
        """Jacobian entry for coupling (d/dx_from of dxdt_cpl)."""
        return self.k_state + self.k

    def jac_diag(self, x_from: float, x_to: float) -> float:
        """Jacobian entry for diagonal (d/dx_to of dxdt_cpl)."""
        return -(self.k_state + self.k)


class AsymmetricEnergyCoupling(EnergyCoupling):
    """
    Asymmetric coupling based on thermodynamic favorability.

    Different coupling strengths for "downhill" (energetically favorable)
    vs "uphill" (unfavorable) energy transfer. This models the fact that
    cascades should flow more easily toward lower-energy states.

    The asymmetry ratio (k_forward / k_backward) determines how much
    easier it is for perturbations to cascade in the favorable direction.

    Parameters
    ----------
    k_forward : float
        Coupling strength for favorable (downhill) transfers
    k_backward : float
        Coupling strength for unfavorable (uphill) transfers
    threshold : float
        Energy difference threshold for asymmetry (default 0)

    Notes
    -----
    This may naturally emerge from activation energy considerations:
    unfavorable transfers require overcoming an additional barrier.

    Phase 2 finding: The original Wunderling network has asymmetric
    couplings that protect GIS. This class allows testing whether
    such asymmetry has thermodynamic origins.
    """

    def __init__(
        self,
        k_forward: float = 0.1,
        k_backward: float = 0.01,
        threshold: float = 0.0
    ):
        self.k_fwd = k_forward
        self.k_bwd = k_backward
        self.threshold = threshold

    @property
    def asymmetry_ratio(self) -> float:
        """Ratio of forward to backward coupling strength."""
        if self.k_bwd > 0:
            return self.k_fwd / self.k_bwd
        return float('inf')

    def energy_transfer_rate(
        self,
        x_from: float,
        x_to: float,
        E_from: float,
        E_to: float
    ) -> float:
        """
        Asymmetric energy transfer: easier to cascade 'downhill'.

        When E_from > E_to + threshold: use k_forward (favorable)
        Otherwise: use k_backward (unfavorable, slower)

        Parameters
        ----------
        x_from, x_to : float
            State variables
        E_from, E_to : float
            Element energies

        Returns
        -------
        float
            Energy transfer rate (can be positive or negative)
        """
        delta_E = E_from - E_to

        if delta_E > self.threshold:
            # Favorable: energy flows to lower potential
            return self.k_fwd * delta_E
        else:
            # Unfavorable: reduced coupling
            return self.k_bwd * delta_E

    def dxdt_cpl(
        self,
        x_from: float,
        x_to: float,
        E_from: Optional[float] = None,
        E_to: Optional[float] = None
    ) -> float:
        """
        State coupling with asymmetric energy effects.

        Parameters
        ----------
        x_from, x_to : float
            State variables
        E_from, E_to : float, optional
            Element energies

        Returns
        -------
        float
            Rate of change contribution to x_to
        """
        if E_from is not None and E_to is not None:
            return self.energy_transfer_rate(x_from, x_to, E_from, E_to)
        else:
            # Fallback: linear with average strength
            k_avg = (self.k_fwd + self.k_bwd) / 2
            return k_avg * (x_from - x_to)

    def jac_cpl(self, x_from: float, x_to: float) -> float:
        """Jacobian (approximate - uses average strength)."""
        k_avg = (self.k_fwd + self.k_bwd) / 2
        return k_avg

    def jac_diag(self, x_from: float, x_to: float) -> float:
        """Diagonal Jacobian (approximate)."""
        k_avg = (self.k_fwd + self.k_bwd) / 2
        return -k_avg


class DissipativeCoupling(EnergyCoupling):
    """
    Energy transfer with dissipation losses.

    Some energy is lost during transfer between elements, contributing
    to total entropy production. This models realistic coupling where
    intermediate processes dissipate energy.

    Parameters
    ----------
    conductivity : float
        Base coupling strength
    efficiency : float
        Fraction of transferred energy reaching destination (0 < η ≤ 1)

    Notes
    -----
    Energy balance:
        Energy sent = Φ_ij
        Energy received = η * Φ_ij
        Energy dissipated = (1 - η) * |Φ_ij|

    Low efficiency couplings dissipate more, but also transfer less
    "cascade signal" - potentially acting as firewalls.
    """

    def __init__(
        self,
        conductivity: float = 0.1,
        efficiency: float = 0.9
    ):
        self.k = conductivity
        self.eta = np.clip(efficiency, 0.01, 1.0)  # Prevent division by zero

    def energy_transfer_rate(
        self,
        x_from: float,
        x_to: float,
        E_from: float,
        E_to: float
    ) -> float:
        """
        Energy received by destination (after losses).

        Net transfer = η * k * (E_from - E_to)
        """
        delta_E = E_from - E_to
        return self.eta * self.k * delta_E

    def dissipation_in_transfer(
        self,
        x_from: float,
        x_to: float,
        E_from: float,
        E_to: float
    ) -> float:
        """
        Energy dissipated during transfer.

        D = (1 - η) * |k * (E_from - E_to)|
        """
        delta_E = E_from - E_to
        gross_transfer = self.k * delta_E
        return (1 - self.eta) * np.abs(gross_transfer)

    def dxdt_cpl(
        self,
        x_from: float,
        x_to: float,
        E_from: Optional[float] = None,
        E_to: Optional[float] = None
    ) -> float:
        """State coupling (uses effective conductivity)."""
        k_eff = self.eta * self.k
        if E_from is not None and E_to is not None:
            return k_eff * (E_from - E_to)
        else:
            return k_eff * (x_from - x_to)

    def jac_cpl(self, x_from: float, x_to: float) -> float:
        return self.eta * self.k

    def jac_diag(self, x_from: float, x_to: float) -> float:
        return -self.eta * self.k


class ActivationBarrierCoupling(EnergyCoupling):
    """
    Coupling with explicit activation barrier for unfavorable transfers.

    Models Kramers-like activation: unfavorable transfers require
    thermal activation over an energy barrier.

    Rate = k * exp(-E_barrier / T) when E_from < E_to
    Rate = k                       when E_from >= E_to

    This creates strong asymmetry near tipping thresholds.

    Parameters
    ----------
    conductivity : float
        Base coupling strength
    activation_energy : float
        Barrier height for unfavorable transfers
    temperature : float
        Effective temperature (noise amplitude)
    """

    def __init__(
        self,
        conductivity: float = 0.1,
        activation_energy: float = 0.5,
        temperature: float = 1.0
    ):
        self.k = conductivity
        self.E_act = activation_energy
        self.T = temperature

    def energy_transfer_rate(
        self,
        x_from: float,
        x_to: float,
        E_from: float,
        E_to: float
    ) -> float:
        """
        Energy transfer with activation barrier for uphill.

        Favorable (downhill): Φ = k * ΔE
        Unfavorable (uphill): Φ = k * ΔE * exp(-E_act / T)
        """
        delta_E = E_from - E_to

        if delta_E >= 0:
            # Favorable: no barrier
            return self.k * delta_E
        else:
            # Unfavorable: activated
            activation = np.exp(-self.E_act / self.T)
            return self.k * delta_E * activation

    def dxdt_cpl(
        self,
        x_from: float,
        x_to: float,
        E_from: Optional[float] = None,
        E_to: Optional[float] = None
    ) -> float:
        """State coupling with activation."""
        if E_from is not None and E_to is not None:
            return self.energy_transfer_rate(x_from, x_to, E_from, E_to)
        else:
            # Without energies, use state as proxy
            return self.k * (x_from - x_to)

    def jac_cpl(self, x_from: float, x_to: float) -> float:
        return self.k

    def jac_diag(self, x_from: float, x_to: float) -> float:
        return -self.k


class StateDependentCoupling(EnergyCoupling):
    """
    Coupling where strength depends on source element state.

    Models the biological reality that tipped/degraded elements provide
    less support to their neighbors:
    - Dead forest can't recycle moisture to neighbors
    - Collapsed ecosystems can't provide services to adjacent regions
    - Fire-damaged areas can't buffer neighboring forests

    When the source element is tipped (x > 0), coupling strength is
    reduced by degradation_factor.

    Parameters
    ----------
    base_conductivity : float
        Coupling strength when source is in stable state (default 0.1)
    degradation_factor : float
        Multiplier when source is tipped, 0 < factor < 1 (default 0.3)
        e.g., 0.3 means tipped cells provide 30% of normal support
    state_threshold : float
        State value above which element is considered tipped (default 0)

    Notes
    -----
    This creates a positive feedback loop:
    1. Cell A tips
    2. Cell A provides less support to neighbor B
    3. Cell B is more likely to tip
    4. Cascade propagates more easily than it recovers

    References
    ----------
    - Westen (2023): AMOC hysteresis from state-dependent feedbacks
    - Amazon moisture recycling: degraded forest reduces evapotranspiration
    """

    def __init__(
        self,
        base_conductivity: float = 0.1,
        degradation_factor: float = 0.3,
        state_threshold: float = 0.0
    ):
        self.k_base = base_conductivity
        self.degradation = np.clip(degradation_factor, 0.0, 1.0)
        self.threshold = state_threshold

    def effective_conductivity(self, x_from: float) -> float:
        """
        Compute effective coupling strength based on source state.

        Parameters
        ----------
        x_from : float
            State of source element

        Returns
        -------
        float
            Effective coupling strength
        """
        if x_from > self.threshold:
            # Source is tipped: reduced coupling
            return self.k_base * self.degradation
        else:
            # Source is stable: full coupling
            return self.k_base

    def energy_transfer_rate(
        self,
        x_from: float,
        x_to: float,
        E_from: float,
        E_to: float
    ) -> float:
        """
        Energy transfer with state-dependent conductivity.

        Φ = k_eff(x_from) * (E_from - E_to)

        Parameters
        ----------
        x_from, x_to : float
            State variables
        E_from, E_to : float
            Element energies

        Returns
        -------
        float
            Energy transfer rate
        """
        if not (np.isfinite(E_from) and np.isfinite(E_to)):
            return 0.0

        k_eff = self.effective_conductivity(x_from)
        return k_eff * (E_from - E_to)

    def dxdt_cpl(
        self,
        x_from: float,
        x_to: float,
        E_from: Optional[float] = None,
        E_to: Optional[float] = None
    ) -> float:
        """
        State coupling with state-dependent strength.

        Parameters
        ----------
        x_from, x_to : float
            State variables
        E_from, E_to : float, optional
            Element energies

        Returns
        -------
        float
            Rate of change contribution to x_to
        """
        x_from = np.clip(x_from, -10.0, 10.0)
        x_to = np.clip(x_to, -10.0, 10.0)

        k_eff = self.effective_conductivity(x_from)

        if E_from is not None and E_to is not None:
            if np.isfinite(E_from) and np.isfinite(E_to):
                return k_eff * (E_from - E_to)
            return 0.0
        else:
            # Fallback: state-based coupling
            return k_eff * (x_from - x_to)

    def jac_cpl(self, x_from: float, x_to: float) -> float:
        """Jacobian (uses current effective conductivity)."""
        return self.effective_conductivity(x_from)

    def jac_diag(self, x_from: float, x_to: float) -> float:
        """Diagonal Jacobian."""
        return -self.effective_conductivity(x_from)


class TrendForcingCoupling(EnergyCoupling):
    """
    Coupling that includes time-varying climate trend forcing.

    Models unidirectional climate change as a drift term that
    pushes the system toward tipping over time.

    This is not technically a "coupling" but uses the coupling
    interface to inject time-dependent forcing into the dynamics.

    Parameters
    ----------
    initial_forcing : float
        Initial value of forcing parameter c (default 0)
    trend_rate : float
        Rate of forcing change per time unit (default 0.001)
        Positive = toward tipping
    max_forcing : float
        Maximum forcing value (caps the trend, default 1.0)

    Notes
    -----
    The forcing c(t) shifts the cusp potential:
        dx/dt = a*x³ + b*x + c(t)

    Positive c pushes the stable equilibrium toward x=0 (the barrier),
    making tipping more likely over time.

    To use: add as a "self-coupling" where source = destination,
    then the dxdt_cpl contribution acts as external forcing.
    """

    def __init__(
        self,
        initial_forcing: float = 0.0,
        trend_rate: float = 0.001,
        max_forcing: float = 1.0
    ):
        self.c_0 = initial_forcing
        self.trend = trend_rate
        self.c_max = max_forcing
        self._current_time = 0.0

    def set_time(self, t: float) -> None:
        """Update current time for forcing calculation."""
        self._current_time = t

    def get_forcing(self, t: Optional[float] = None) -> float:
        """
        Compute forcing at given time.

        Parameters
        ----------
        t : float, optional
            Time (uses stored time if not provided)

        Returns
        -------
        float
            Forcing value c(t)
        """
        if t is None:
            t = self._current_time

        c = self.c_0 + self.trend * t
        return np.clip(c, -self.c_max, self.c_max)

    def energy_transfer_rate(
        self,
        x_from: float,
        x_to: float,
        E_from: float,
        E_to: float
    ) -> float:
        """Not used for trend forcing."""
        return 0.0

    def dxdt_cpl(
        self,
        x_from: float,
        x_to: float,
        E_from: Optional[float] = None,
        E_to: Optional[float] = None,
        t: Optional[float] = None
    ) -> float:
        """
        Return forcing contribution to dynamics.

        When used as self-coupling, this adds c(t) to dx/dt.

        Parameters
        ----------
        x_from, x_to : float
            State variables (ignored)
        E_from, E_to : float, optional
            Energies (ignored)
        t : float, optional
            Time for forcing calculation

        Returns
        -------
        float
            Forcing value c(t)
        """
        return self.get_forcing(t)

    def jac_cpl(self, x_from: float, x_to: float) -> float:
        """No state dependence in forcing."""
        return 0.0

    def jac_diag(self, x_from: float, x_to: float) -> float:
        """No state dependence in forcing."""
        return 0.0


# Factory function for creating Earth system couplings
def create_coupling_matrix(
    coupling_type: str = 'gradient',
    base_strengths: Optional[np.ndarray] = None,
    **kwargs
) -> Callable[[int, int], EnergyCoupling]:
    """
    Create a factory for energy-aware couplings.

    Parameters
    ----------
    coupling_type : str
        One of: 'gradient', 'asymmetric', 'dissipative', 'activation'
    base_strengths : array-like, optional
        Matrix of base coupling strengths (uses standard Wunderling if None)
    **kwargs
        Additional parameters passed to coupling constructor

    Returns
    -------
    callable
        Function(i, j) -> EnergyCoupling for edge (i, j)

    Examples
    --------
    >>> factory = create_coupling_matrix('asymmetric', k_forward=0.2)
    >>> coupling = factory(0, 1)  # Coupling from element 0 to 1
    """
    # Default: Wunderling coupling strengths
    if base_strengths is None:
        base_strengths = np.array([
            [0.000, 0.312, 0.271, 0.000],
            [0.578, 0.000, 0.132, 0.372],
            [0.193, 0.120, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000]
        ])

    coupling_classes = {
        'gradient': GradientDrivenCoupling,
        'asymmetric': AsymmetricEnergyCoupling,
        'dissipative': DissipativeCoupling,
        'activation': ActivationBarrierCoupling,
    }

    if coupling_type not in coupling_classes:
        raise ValueError(f"Unknown coupling type: {coupling_type}")

    CouplingClass = coupling_classes[coupling_type]

    def factory(i: int, j: int) -> EnergyCoupling:
        strength = base_strengths[i, j]
        if coupling_type == 'gradient':
            return CouplingClass(conductivity=strength, **kwargs)
        elif coupling_type == 'asymmetric':
            # Scale k_forward and k_backward by base strength
            return CouplingClass(
                k_forward=strength * kwargs.get('k_forward', 1.0),
                k_backward=strength * kwargs.get('k_backward', 0.1),
                threshold=kwargs.get('threshold', 0.0)
            )
        elif coupling_type == 'dissipative':
            return CouplingClass(conductivity=strength, **kwargs)
        elif coupling_type == 'activation':
            return CouplingClass(conductivity=strength, **kwargs)

    return factory
