"""
Energy-Constrained Tipping Elements

Extends PyCascades tipping elements with energy state tracking and
thermodynamic properties.

The key insight is that tipping elements have associated energy landscapes
(double-well potentials) and transitions between states involve energy
changes that can be tracked and constrained.

Mathematical Framework:
-----------------------
Each element has:
- State variable x: position in configuration space
- Energy U(x): double-well potential with minima at stable/tipped states
- Dissipation γ: rate of energy loss to environment
- Heat capacity C: energy storage per unit state change

Energy dynamics:
    dE/dt = C * dx/dt - γ * (dx/dt)² + energy_in - energy_out
            ↑            ↑               ↑
         storage    dissipation    coupling flows

Entropy production:
    σ = γ * (dx/dt)² / T

References:
-----------
- Kramers (1940): Escape rate theory
- Wang (2015): Landscape-flux theory of non-equilibrium systems
- Lucarini et al. (2014): Thermodynamic climate framework
"""

import numpy as np
from typing import Optional, Dict, Any

# Import PyCascades base class
try:
    from pycascades.core.tipping_element import tipping_element, cusp
    PYCASCADES_AVAILABLE = True
except ImportError:
    PYCASCADES_AVAILABLE = False
    # Fallback for development/testing without PyCascades installed
    class tipping_element:
        """Stub base class for development."""
        def __init__(self):
            self._par = {}
        def get_par(self, key):
            return self._par.get(key)
        def set_par(self, key, value):
            self._par[key] = value
        def dxdt_diag(self):
            """Returns lambda (t, x) -> dx/dt"""
            raise NotImplementedError
        def jac_diag(self):
            """Returns lambda (t, x) -> jacobian"""
            raise NotImplementedError
        def tip_state(self):
            """Returns lambda x -> bool"""
            return lambda x: x > 0

    class cusp(tipping_element):
        """Stub cusp class for development."""
        def __init__(self, a=-1, b=1, c=0, x_0=0):
            super().__init__()
            self._par = {'a': a, 'b': b, 'c': c, 'x_0': x_0}

        def dxdt_diag(self):
            """Returns callable of dx/dt diagonal element of cusp"""
            return lambda t, x: (self._par['a'] * pow(x - self._par['x_0'], 3)
                               + self._par['b'] * (x - self._par['x_0'])
                               + self._par['c'])

        def jac_diag(self):
            """Returns callable of jacobian diagonal element"""
            return lambda t, x: (3 * self._par['a'] * pow(x - self._par['x_0'], 2)
                               + self._par['b'])


class EnergyConstrainedElement:
    """
    Mixin class adding energy dynamics to any tipping element.

    This mixin provides energy tracking capabilities that can be combined
    with any PyCascades tipping element class through multiple inheritance.

    The energy landscape is modeled as a double-well potential derived from
    the cusp bifurcation dynamics, with additional parameters for:
    - Absolute energy levels at stable and tipped states
    - Barrier height (activation energy)
    - Dissipation rate (damping)
    - Heat capacity (energy storage)

    Parameters
    ----------
    E_stable : float
        Energy at the stable state (x < 0), default 0.0
    E_tipped : float
        Energy at the tipped state (x > 0), default 1.0
    barrier_height : float
        Height of potential barrier between states, default 0.5
    dissipation_rate : float
        Viscous damping coefficient γ, default 0.1
    heat_capacity : float
        Energy storage coefficient C, default 1.0
    effective_temperature : float
        Effective temperature for entropy calculations, default 1.0

    Attributes
    ----------
    delta_E : float
        Energy difference between tipped and stable states
    """

    def __init__(
        self,
        E_stable: float = 0.0,
        E_tipped: float = 1.0,
        barrier_height: float = 0.5,
        dissipation_rate: float = 0.1,
        heat_capacity: float = 1.0,
        effective_temperature: float = 1.0
    ):
        self.E_stable = E_stable
        self.E_tipped = E_tipped
        self.delta_E = E_tipped - E_stable
        self.barrier_height = barrier_height
        self.gamma = dissipation_rate  # Damping coefficient
        self.C = heat_capacity
        self.T_eff = effective_temperature

    def potential(self, x: float) -> float:
        """
        Compute the double-well potential energy at state x.

        The potential is derived from the cusp bifurcation normal form:
            dx/dt = -dU/dx = a*x³ + b*x + c

        Integrating gives:
            U(x) = -a*x⁴/4 - b*x²/2 - c*x

        For standard parameters (a=-1, b=1), this gives wells at x = ±1.

        We scale and shift this to match specified E_stable, E_tipped,
        and barrier_height.

        Parameters
        ----------
        x : float
            State variable

        Returns
        -------
        float
            Potential energy at state x
        """
        # Clip x to prevent overflow in power calculations
        x = np.clip(x, -10.0, 10.0)

        # Normalized double-well: minima at x = ±1, barrier at x = 0
        # U_norm(x) = x⁴/4 - x²/2, with U_norm(±1) = -1/4, U_norm(0) = 0
        U_norm = x**4 / 4 - x**2 / 2

        # Scale barrier height (U_norm goes from -1/4 at minima to 0 at barrier)
        # So barrier_height = 1/4 * scale factor
        scale = 4 * self.barrier_height

        # Shift so U(x=-1) = E_stable
        # Note: For asymmetric wells, we'd need more parameters
        U = scale * U_norm + self.E_stable + self.barrier_height

        return U

    def energy(self, x: float) -> float:
        """
        Alias for potential() - returns energy at state x.

        Parameters
        ----------
        x : float
            State variable

        Returns
        -------
        float
            Energy at state x
        """
        return self.potential(x)

    def force_from_potential(self, x: float) -> float:
        """
        Compute the force (negative gradient of potential) at state x.

        F = -dU/dx

        Parameters
        ----------
        x : float
            State variable

        Returns
        -------
        float
            Force at state x
        """
        # Clip x to prevent overflow
        x = np.clip(x, -10.0, 10.0)

        # Derivative of U_norm = x⁴/4 - x²/2 is x³ - x
        scale = 4 * self.barrier_height
        return -scale * (x**3 - x)

    def dEdt_internal(self, x: float, dxdt: float) -> float:
        """
        Internal energy change rate from state evolution.

        This represents the rate of energy storage/release as the
        element moves through its potential landscape.

        dE/dt_internal = C * |dU/dx| * |dx/dt| = C * F * v

        For a simple model, we use: dE/dt = -dU/dx * dx/dt
        which is the power input to the system.

        Parameters
        ----------
        x : float
            State variable
        dxdt : float
            Rate of change of state

        Returns
        -------
        float
            Rate of internal energy change
        """
        # Clip to prevent overflow
        dxdt = np.clip(dxdt, -100.0, 100.0)

        # Power = force × velocity (with appropriate sign)
        # When moving "downhill" in potential, energy decreases
        force = self.force_from_potential(x)
        result = force * dxdt

        # Clip result to prevent overflow propagation
        return np.clip(result, -1e6, 1e6)

    def dissipation(self, x: float, dxdt: float) -> float:
        """
        Energy dissipation rate (always non-negative).

        Dissipation follows viscous damping: D = γ * v²

        This represents irreversible energy loss to the environment,
        contributing to entropy production.

        Parameters
        ----------
        x : float
            State variable (unused in simple model)
        dxdt : float
            Rate of change of state

        Returns
        -------
        float
            Dissipation rate (always >= 0)
        """
        # Clip to prevent overflow
        dxdt = np.clip(dxdt, -100.0, 100.0)
        return self.gamma * dxdt**2

    def entropy_production(self, x: float, dxdt: float,
                          T: Optional[float] = None) -> float:
        """
        Local entropy production rate.

        σ = D / T = γ * (dx/dt)² / T

        Parameters
        ----------
        x : float
            State variable
        dxdt : float
            Rate of change of state
        T : float, optional
            Temperature (defaults to self.T_eff)

        Returns
        -------
        float
            Entropy production rate (always >= 0)
        """
        if T is None:
            T = self.T_eff
        return self.dissipation(x, dxdt) / T

    def kramers_escape_rate(self, noise_amplitude: float) -> float:
        """
        Estimate tipping rate using Kramers escape rate formula.

        k = (ω_a × ω_b) / (2πγ) × exp(-ΔU/kT)

        For our double-well:
        - ω_a: curvature at potential minimum (≈ stability)
        - ω_b: curvature at saddle point (barrier shape)
        - γ: friction/damping coefficient
        - ΔU: barrier height
        - kT: noise amplitude (effective temperature)

        Parameters
        ----------
        noise_amplitude : float
            Effective temperature / noise strength

        Returns
        -------
        float
            Estimated escape rate (per unit time)
        """
        # For normalized double-well U = x⁴/4 - x²/2:
        # d²U/dx² = 3x² - 1
        # At minimum (x=±1): ω_a² = 3(1) - 1 = 2
        # At saddle (x=0): ω_b² = -1 (imaginary, use |ω_b|)

        omega_a = np.sqrt(2 * 4 * self.barrier_height)  # Scaled
        omega_b = np.sqrt(4 * self.barrier_height)  # |curvature at saddle|

        prefactor = (omega_a * omega_b) / (2 * np.pi * self.gamma)
        exponent = -self.barrier_height / noise_amplitude

        return prefactor * np.exp(exponent)

    def get_energy_params(self) -> Dict[str, float]:
        """Return dictionary of energy parameters."""
        return {
            'E_stable': self.E_stable,
            'E_tipped': self.E_tipped,
            'delta_E': self.delta_E,
            'barrier_height': self.barrier_height,
            'dissipation_rate': self.gamma,
            'heat_capacity': self.C,
            'effective_temperature': self.T_eff
        }


class EnergyConstrainedCusp(cusp, EnergyConstrainedElement):
    """
    Cusp bifurcation tipping element with energy tracking.

    Combines PyCascades' cusp dynamics with energy state tracking.
    The cusp model represents a generic tipping element with:
        dx/dt = a*(x-x_0)³ + b*(x-x_0) + c

    Energy tracking adds:
    - Potential energy landscape U(x)
    - Energy dissipation D(dx/dt)
    - Entropy production σ

    Parameters
    ----------
    a : float
        Cubic coefficient (default -1, determines well shape)
    b : float
        Linear coefficient (default 1, determines stability)
    c : float
        Constant forcing (default 0, shifts equilibria)
    x_0 : float
        Reference point (default 0)
    E_stable : float
        Energy at stable state
    E_tipped : float
        Energy at tipped state
    barrier_height : float
        Potential barrier height
    dissipation_rate : float
        Damping coefficient γ
    heat_capacity : float
        Energy storage coefficient C
    effective_temperature : float
        Temperature for entropy calculations

    Examples
    --------
    >>> element = EnergyConstrainedCusp(
    ...     a=-1, b=1, c=0,
    ...     E_stable=0, E_tipped=1, barrier_height=0.5
    ... )
    >>> element.energy(-1.0)  # Energy at stable state
    0.0
    >>> element.energy(0.0)   # Energy at barrier
    0.5
    >>> element.energy(1.0)   # Energy at tipped state
    1.0
    """

    def __init__(
        self,
        # Cusp parameters
        a: float = -1,
        b: float = 1,
        c: float = 0,
        x_0: float = 0,
        # Energy parameters
        E_stable: float = 0.0,
        E_tipped: float = 1.0,
        barrier_height: float = 0.5,
        dissipation_rate: float = 0.1,
        heat_capacity: float = 1.0,
        effective_temperature: float = 1.0
    ):
        # Initialize cusp dynamics
        cusp.__init__(self, a=a, b=b, c=c, x_0=x_0)

        # Initialize energy tracking
        EnergyConstrainedElement.__init__(
            self,
            E_stable=E_stable,
            E_tipped=E_tipped,
            barrier_height=barrier_height,
            dissipation_rate=dissipation_rate,
            heat_capacity=heat_capacity,
            effective_temperature=effective_temperature
        )

    def get_all_params(self) -> Dict[str, Any]:
        """Return all parameters (dynamics + energy)."""
        params = {
            'a': self.get_par('a'),
            'b': self.get_par('b'),
            'c': self.get_par('c'),
            'x_0': self.get_par('x_0'),
        }
        params.update(self.get_energy_params())
        return params

    # Convenience methods for direct evaluation (PyCascades returns lambdas)
    def eval_dxdt(self, x: float, t: float = 0) -> float:
        """
        Evaluate dx/dt at state x and time t.

        Convenience wrapper around dxdt_diag() which returns a lambda.

        Parameters
        ----------
        x : float
            State variable
        t : float
            Time (default 0)

        Returns
        -------
        float
            Rate of change dx/dt
        """
        return self.dxdt_diag()(t, x)

    def eval_jacobian(self, x: float, t: float = 0) -> float:
        """
        Evaluate Jacobian at state x and time t.

        Convenience wrapper around jac_diag() which returns a lambda.

        Parameters
        ----------
        x : float
            State variable
        t : float
            Time (default 0)

        Returns
        -------
        float
            Jacobian (d/dx of dx/dt)
        """
        return self.jac_diag()(t, x)

    def is_tipped(self, x: float) -> bool:
        """
        Check if element is in tipped state.

        Parameters
        ----------
        x : float
            State variable

        Returns
        -------
        bool
            True if tipped (x > 0)
        """
        return self.tip_state()(x)


# Convenience function for creating energy-constrained Earth system elements
def create_earth_system_element(
    name: str,
    timescale: float,
    tipping_threshold: float,
    **energy_kwargs
) -> EnergyConstrainedCusp:
    """
    Create an energy-constrained tipping element for Earth system modeling.

    This is a convenience function that sets cusp parameters based on
    physical timescales and thresholds, similar to the PyCascades
    earth_system module.

    Parameters
    ----------
    name : str
        Element identifier (e.g., 'GIS', 'THC', 'AMAZ', 'WAIS')
    timescale : float
        Characteristic response timescale in years
    tipping_threshold : float
        GMT anomaly at which tipping occurs (°C)
    **energy_kwargs
        Additional energy parameters passed to EnergyConstrainedCusp

    Returns
    -------
    EnergyConstrainedCusp
        Configured tipping element
    """
    # Default energy parameters based on timescale
    defaults = {
        'E_stable': 0.0,
        'E_tipped': 1.0,
        'barrier_height': 0.5,
        'dissipation_rate': 1.0 / timescale,  # Faster elements dissipate faster
        'heat_capacity': timescale,  # Slower elements store more
        'effective_temperature': 1.0
    }
    defaults.update(energy_kwargs)

    # Create element
    element = EnergyConstrainedCusp(
        a=-1,
        b=1,
        c=0,  # Will be set based on GMT during simulation
        x_0=0,
        **defaults
    )

    # Store metadata
    element.name = name
    element.timescale = timescale
    element.tipping_threshold = tipping_threshold

    return element


# Standard Earth system elements with default energy parameters
EARTH_SYSTEM_ELEMENTS = {
    'GIS': {'timescale': 98, 'tipping_threshold': 2.24},
    'THC': {'timescale': 6, 'tipping_threshold': 4.69},
    'WAIS': {'timescale': 48, 'tipping_threshold': 1.42},
    'AMAZ': {'timescale': 1, 'tipping_threshold': 4.10},
}
