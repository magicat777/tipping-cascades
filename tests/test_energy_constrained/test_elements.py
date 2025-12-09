"""
Unit tests for energy-constrained tipping elements.
"""

import numpy as np
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from energy_constrained.elements import (
    EnergyConstrainedElement,
    EnergyConstrainedCusp,
    create_earth_system_element,
    EARTH_SYSTEM_ELEMENTS
)


class TestEnergyConstrainedElement:
    """Tests for the EnergyConstrainedElement mixin."""

    def test_initialization(self):
        """Test default initialization."""
        elem = EnergyConstrainedElement()
        assert elem.E_stable == 0.0
        assert elem.E_tipped == 1.0
        assert elem.delta_E == 1.0
        assert elem.barrier_height == 0.5
        assert elem.gamma == 0.1
        assert elem.C == 1.0
        assert elem.T_eff == 1.0

    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        elem = EnergyConstrainedElement(
            E_stable=-1.0,
            E_tipped=2.0,
            barrier_height=1.5,
            dissipation_rate=0.2,
            heat_capacity=2.0,
            effective_temperature=0.5
        )
        assert elem.E_stable == -1.0
        assert elem.E_tipped == 2.0
        assert elem.delta_E == 3.0
        assert elem.barrier_height == 1.5
        assert elem.gamma == 0.2
        assert elem.C == 2.0
        assert elem.T_eff == 0.5

    def test_potential_double_well(self):
        """Test that potential has correct double-well shape."""
        elem = EnergyConstrainedElement(barrier_height=0.5)

        # Minima should be at approximately x = ±1
        U_min_left = elem.potential(-1.0)
        U_min_right = elem.potential(1.0)
        U_barrier = elem.potential(0.0)

        # Barrier should be higher than minima
        assert U_barrier > U_min_left
        assert U_barrier > U_min_right

        # Barrier height should be approximately correct
        assert abs(U_barrier - U_min_left - elem.barrier_height) < 0.1

    def test_force_from_potential(self):
        """Test force at equilibria and barrier."""
        elem = EnergyConstrainedElement(barrier_height=0.5)

        # Force should be near zero at equilibria (x = ±1)
        assert abs(elem.force_from_potential(-1.0)) < 0.1
        assert abs(elem.force_from_potential(1.0)) < 0.1

        # Force should be near zero at saddle (x = 0)
        assert abs(elem.force_from_potential(0.0)) < 0.1

    def test_dissipation_always_positive(self):
        """Test that dissipation is always non-negative."""
        elem = EnergyConstrainedElement(dissipation_rate=0.1)

        # Test various velocities
        for dxdt in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            D = elem.dissipation(0.0, dxdt)
            assert D >= 0

        # Test that dissipation increases with velocity squared
        D_slow = elem.dissipation(0.0, 1.0)
        D_fast = elem.dissipation(0.0, 2.0)
        assert D_fast > D_slow
        assert abs(D_fast / D_slow - 4.0) < 0.01  # Should be 4x

    def test_entropy_production(self):
        """Test entropy production is dissipation / temperature."""
        elem = EnergyConstrainedElement(
            dissipation_rate=0.1,
            effective_temperature=2.0
        )

        dxdt = 1.5
        sigma = elem.entropy_production(0.0, dxdt)
        D = elem.dissipation(0.0, dxdt)

        assert abs(sigma - D / 2.0) < 1e-10

    def test_kramers_rate_scaling(self):
        """Test Kramers rate decreases with barrier height."""
        elem_low = EnergyConstrainedElement(barrier_height=0.5)
        elem_high = EnergyConstrainedElement(barrier_height=1.0)

        noise = 0.1
        rate_low = elem_low.kramers_escape_rate(noise)
        rate_high = elem_high.kramers_escape_rate(noise)

        # Higher barrier = lower escape rate
        assert rate_high < rate_low

    def test_get_energy_params(self):
        """Test parameter dictionary."""
        elem = EnergyConstrainedElement(
            E_stable=0.5,
            barrier_height=1.0
        )
        params = elem.get_energy_params()

        assert 'E_stable' in params
        assert 'barrier_height' in params
        assert params['E_stable'] == 0.5
        assert params['barrier_height'] == 1.0


class TestEnergyConstrainedCusp:
    """Tests for the EnergyConstrainedCusp class."""

    def test_initialization(self):
        """Test combined initialization."""
        cusp_elem = EnergyConstrainedCusp(
            a=-1, b=1, c=0, x_0=0,
            E_stable=0, E_tipped=1, barrier_height=0.5
        )

        # Check cusp parameters
        assert cusp_elem.get_par('a') == -1
        assert cusp_elem.get_par('b') == 1

        # Check energy parameters
        assert cusp_elem.E_stable == 0
        assert cusp_elem.E_tipped == 1

    def test_cusp_dynamics(self):
        """Test cusp bifurcation dynamics."""
        cusp_elem = EnergyConstrainedCusp(a=-1, b=1, c=0)

        # Equilibria at x = ±1 (approximately)
        # dxdt = -x³ + x = -x(x² - 1) = 0 at x = 0, ±1

        dxdt_stable = cusp_elem.eval_dxdt(-1.0)
        dxdt_unstable = cusp_elem.eval_dxdt(0.0)
        dxdt_tipped = cusp_elem.eval_dxdt(1.0)

        assert abs(dxdt_stable) < 0.01
        assert abs(dxdt_unstable) < 0.01
        assert abs(dxdt_tipped) < 0.01

    def test_jacobian(self):
        """Test Jacobian computation."""
        cusp_elem = EnergyConstrainedCusp(a=-1, b=1)

        # Jacobian: d/dx(-x³ + x) = -3x² + 1
        # At x=0: jac = 1 (unstable)
        # At x=±1: jac = -3 + 1 = -2 (stable)

        jac_unstable = cusp_elem.eval_jacobian(0.0)
        jac_stable = cusp_elem.eval_jacobian(1.0)

        assert jac_unstable > 0  # Positive = unstable
        assert jac_stable < 0   # Negative = stable

    def test_tip_state(self):
        """Test tipping state detection."""
        cusp_elem = EnergyConstrainedCusp()

        assert not cusp_elem.is_tipped(-1.0)
        assert cusp_elem.is_tipped(1.0)
        assert not cusp_elem.is_tipped(-0.5)
        assert cusp_elem.is_tipped(0.5)

    def test_get_all_params(self):
        """Test combined parameter dictionary."""
        cusp_elem = EnergyConstrainedCusp(a=-1, b=2, E_stable=0.5)
        params = cusp_elem.get_all_params()

        assert params['a'] == -1
        assert params['b'] == 2
        assert params['E_stable'] == 0.5

    def test_energy_along_trajectory(self):
        """Test energy changes during state evolution."""
        cusp_elem = EnergyConstrainedCusp(
            E_stable=0, E_tipped=1, barrier_height=0.5
        )

        # Energy at stable state
        E_stable = cusp_elem.energy(-1.0)

        # Energy at barrier
        E_barrier = cusp_elem.energy(0.0)

        # Energy at tipped state
        E_tipped = cusp_elem.energy(1.0)

        # Check ordering
        assert E_barrier > E_stable
        assert E_barrier > E_tipped


class TestEarthSystemElements:
    """Tests for Earth system convenience functions."""

    def test_earth_system_element_creation(self):
        """Test creating Earth system elements."""
        gis = create_earth_system_element('GIS', timescale=98, tipping_threshold=2.24)

        assert gis.name == 'GIS'
        assert gis.timescale == 98
        assert gis.tipping_threshold == 2.24

        # Dissipation should scale with timescale
        assert gis.gamma == 1.0 / 98

    def test_all_standard_elements(self):
        """Test all standard Earth system elements."""
        for name, params in EARTH_SYSTEM_ELEMENTS.items():
            elem = create_earth_system_element(name, **params)
            assert elem.name == name
            assert elem.timescale == params['timescale']
            assert elem.tipping_threshold == params['tipping_threshold']

    def test_custom_energy_params(self):
        """Test passing custom energy parameters."""
        amaz = create_earth_system_element(
            'AMAZ',
            timescale=1,
            tipping_threshold=4.10,
            barrier_height=0.8,
            E_tipped=2.0
        )

        assert amaz.barrier_height == 0.8
        assert amaz.E_tipped == 2.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
