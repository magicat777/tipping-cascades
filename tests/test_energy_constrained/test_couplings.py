"""
Unit tests for energy-based coupling functions.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from energy_constrained.couplings import (
    EnergyCoupling,
    GradientDrivenCoupling,
    AsymmetricEnergyCoupling,
    DissipativeCoupling,
    ActivationBarrierCoupling,
    create_coupling_matrix
)


class TestGradientDrivenCoupling:
    """Tests for gradient-driven coupling."""

    def test_initialization(self):
        """Test default initialization."""
        coupling = GradientDrivenCoupling()
        assert coupling.k == 0.1
        assert coupling.k_state == 0.0

    def test_energy_flows_downhill(self):
        """Test that energy flows from high to low."""
        coupling = GradientDrivenCoupling(conductivity=0.5)

        # High energy to low energy
        rate = coupling.energy_transfer_rate(0, 0, E_from=2.0, E_to=1.0)
        assert rate > 0  # Positive = flow to destination

        # Low to high
        rate = coupling.energy_transfer_rate(0, 0, E_from=1.0, E_to=2.0)
        assert rate < 0  # Negative = flow from destination

    def test_zero_transfer_at_equilibrium(self):
        """Test no transfer when energies equal."""
        coupling = GradientDrivenCoupling(conductivity=0.5)
        rate = coupling.energy_transfer_rate(0, 0, E_from=1.5, E_to=1.5)
        assert rate == 0

    def test_linear_in_gradient(self):
        """Test transfer rate is linear in energy difference."""
        coupling = GradientDrivenCoupling(conductivity=0.2)

        rate1 = coupling.energy_transfer_rate(0, 0, E_from=2.0, E_to=1.0)  # ΔE = 1
        rate2 = coupling.energy_transfer_rate(0, 0, E_from=3.0, E_to=1.0)  # ΔE = 2

        assert abs(rate2 / rate1 - 2.0) < 0.01

    def test_dxdt_cpl_with_energy(self):
        """Test state coupling with energy information."""
        coupling = GradientDrivenCoupling(conductivity=0.1)

        effect = coupling.dxdt_cpl(
            x_from=0.5, x_to=-0.5,
            E_from=1.5, E_to=0.5
        )
        # Energy difference of 1.0, k=0.1, so effect should be 0.1
        assert abs(effect - 0.1) < 0.01

    def test_dxdt_cpl_fallback(self):
        """Test fallback to state-based coupling."""
        coupling = GradientDrivenCoupling(conductivity=0.1)

        # Without energy, should use state difference
        effect = coupling.dxdt_cpl(x_from=1.0, x_to=0.0)
        assert abs(effect - 0.1) < 0.01


class TestAsymmetricEnergyCoupling:
    """Tests for asymmetric energy coupling."""

    def test_initialization(self):
        """Test default initialization."""
        coupling = AsymmetricEnergyCoupling()
        assert coupling.k_fwd == 0.1
        assert coupling.k_bwd == 0.01
        assert coupling.threshold == 0.0

    def test_asymmetry_ratio(self):
        """Test asymmetry ratio calculation."""
        coupling = AsymmetricEnergyCoupling(k_forward=0.5, k_backward=0.1)
        assert coupling.asymmetry_ratio == 5.0

    def test_favorable_direction_faster(self):
        """Test that favorable transfers use faster rate."""
        coupling = AsymmetricEnergyCoupling(
            k_forward=0.5,
            k_backward=0.1,
            threshold=0
        )

        # Favorable: E_from > E_to
        rate_fwd = coupling.energy_transfer_rate(0, 0, E_from=2.0, E_to=1.0)

        # Unfavorable: E_from < E_to
        rate_bwd = coupling.energy_transfer_rate(0, 0, E_from=1.0, E_to=2.0)

        # Favorable should be 5x stronger (per unit ΔE)
        # rate_fwd = 0.5 * 1 = 0.5
        # rate_bwd = 0.1 * (-1) = -0.1
        assert rate_fwd == 0.5
        assert rate_bwd == -0.1

    def test_threshold_effect(self):
        """Test threshold affects which rate is used."""
        coupling = AsymmetricEnergyCoupling(
            k_forward=0.5,
            k_backward=0.1,
            threshold=0.5
        )

        # Small favorable gradient (below threshold)
        rate_small = coupling.energy_transfer_rate(0, 0, E_from=1.3, E_to=1.0)
        # Should use backward rate: 0.1 * 0.3 = 0.03
        assert abs(rate_small - 0.03) < 0.01

        # Large favorable gradient (above threshold)
        rate_large = coupling.energy_transfer_rate(0, 0, E_from=2.0, E_to=1.0)
        # Should use forward rate: 0.5 * 1.0 = 0.5
        assert abs(rate_large - 0.5) < 0.01


class TestDissipativeCoupling:
    """Tests for dissipative coupling."""

    def test_initialization(self):
        """Test default initialization."""
        coupling = DissipativeCoupling()
        assert coupling.k == 0.1
        assert coupling.eta == 0.9

    def test_efficiency_bounds(self):
        """Test efficiency is bounded to [0.01, 1.0]."""
        coupling_low = DissipativeCoupling(efficiency=0.0)
        coupling_high = DissipativeCoupling(efficiency=1.5)

        assert coupling_low.eta == 0.01  # Clipped to minimum
        assert coupling_high.eta == 1.0  # Clipped to maximum

    def test_energy_loss_in_transfer(self):
        """Test that some energy is dissipated."""
        coupling = DissipativeCoupling(conductivity=0.5, efficiency=0.8)

        E_from, E_to = 2.0, 1.0
        received = coupling.energy_transfer_rate(0, 0, E_from, E_to)
        dissipated = coupling.dissipation_in_transfer(0, 0, E_from, E_to)

        # Gross transfer = 0.5 * 1.0 = 0.5
        # Received = 0.8 * 0.5 = 0.4
        # Dissipated = 0.2 * 0.5 = 0.1
        assert abs(received - 0.4) < 0.01
        assert abs(dissipated - 0.1) < 0.01

        # Check conservation: received + dissipated = gross
        gross = coupling.k * (E_from - E_to)
        assert abs(received + dissipated - gross) < 0.01

    def test_perfect_efficiency(self):
        """Test no dissipation at 100% efficiency."""
        coupling = DissipativeCoupling(conductivity=0.5, efficiency=1.0)

        dissipated = coupling.dissipation_in_transfer(0, 0, E_from=2.0, E_to=1.0)
        assert dissipated == 0


class TestActivationBarrierCoupling:
    """Tests for activation barrier coupling."""

    def test_initialization(self):
        """Test default initialization."""
        coupling = ActivationBarrierCoupling()
        assert coupling.k == 0.1
        assert coupling.E_act == 0.5
        assert coupling.T == 1.0

    def test_favorable_no_barrier(self):
        """Test no barrier for favorable transfers."""
        coupling = ActivationBarrierCoupling(
            conductivity=0.5,
            activation_energy=0.5,
            temperature=1.0
        )

        # Favorable: downhill
        rate = coupling.energy_transfer_rate(0, 0, E_from=2.0, E_to=1.0)
        # Should be k * ΔE = 0.5 * 1.0 = 0.5
        assert abs(rate - 0.5) < 0.01

    def test_unfavorable_has_barrier(self):
        """Test barrier reduces unfavorable transfers."""
        coupling = ActivationBarrierCoupling(
            conductivity=0.5,
            activation_energy=0.5,
            temperature=1.0
        )

        # Unfavorable: uphill
        rate = coupling.energy_transfer_rate(0, 0, E_from=1.0, E_to=2.0)

        # Should be k * ΔE * exp(-E_act/T) = 0.5 * (-1) * exp(-0.5)
        expected = 0.5 * (-1.0) * np.exp(-0.5)
        assert abs(rate - expected) < 0.01

    def test_temperature_affects_barrier(self):
        """Test higher temperature lowers effective barrier."""
        coupling_cold = ActivationBarrierCoupling(
            conductivity=0.5,
            activation_energy=1.0,
            temperature=0.5
        )
        coupling_hot = ActivationBarrierCoupling(
            conductivity=0.5,
            activation_energy=1.0,
            temperature=2.0
        )

        # Unfavorable transfer
        rate_cold = coupling_cold.energy_transfer_rate(0, 0, E_from=1.0, E_to=2.0)
        rate_hot = coupling_hot.energy_transfer_rate(0, 0, E_from=1.0, E_to=2.0)

        # Hot should have higher magnitude (less suppressed)
        assert abs(rate_hot) > abs(rate_cold)


class TestCouplingMatrixFactory:
    """Tests for coupling matrix factory."""

    def test_create_gradient_couplings(self):
        """Test creating gradient coupling matrix."""
        factory = create_coupling_matrix('gradient')

        coupling_01 = factory(0, 1)
        assert isinstance(coupling_01, GradientDrivenCoupling)

    def test_create_asymmetric_couplings(self):
        """Test creating asymmetric coupling matrix."""
        factory = create_coupling_matrix(
            'asymmetric',
            k_forward=0.5,
            k_backward=0.1
        )

        coupling_01 = factory(0, 1)
        assert isinstance(coupling_01, AsymmetricEnergyCoupling)

    def test_strength_scaling(self):
        """Test that strengths scale from base matrix."""
        # Wunderling matrix has 0.312 for GIS->THC
        factory = create_coupling_matrix('gradient')
        coupling_01 = factory(0, 1)

        assert abs(coupling_01.k - 0.312) < 0.01

    def test_custom_base_matrix(self):
        """Test using custom coupling matrix."""
        custom_matrix = np.array([
            [0, 0.5, 0],
            [0.3, 0, 0.2],
            [0, 0, 0]
        ])

        factory = create_coupling_matrix('gradient', base_strengths=custom_matrix)
        coupling_01 = factory(0, 1)

        assert abs(coupling_01.k - 0.5) < 0.01

    def test_invalid_coupling_type(self):
        """Test error on invalid coupling type."""
        with pytest.raises(ValueError):
            create_coupling_matrix('invalid_type')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
