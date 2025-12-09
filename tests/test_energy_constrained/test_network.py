"""
Unit tests for energy-constrained network.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from energy_constrained.network import (
    EnergyConstrainedNetwork,
    create_earth_system_network
)
from energy_constrained.elements import EnergyConstrainedCusp
from energy_constrained.couplings import GradientDrivenCoupling


class TestEnergyConstrainedNetwork:
    """Tests for the energy-constrained network class."""

    @pytest.fixture
    def simple_network(self):
        """Create a simple 2-element network for testing."""
        net = EnergyConstrainedNetwork()

        elem1 = EnergyConstrainedCusp(
            E_stable=0.0, E_tipped=1.0, barrier_height=0.5
        )
        elem2 = EnergyConstrainedCusp(
            E_stable=0.0, E_tipped=1.5, barrier_height=0.7
        )

        net.add_element('A', elem1)
        net.add_element('B', elem2)

        coupling = GradientDrivenCoupling(conductivity=0.1)
        net.add_coupling('A', 'B', coupling)

        return net

    def test_add_elements(self, simple_network):
        """Test adding elements to network."""
        assert simple_network.n_elements == 2
        assert 'A' in simple_network.nodes()
        assert 'B' in simple_network.nodes()

    def test_get_element(self, simple_network):
        """Test retrieving elements."""
        elem_a = simple_network.get_element('A')
        assert isinstance(elem_a, EnergyConstrainedCusp)
        assert elem_a.E_tipped == 1.0

    def test_get_coupling(self, simple_network):
        """Test retrieving couplings."""
        coupling = simple_network.get_coupling('A', 'B')
        assert isinstance(coupling, GradientDrivenCoupling)

    def test_compute_element_energies(self, simple_network):
        """Test computing energies from states."""
        x = np.array([-1.0, 1.0])  # A stable, B tipped
        E = simple_network.compute_element_energies(x)

        assert len(E) == 2
        # Stable state should have lower energy than tipped
        assert E[0] < E[1]

    def test_compute_total_energy(self, simple_network):
        """Test total energy computation."""
        x = np.array([-1.0, -1.0])  # Both stable
        E_total = simple_network.compute_total_energy(x)

        E_elements = simple_network.compute_element_energies(x)
        assert abs(E_total - np.sum(E_elements)) < 1e-10

    def test_compute_energy_flows(self, simple_network):
        """Test energy flow matrix computation."""
        x = np.array([-1.0, 1.0])
        E = simple_network.compute_element_energies(x)
        flows = simple_network.compute_energy_flows(x, E)

        assert flows.shape == (2, 2)

        # Flow from A to B should exist
        assert flows[0, 1] != 0

        # No reverse coupling in simple network
        assert flows[1, 0] == 0

    def test_f_state(self, simple_network):
        """Test state dynamics computation."""
        x = np.array([0.0, 0.0])  # Both at unstable equilibrium
        dxdt = simple_network.f_state(x, t=0)

        assert len(dxdt) == 2
        # Near equilibrium, derivatives should be small
        assert abs(dxdt[0]) < 0.5
        assert abs(dxdt[1]) < 0.5

    def test_f_extended(self, simple_network):
        """Test extended dynamics (state + energy)."""
        y0 = simple_network.get_initial_state()
        assert len(y0) == 4  # 2 states + 2 energies

        dydt = simple_network.f_extended(y0, t=0)
        assert len(dydt) == 4

    def test_get_initial_state(self, simple_network):
        """Test initial state generation."""
        # Default: stable states
        y0_default = simple_network.get_initial_state()
        assert y0_default[0] == -1.0  # x_A
        assert y0_default[1] == -1.0  # x_B

        # Custom initial states
        x0 = np.array([0.5, -0.5])
        y0_custom = simple_network.get_initial_state(x0)
        assert y0_custom[0] == 0.5
        assert y0_custom[1] == -0.5


class TestEarthSystemNetwork:
    """Tests for Earth system network factory."""

    def test_create_default_network(self):
        """Test creating default Earth system network."""
        net = create_earth_system_network()

        assert net.n_elements == 4
        assert 'GIS' in net.nodes()
        assert 'THC' in net.nodes()
        assert 'WAIS' in net.nodes()
        assert 'AMAZ' in net.nodes()

    def test_coupling_structure(self):
        """Test that coupling structure matches Wunderling."""
        net = create_earth_system_network()

        # Check expected couplings exist
        assert net.has_edge('GIS', 'THC')
        assert net.has_edge('THC', 'GIS')
        assert net.has_edge('THC', 'AMAZ')

        # Amazon has no outgoing couplings
        assert not net.has_edge('AMAZ', 'GIS')
        assert not net.has_edge('AMAZ', 'THC')

    def test_different_coupling_types(self):
        """Test creating networks with different coupling types."""
        net_grad = create_earth_system_network(coupling_type='gradient')
        net_asym = create_earth_system_network(coupling_type='asymmetric')

        # Both should have same structure
        assert net_grad.n_elements == net_asym.n_elements

        # But different coupling types
        coupling_grad = net_grad.get_coupling('GIS', 'THC')
        coupling_asym = net_asym.get_coupling('GIS', 'THC')

        assert type(coupling_grad) != type(coupling_asym)

    def test_initial_state(self):
        """Test initial state for Earth system network."""
        net = create_earth_system_network()
        y0 = net.get_initial_state()

        # Should have 8 values: 4 states + 4 energies
        assert len(y0) == 8

        # All states should start stable
        x0 = y0[:4]
        assert np.all(x0 < 0)


class TestNetworkAnalysis:
    """Tests for network analysis methods."""

    @pytest.fixture
    def network_with_trajectory(self):
        """Create network with simulated trajectory for testing."""
        net = create_earth_system_network()

        # Create fake trajectory
        n_times = 100
        n_elements = 4

        t = np.linspace(0, 100, n_times)

        # Simple trajectory: element 0 tips, others stay stable
        x = np.zeros((n_times, n_elements))
        x[:, 0] = np.linspace(-1, 1, n_times)  # GIS tips
        x[:, 1:] = -0.8  # Others stable

        E = np.zeros((n_times, n_elements))
        for i, name in enumerate(net.node_list):
            elem = net.get_element(name)
            E[:, i] = [elem.energy(x[j, i]) for j in range(n_times)]

        y = np.column_stack([x, E])

        return net, t, y

    def test_identify_tipping_events(self, network_with_trajectory):
        """Test tipping event identification."""
        net, t, y = network_with_trajectory

        events = net.identify_tipping_events(y, t)

        # Should find at least one tipping event
        assert len(events) > 0

        # First event should be GIS tipping
        tip_events = [e for e in events if e['direction'] == 'tip']
        assert len(tip_events) > 0

    def test_get_energy_budget(self, network_with_trajectory):
        """Test energy budget computation."""
        net, t, y = network_with_trajectory

        budget = net.get_energy_budget(y, t)

        assert 'E_total' in budget
        assert 'dissipation_rate' in budget
        assert len(budget['E_total']) == len(t)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
