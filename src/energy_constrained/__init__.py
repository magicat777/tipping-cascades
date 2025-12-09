"""
Energy-Constrained Tipping Cascades Module

This module extends the PyCascades framework with thermodynamic constraints
to investigate how energy flow and entropy production affect tipping cascade
dynamics.

Key components:
- elements: Energy-tracking tipping elements (EnergyConstrainedCusp)
- couplings: Energy-based coupling functions
- network: Dissipative network with energy balance tracking
- solvers: Energy-constrained SDE integrators
- analysis: Entropy production and energy flow analysis tools
"""

from .elements import EnergyConstrainedElement, EnergyConstrainedCusp
from .couplings import EnergyCoupling, GradientDrivenCoupling, AsymmetricEnergyCoupling
from .network import EnergyConstrainedNetwork
from .solvers import energy_constrained_euler_maruyama
from .analysis import EnergyAnalyzer

__all__ = [
    'EnergyConstrainedElement',
    'EnergyConstrainedCusp',
    'EnergyCoupling',
    'GradientDrivenCoupling',
    'AsymmetricEnergyCoupling',
    'EnergyConstrainedNetwork',
    'energy_constrained_euler_maruyama',
    'EnergyAnalyzer',
]

__version__ = '0.1.0'
