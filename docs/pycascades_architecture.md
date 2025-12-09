# PyCascades Architecture Documentation

## Overview

PyCascades is a Python framework developed by the Potsdam Institute for Climate Impact Research (PIK) for simulating tipping cascades on complex networks. Version 1.0.2.

**Paper:** Wunderling et al., "Modelling nonlinear dynamics of interacting tipping elements on complex networks: the PyCascades package", European Physical Journal Special Topics (2021)

## Core Components

### 1. Tipping Elements (`pycascades/core/tipping_element.py`)

Abstract base class `tipping_element` with concrete implementations:

| Class | Bifurcation Type | Key Parameters | Equation |
|-------|-----------------|----------------|----------|
| `cusp` | Cusp bifurcation | a, b, c, x_0 | dx/dt = a(x-x_0)^3 + b(x-x_0) + c |
| `realistic_cusp` | Cusp with physical params | timescale, x_tuple, rho_tuple | Maps temperature/forcing to cusp parameters |
| `hopf` | Hopf bifurcation | a, c | dr/dt = (c - r^2) * r * a |

**Key Methods:**
- `dxdt_diag()`: Returns callable for dx/dt (intrinsic dynamics)
- `jac_diag()`: Returns callable for Jacobian diagonal element
- `tip_state()`: Returns callable to determine if element has tipped

### 2. Tipping Network (`pycascades/core/tipping_network.py`)

Extends `networkx.DiGraph` to create networks of interacting tipping elements.

**Key Methods:**
- `add_element(tipping_element)`: Add node with tipping element
- `add_coupling(from_id, to_id, coupling)`: Add directed edge with coupling
- `set_param(node_id, key, val)`: Modify element parameters
- `f(x, t)`: Compute full system dynamics (intrinsic + coupling)
- `jac(x, t)`: Compute full Jacobian matrix
- `get_tip_states(x)`: Return boolean array of tipped elements
- `get_number_tipped(x)`: Count tipped elements

### 3. Coupling (`pycascades/core/coupling.py`)

Defines how tipping elements influence each other:

| Class | Description | Coupling Term |
|-------|-------------|---------------|
| `linear_coupling` | Simple linear | strength * x_from |
| `cusp_to_hopf` | Cusp affects Hopf | a * r_to * strength * x_from |
| `hopf_x_to_cusp` | Hopf x-component to cusp | strength * r_from * cos(b*t) |
| `hopf_y_to_cusp` | Hopf y-component to cusp | strength * r_from * sin(b*t) |
| `hopf_x_to_hopf` | Hopf to Hopf (x) | Complex coupling |
| `hopf_y_to_hopf` | Hopf to Hopf (y) | Complex coupling |

### 4. Evolution (`pycascades/core/evolve.py`, `evolve_sde.py`)

Simulates network dynamics over time:

**Deterministic (`evolve`):**
- Uses `scipy.integrate.odeint` with Jacobian
- `integrate(t_step, t_end)`: Manual integration
- `equilibrate(tol, t_step, t_break)`: Integrate until equilibrium
- `is_equilibrium(tol)`: Check if |f(x)| < tol
- `is_stable()`: Check eigenvalues of Jacobian

**Stochastic (`evolve_sde`):**
- Uses `sdeint` for stochastic differential equations
- Supports Gaussian, Levy, and Cauchy noise processes

## Module Structure

```
pycascades/
├── __init__.py          # Main exports
├── core/
│   ├── tipping_element.py      # Cusp, Hopf bifurcations
│   ├── tipping_network.py      # Network container
│   ├── coupling.py             # Coupling functions
│   ├── evolve.py               # Deterministic solver
│   ├── evolve_sde.py           # Stochastic solver
│   ├── tipping_element_economic.py
│   ├── tipping_network_economic.py
│   ├── coupling_economic.py
│   └── evolve_economic.py
├── gen/
│   └── networks.py             # Network generators (ER, BA, WS, spatial)
├── utils/
│   └── plotter.py              # Visualization tools
├── amazon/                     # Amazon rainforest module
└── earth_system/               # Climate tipping elements module
```

## Typical Workflow

```python
import pycascades as pc

# 1. Create tipping elements
element_0 = pc.cusp(a=-4, b=1, c=0, x_0=0.5)
element_1 = pc.cusp(a=-4, b=1, c=0, x_0=0.5)

# 2. Create couplings
coupling_01 = pc.linear_coupling(strength=0.2)
coupling_10 = pc.linear_coupling(strength=0.05)

# 3. Build network
net = pc.tipping_network()
net.add_element(element_0)
net.add_element(element_1)
net.add_coupling(0, 1, coupling_01)
net.add_coupling(1, 0, coupling_10)

# 4. Simulate
initial_state = [0.1, 0.9]
ev = pc.evolve(net, initial_state)
ev.integrate(timestep=0.01, t_end=10)

# 5. Analyze
times, states = ev.get_timeseries()
tipped = net.get_tip_states(states[-1])
```

## Network Generators (`pycascades.gen.networks`)

- `from_nxgraph()`: Convert NetworkX graph to tipping network
- `spatial_graph()`: Create spatially-embedded network
- Supports Erdos-Renyi, Barabasi-Albert, Watts-Strogatz via NetworkX

## Visualization (`pycascades.plotter`)

- `network(net)`: Network topology plot
- `phase_flow(net)`: Vector field visualization
- `phase_space(net)`: Phase space with nullclines
- `stability(net)`: Stability regions (eigenvalue analysis)
- `series(times, states)`: Time series plot

## Application Domains

### Amazon Rainforest (`pycascades.amazon`)
- Moisture recycling network
- Critical rainfall thresholds
- Spatial cascade dynamics

### Earth System (`pycascades.earth_system`)
- Climate tipping elements (AMOC, ice sheets, monsoons, etc.)
- Interaction strengths from literature
- Global warming scenarios

**Import Paths for Custom Network Building:**

```python
# Core imports (when building networks manually)
from pycascades.core.tipping_element import cusp
from pycascades.earth_system.tipping_network_earth_system import tipping_network
from pycascades.earth_system.earth import linear_coupling_earth_system
from pycascades.earth_system.functions_earth_system import global_functions

# Pre-built earth system (simpler usage)
from pycascades.earth_system import Earth_System
```

**Earth System Elements:**

| Element | Abbrev | Timescale | Threshold (°C) | Description |
|---------|--------|-----------|----------------|-------------|
| Greenland Ice Sheet | GIS | 98 yr | 2.24 | Slow ice sheet dynamics |
| Thermohaline Circulation | THC | 6 yr | 4.69 | Atlantic overturning |
| West Antarctic Ice Sheet | WAIS | 48 yr | 1.42 | Marine ice sheet |
| Amazon Rainforest | AMAZ | 1 yr | 4.10 | Fast vegetation response |

**Coupling Matrix (Wunderling et al.):**

```
         TO →   GIS     THC     WAIS    AMAZ
FROM ↓
  GIS           -      0.312   0.271    0
  THC         0.578     -      0.132   0.372
  WAIS        0.193   0.120     -       0
  AMAZ          0       0       0       -
```

Note: THC→GIS is stabilizing (negative sign), all others destabilizing

## Extension Points for Energy-Constrained Research

1. **New Tipping Element Types**: Extend `tipping_element` class with energy flux terms
2. **Energy-Based Couplings**: Create couplings that represent energy/entropy transfer
3. **Dissipative Network**: Extend `tipping_network` to track system-wide energy balance
4. **Custom Solvers**: Modify `evolve` to include energy constraints

## References

1. Wunderling et al. (2021) - PyCascades description paper
2. Wunderling et al. (2020) - Motifs and tipping cascades
3. Kronke et al. (2020) - Dynamics of tipping cascades
4. Wunderling et al. (2021) - Climate domino effects
