# Energy-Constrained Tipping Cascades Research

**Researcher:** Jason Holt
**Started:** December 2025
**Status:** Phase 2 Complete, Ready for Phase 3

## Project Overview

Investigating how energy constraints and thermodynamic principles affect tipping cascade dynamics in climate systems, using the PyCascades framework from PIK.

## Current Progress

### Completed

- **Phase 0: Infrastructure Setup**
  - K3s cluster with cascades namespace
  - JupyterLab deployment (port 30888)
  - MLflow experiment tracking (port 30505)
  - Dask distributed computing (port 30787)
  - Custom container image with PyCascades v1.0.2

- **Phase 0.5: K3S Storage Migration**
  - Migrated k3s data to dedicated LVM volume
  - Freed 72GB on root partition

- **Phase 1: Foundation**
  - Cloned repositories: pycascades, pyunicorn, pycopancore
  - Documented PyCascades architecture (`docs/pycascades_architecture.md`)
  - Set up notebook workflow

- **Phase 2: Baseline Reproduction** ✓ Complete
  - Ran example_network_tipping_cascades.ipynb
  - Explored earth_sys_levy_stable_noise.ipynb extensively
  - Documented findings in `notebooks/01_earth_system_tipping_exploration.ipynb`
  - Completed asymmetric coupling analysis (`notebooks/02_asymmetric_coupling_exploration.ipynb`)
  - Completed Amazon moisture recycling analysis (`notebooks/03_amazon_moisture_recycling.ipynb`)
  - **Full conclusions documented in `docs/phase2_conclusions.md`**

### Key Findings (Phase 2)

1. **Timescale Mismatch Barrier**: The 98-year GIS vs 1-year Amazon timescale difference prevents cascade propagation even with strong coupling (0.5)

2. **Noise Type Matters**: Lévy noise (α=1.5) enables back-tipping and extreme events; Gaussian (α=2.0) produces bounded dynamics

3. **Coupling Asymmetry is a Stabilizing Mechanism** ⭐ *Major Finding*
   - Original Wunderling network has asymmetric couplings that protect GIS from cascading
   - Symmetrizing couplings causes GIS to tip (0% → 85% time in tipped state)
   - GIS-THC correlation flips from negative (-0.2) to positive (+0.35)
   - Statistical significance confirmed via ensemble analysis (N=20, p < 0.001)

4. **Amazon is a Network Sink**: Original model has no outgoing connections from Amazon; adding feedback links (AMAZ→THC) moderately affects dynamics

5. **Ensemble Methods Essential**: Single trajectories misleading due to Lévy noise; ensemble statistics required to detect coupling effects

6. **Moisture Recycling is Protective** ⭐ *Major Finding*
   - Amazon spatial model shows coupling PREVENTS tipping (negative cascade amplification)
   - At r_crit=1800 mm/year, coupling saves 22 cells (17% reduction in tipping)
   - Edge cells benefit most from interior forest moisture support
   - Natural "cascade firewalls" exist in moisture recycling networks

## Next: Phase 3 - Energy-Constrained Extensions

Design and implement energy/thermodynamic constraints:
1. New tipping element types with energy flux terms
2. Energy-based coupling functions
3. Dissipative network tracking system-wide energy balance
4. Custom solvers with energy constraints

## Directory Structure

```
cascades/
├── README.md                    # This file
├── energy_constrained_tipping_cascades_research_project.md  # Full project plan
├── foundations_reading_list.md  # Background literature
├── docs/
│   ├── pycascades_architecture.md  # Framework documentation
│   ├── phase2_conclusions.md       # Phase 2 findings & analysis
│   └── thermodynamic_tipping_points_literature_review.md  # Phase 3 foundation
├── notebooks/
│   ├── 01_earth_system_tipping_exploration.ipynb
│   ├── 02_asymmetric_coupling_exploration.ipynb
│   └── 03_amazon_moisture_recycling.ipynb
├── external/
│   ├── pycascades/             # PIK framework (cloned)
│   ├── pyunicorn/              # Complex networks (cloned)
│   └── pycopancore/            # World-Earth modeling (cloned)
└── configs/
    └── Dockerfile              # JupyterLab container
```

## Quick Start

```bash
# Access JupyterLab
http://localhost:30888

# Access MLflow
http://localhost:30505

# Check pod status
kubectl get pods -n cascades
```

## Key Import Paths (PyCascades)

```python
# For custom network building
from pycascades.core.tipping_element import cusp
from pycascades.earth_system.tipping_network_earth_system import tipping_network
from pycascades.earth_system.earth import linear_coupling_earth_system
from pycascades.earth_system.functions_earth_system import global_functions
```

## References

- Wunderling et al. (2021) - PyCascades framework paper
- Full reading list in `foundations_reading_list.md`
