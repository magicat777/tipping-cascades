# Energy-Constrained Tipping Cascades Research

**Researcher:** Jason Holt
**Started:** December 2025
**Status:** Phase 4 In Progress - Network Resilience & Recovery Dynamics

## Project Overview

Investigating how energy constraints and thermodynamic principles affect tipping cascade dynamics in climate systems, with a focus on **Amazon rainforest resilience**. This research explores how network fragmentation from deforestation creates asymmetric dynamics where ecosystem degradation becomes progressively easier than recovery.

### Key Research Questions

1. **Does network fragmentation create tipping asymmetry?** ✅ **YES** - Confirmed in Phase 4
2. **Can passive recovery occur under normal climate variability?** Testing in progress
3. **What level of active intervention is required for restoration?** Testing in progress
4. **Which network connections are critical for resilience?** Planned

---

## Major Findings

### Phase 4: Network Fragmentation Creates "One-Way Valve" Effect ⭐

| Metric | Intact Network (100%) | Fragmented (10%) | Change |
|--------|----------------------|------------------|--------|
| Tip/Recovery Ratio | 1.005 | **1.148** | +14.3% |
| Total Entropy | 11,633 | 406 | -96.5% |
| Transitions/Run | 4,678 | 142 | -97% |

**Key Discovery**: At 10% edge retention, tipping is **14.8% more likely than recovery**. This creates a self-reinforcing degradation loop where deforestation makes future recovery progressively harder.

**Unexpected Finding**: Random fragmentation creates MORE asymmetry (17.4%) than targeted removal of high-betweenness edges (12.3%), suggesting critical recovery pathways may be distributed throughout the network.

### Phase 3: Energy-Constrained Cusp Model

- Developed `EnergyConstrainedCusp` elements with explicit potential landscapes
- Implemented Lévy stable noise for extreme event modeling
- Created `run_two_phase_experiment()` for cascade/recovery simulations
- Discovered "noise-type bifurcation" in tipping thermodynamics

### Phase 2: Coupling Asymmetry & Amazon Moisture Recycling

- **Coupling asymmetry is protective**: Original Wunderling network design prevents Greenland Ice Sheet cascading
- **Moisture recycling prevents tipping**: Interior forest moisture support saves 17% of edge cells at critical rainfall threshold
- **Lévy vs Gaussian noise**: Fat-tailed noise (α<2) enables both extreme tipping AND recovery events

---

## Current Progress

### ✅ Completed

- **Phase 0**: K3s infrastructure with JupyterLab, MLflow, Dask (14 workers)
- **Phase 1**: PyCascades framework analysis and documentation
- **Phase 2**: Baseline reproduction, coupling asymmetry discovery
- **Phase 3**: Energy-constrained module development
- **Phase 4 Experiment 8**: Network fragmentation analysis ✅ **(validated Dec 12, 2025)**

### 🔄 In Progress

- **Experiment 10c**: Restoration forcing - can active intervention restore tipped ecosystems?
- **Experiments 9, 10, 10b**: Pending re-validation with fixed solver

### 📋 Planned

- **Experiment 11**: Keystone edge identification
- **Experiment 8b**: Extended fragmentation (5%, 2%, 1% retention)
- Climate trend forcing during cascade phase
- Validation against observed Amazon data

---

## Technical Implementation

### Custom Module: `energy_constrained`

```python
from energy_constrained import (
    EnergyConstrainedNetwork,      # Network container
    EnergyConstrainedCusp,         # Bistable elements with potential
    GradientDrivenCoupling,        # Thermodynamically-consistent coupling
    run_two_phase_experiment,      # Cascade → Recovery simulation
    EnergyAnalyzer,                # Entropy & tipping event analysis
    get_dask_client                # Parallel execution on k3s
)
```

### Critical Solver Fix (December 2025)

A boundary oscillation bug was discovered and fixed in the Euler-Maruyama solver:
- **Problem**: Hard clamp at ±10 caused 96.8% of trajectory time at boundaries
- **Fix**: Soft reflection at ±2 keeps cells in bistable region
- **Impact**: All experiments prior to Dec 12, 2025 require re-validation

---

## Directory Structure

```
cascades/
├── src/energy_constrained/          # Custom thermodynamic module
│   ├── elements.py                  # EnergyConstrainedCusp
│   ├── couplings.py                 # GradientDrivenCoupling
│   ├── network.py                   # Network container
│   ├── solvers.py                   # Euler-Maruyama with Lévy noise
│   ├── analysis.py                  # EnergyAnalyzer
│   └── dask_utils.py                # Parallel execution utilities
├── notebooks/
│   ├── 01-05: Phase 2-3 exploration
│   ├── 06_network_fragmentation.ipynb    # Experiment 8
│   ├── 07_recovery_dynamics.ipynb        # Experiment 9
│   ├── 08_alpha_sweep.ipynb              # Experiment 10
│   ├── 09_alpha_sigma_sweep.ipynb        # Experiment 10b
│   └── 10_restoration_forcing.ipynb      # Experiment 10c
├── docs/
│   ├── phase4_results.md            # Comprehensive experiment results
│   ├── phase2_conclusions.md        # Earlier findings
│   └── pycascades_architecture.md   # Framework documentation
├── data/                            # Amazon moisture recycling data
│   └── amazon/                      # Wunderling et al. 2022 dataset
└── external/
    └── pycascades/                  # PIK framework (dependency)
```

---

## Infrastructure

### K3s Cluster (Single Node)

| Service | Port | Purpose |
|---------|------|---------|
| JupyterLab | 30888 | Interactive notebooks |
| Dask Dashboard | 30787 | Parallel task monitoring |
| MLflow | 30505 | Experiment tracking |

### Dask Configuration

- **14 workers** (1.5 CPU, 1GB RAM each)
- Optimized scatter-based task distribution
- ~2x speedup over initial configuration

```bash
# Access services
http://localhost:30888  # JupyterLab
http://localhost:30787  # Dask Dashboard
http://localhost:30505  # MLflow

# Check cluster status
kubectl get pods -n cascades
```

---

## Key References

- Wunderling et al. (2021) - PyCascades framework
- Wunderling et al. (2022) - Amazon moisture recycling network data
- Lenton et al. (2008) - Tipping elements in Earth's climate system

---

## Research Outputs

### Phase 4 Documentation
- [Experiment Results](docs/phase4_results.md) - Comprehensive findings with methodology
- [Research Plan](docs/phase4_research_plan.md) - Experimental design

### Key Metrics Tracked
- Tip/Recovery ratio by fragmentation level
- Entropy production (thermodynamic activity)
- Tipping event counts and transitions
- Recovery fraction under various noise regimes

---

*Last Updated: December 12, 2025*
