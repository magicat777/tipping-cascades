# Phase 2 Conclusions: Baseline Reproduction & Exploration

**Author:** Jason Holt
**Date:** December 2025
**Status:** Complete

---

## Executive Summary

Phase 2 investigated the dynamics of climate tipping cascades using the PyCascades framework, focusing on how noise characteristics, coupling strength, and network topology affect cascade behavior in a 4-element Earth system model (GIS, THC, WAIS, AMAZ).

**Key Discoveries:**

1. **Coupling asymmetry stabilizes cascades:** Network topology (specifically coupling asymmetry) acts as a stabilizing mechanism that can prevent cascade propagation to slow-responding elements like the Greenland Ice Sheet. Symmetrizing the coupling network fundamentally changes system behavior, enabling cascades that are otherwise blocked.

2. **Moisture recycling creates protective feedback:** The Amazon spatial cascade model demonstrates that coupling can be *protective* rather than destructive. Moisture recycling between forest cells consistently reduces tipping compared to uncoupled scenarios, acting as a natural "cascade firewall."

---

## 1. System Overview

### 1.1 Tipping Elements

| Element | Abbreviation | Timescale | Tipping Threshold (GMT) | Role |
|---------|--------------|-----------|-------------------------|------|
| Greenland Ice Sheet | GIS | 98 years | 2.24°C | Slowest responder |
| Thermohaline Circulation | THC | 6 years | 4.69°C | Fast ocean dynamics |
| West Antarctic Ice Sheet | WAIS | 48 years | 1.42°C | Intermediate responder |
| Amazon Rainforest | AMAZ | 1 year | 4.10°C | Fastest responder |

### 1.2 Original Coupling Matrix (Wunderling et al.)

```
         TO →   GIS     THC     WAIS    AMAZ
FROM ↓
  GIS           -      0.312   0.271    0
  THC         0.578     -      0.132   0.372
  WAIS        0.193   0.120     -       0
  AMAZ          0       0       0       -
```

**Critical observation:** Amazon has NO outgoing connections - it is a pure network "sink."

---

## 2. Key Findings

### 2.1 Timescale Mismatch Barrier

The two-order-of-magnitude difference between Amazon (1 year) and Greenland (98 years) timescales creates a fundamental barrier to cascade propagation.

- Fast oscillations in Amazon don't persist long enough to influence slow ice sheet dynamics
- Even with strong coupling (0.5), GIS remains isolated from rapid perturbations
- This is a **structural property** of the system, not a parameter tuning issue

### 2.2 Noise Characteristics Matter

| Noise Type | α Parameter | Behavior | Implications |
|------------|-------------|----------|--------------|
| Lévy stable | 1.5 | Heavy tails, infinite variance | Enables back-tipping, extreme events |
| Gaussian | 2.0 | Light tails, finite variance | Bounded fluctuations, smoother trajectories |

**Physical interpretation:** Lévy noise (α=1.5) better represents real climate forcing where extreme events (volcanic eruptions, sudden ice discharge) occur more frequently than Gaussian statistics predict.

### 2.3 Coupling Topology is a Stabilizing Mechanism

**This is the major finding of Phase 2.**

#### Experiment: Asymmetric vs Symmetric Coupling

We compared four network configurations:
1. **Baseline:** Original asymmetric Wunderling matrix
2. **Symmetric:** Bidirectional couplings (averaged strengths)
3. **Amplified:** Exaggerated asymmetry (doubled ratios)
4. **Amazon Feedback:** Added AMAZ→THC, AMAZ→GIS, AMAZ→WAIS links

#### Results (N=20 ensemble runs per configuration)

| Metric | Baseline | Symmetric | Amplified | Amazon Feedback |
|--------|----------|-----------|-----------|-----------------|
| GIS mean state | -0.95 | +0.85 | -1.0 | -0.80 |
| GIS time tipped | 0% | 85% | 0% | 5% |
| GIS-THC correlation | -0.20 | +0.35 | -0.15 | -0.05 |

**Statistical significance:** 8 of 9 metrics showed significant differences (p < 0.05) between configurations using Kruskal-Wallis H-test.

#### Interpretation

The asymmetric coupling structure **protects GIS from cascading**:

1. In the original network, GIS influences others (GIS→THC, GIS→WAIS) but reverse links are weak
2. GIS is effectively "upstream" in the cascade hierarchy
3. When couplings are symmetrized, THC and WAIS (which tip early) can "pull" GIS into the tipped state
4. The cascade becomes bidirectional, and GIS gets dragged across its threshold

**Implication:** The observed asymmetry in real climate teleconnections may serve as a natural stabilizing mechanism.

### 2.4 Cross-Correlation Signatures

- **Baseline:** GIS-THC correlation is negative (~-0.2) - they move in opposite directions
- **Symmetric:** GIS-THC correlation becomes positive (~+0.35) - synchronized behavior
- **THC-AMAZ correlation:** Robust across all configurations (strong unidirectional link)

This suggests that **correlation structure can diagnose cascade risk** - positive correlations between slow and fast elements indicate higher cascade vulnerability.

### 2.5 Amazon Moisture Recycling: Coupling as Protection

**This finding reinforces and extends the coupling asymmetry discovery.**

#### Experiment: Spatial Cascade Model

Using PyCascades' Amazon module with 2003 moisture recycling data (Staal et al.), we simulated forest dieback cascades across ~567 grid cells at 1° resolution. Each cell can tip from forest to savanna when local rainfall drops below a critical threshold (r_crit).

The key comparison: cells tipping **with** vs **without** moisture coupling between neighbors.

#### Results: Negative Cascade Amplification

| r_crit (mm/year) | Tipped (with coupling) | Tipped (no coupling) | Cascade Amplification |
|------------------|------------------------|----------------------|----------------------|
| 1200 | 2 | 4 | **-2** |
| 1400 | 7 | 11 | **-4** |
| 1600 | 29 | 48 | **-19** |
| 1800 | 107 | 129 | **-22** |

**Critical finding:** Cascade amplification is *negative* across all r_crit values tested. Moisture coupling **prevents** tipping rather than amplifying it.

#### Spatial Pattern of Protection

At r_crit = 1500 mm/year:
- **537 cells** remain stable (green)
- **13 cells** tip due to local conditions regardless of coupling (direct tipping)
- **0 cells** tip only due to cascade (no cascade amplification)
- **17 cells** tip WITHOUT coupling but remain stable WITH coupling (anomalous/protected)

The protected cells cluster on transition zones:
- Western edge (~longitude -75° to -78°)
- Southern edge (~latitude -12° to -15°)

These are boundary regions where moisture recycling from the interior forest provides a critical buffer.

#### Interpretation

1. **Moisture recycling is a stabilizing feedback:** Neighboring forest cells share moisture through evapotranspiration and precipitation recycling, creating mutual support.

2. **Protection scales with stress:** At higher r_crit (more cells vulnerable), coupling saves proportionally more cells. At r_crit=1800, coupling prevents 17% of potential tipping (22/129 cells).

3. **Edge cells benefit most:** Transition zones receive protective moisture from interior forest, explaining the spatial pattern of "anomalous" cells.

4. **Natural cascade firewalls exist:** The Amazon's moisture recycling network appears to create built-in resilience against cascading failure.

#### Connection to Global Tipping Network

This result complements the 4-element Earth system findings:

| System | Coupling Effect | Mechanism |
|--------|-----------------|-----------|
| GIS-THC-WAIS-AMAZ | Asymmetry protects GIS | Directional cascade hierarchy |
| Amazon spatial | Coupling protects edges | Mutual moisture support |

Both demonstrate that **coupling topology can create stabilizing feedback loops**, not just destructive cascades. This has profound implications for Phase 3's energy-constrained modeling.

---

## 3. Methodological Insights

### 3.1 Single Trajectories are Misleading

Initial visual comparison of single simulation runs showed nearly identical dynamics across all configurations. The ensemble statistical approach (N=20 runs with different random seeds) was essential to detect the significant differences.

**Lesson:** Lévy noise creates high trajectory-to-trajectory variability that masks underlying differences. Always use ensemble statistics for stochastic systems.

### 3.2 Metrics for Cascade Analysis

Effective metrics for comparing cascade dynamics:
- **Mean state** (after burn-in period)
- **Time spent tipped** (fraction of time x > 0)
- **Threshold crossing events** (back-tipping frequency)
- **Cross-correlations** (synchronization between elements)

Less informative:
- Final state (high variance due to Lévy noise)
- Maximum excursions (dominated by extreme events)

---

## 4. Implications for Phase 3

### 4.1 Energy Constraints May Create Coupling Asymmetry

The finding that coupling asymmetry stabilizes the system suggests a potential mechanism for Phase 3:

- **Hypothesis:** Energy constraints naturally create asymmetric effective couplings
- Energy must flow "downhill" (from high to low potential)
- This could impose directional structure on cascade propagation
- Thermodynamic constraints may explain observed asymmetries in climate teleconnections

### 4.2 Protective Coupling as Energy Minimization

The Amazon moisture recycling results suggest an additional mechanism:

- **Hypothesis:** Stabilizing coupling networks may represent thermodynamically favorable configurations
- Moisture recycling maintains forest biomass (stored energy/carbon)
- Forest loss releases energy and disrupts the recycling loop
- The protective coupling may emerge from energy minimization principles

### 4.3 Questions to Address in Phase 3

1. Can energy-based coupling functions reproduce the stabilizing asymmetry?
2. Does enforcing energy conservation change cascade thresholds?
3. Can dissipation rates predict cascade vulnerability?
4. Do energy constraints create natural cascade "firewalls"?
5. **NEW:** Can we derive the protective Amazon coupling from thermodynamic principles?
6. **NEW:** What energy cost is associated with cascade propagation vs. stability?

---

## 5. Technical Notes

### 5.1 Simulation Parameters

```python
# Standard configuration
duration = 100000.0  # years
t_step = 15
GMT = 2.5  # °C above pre-industrial
base_strength = 0.5
alphas = [1.5, 1.5, 1.5, 1.5]  # Lévy stability
sigmas = [0.05, 0.05, 0.05, 0.05]  # Noise amplitude
```

### 5.2 Statistical Methods

- **Kruskal-Wallis H-test:** Non-parametric ANOVA for comparing distributions
- **Mann-Whitney U-test:** Pairwise comparisons vs baseline
- **Significance threshold:** p < 0.05, with Bonferroni consideration for multiple comparisons

### 5.3 Notebooks

- `01_earth_system_tipping_exploration.ipynb`: Initial parameter exploration
- `02_asymmetric_coupling_exploration.ipynb`: Coupling topology analysis with ensemble statistics
- `03_amazon_moisture_recycling.ipynb`: Spatial cascade model with moisture coupling analysis

---

## 6. References

1. Wunderling et al. (2021) - PyCascades framework paper
2. Wunderling et al. (2020) - Interacting tipping elements increase risk of climate domino effects
3. Lenton et al. (2008) - Tipping elements in the Earth's climate system
4. Staal et al. (2018) - Forest-rainfall cascades buffer against drought across the Amazon
5. Zemp et al. (2017) - Self-amplified Amazon forest loss due to vegetation-atmosphere feedbacks

---

## Appendix: Coupling Matrices Used

### Baseline (Original Wunderling)
```
[[0.000, 0.312, 0.271, 0.000],
 [0.578, 0.000, 0.132, 0.372],
 [0.193, 0.120, 0.000, 0.000],
 [0.000, 0.000, 0.000, 0.000]]
```

### Symmetric
```
[[0.000, 0.445, 0.232, 0.186],
 [0.445, 0.000, 0.126, 0.186],
 [0.232, 0.126, 0.000, 0.000],
 [0.186, 0.186, 0.000, 0.000]]
```

### Amplified Asymmetry
```
[[0.000, 0.156, 0.136, 0.000],
 [1.000, 0.000, 0.066, 0.744],
 [0.386, 0.240, 0.000, 0.000],
 [0.000, 0.000, 0.000, 0.000]]
```

### Amazon Feedback
```
[[0.000, 0.312, 0.271, 0.000],
 [0.578, 0.000, 0.132, 0.372],
 [0.193, 0.120, 0.000, 0.000],
 [0.100, 0.300, 0.050, 0.000]]
```
