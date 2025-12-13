# Phase 4 Research Plan: Network Resilience and Recovery Dynamics

**Project**: Energy-Constrained Tipping Cascades in Coupled Socio-Ecological Systems
**Phase**: 4 - Analysis and Validation
**Date**: December 2025
**Author**: Jason Holt

---

## Executive Summary

Phase 3 established fundamental insights about thermodynamic constraints on tipping cascades:
- Asymmetric coupling minimizes entropy production
- Lévy noise triggers cascades that Gaussian noise cannot
- Amazon moisture recycling provides scale-invariant thermodynamic buffering
- Extreme events dominate over gradual drought in triggering cascades

Phase 4 builds on these findings with targeted experiments to:
1. Test the robustness of network buffering under fragmentation (deforestation)
2. Characterize recovery dynamics after cascade events
3. Map the Lévy-Gaussian transition with fine resolution
4. Strengthen statistical significance for publication

---

## Research Questions

| ID | Question | Hypothesis | Priority |
|----|----------|------------|----------|
| RQ1 | At what connectivity threshold does thermodynamic buffering collapse? | Below ~50% edge retention, asymmetric dynamics emerge | High |
| RQ2 | Do drought-weakened networks trap cells in tipped states? | 2010 network recovers slower than 2003 after cascade | High |
| RQ3 | Is the Lévy-Gaussian transition sharp or gradual? | Critical α ≈ 1.7 separates regimes | Medium |
| RQ4 | Which connections are "keystones" for network resilience? | High-betweenness edges are disproportionately important | Medium |
| RQ5 | Does network topology (not just density) determine buffering? | Amazon-specific topology provides unique resilience | Low |

---

## Experiment 8: Network Fragmentation and Deforestation Scenarios

### Objective

Test whether progressive removal of network connections causes thermodynamic buffering to collapse, and identify critical connectivity thresholds.

### Scientific Rationale

Phase 3 found that the Amazon moisture recycling network provides scale-invariant buffering (tip/recovery ratio ≈ 1.0). This suggests network connectivity is essential for resilience. Deforestation breaks moisture recycling pathways—if buffering depends on connectivity, there may be a critical threshold below which resilience collapses catastrophically.

### Experimental Design

**Network Construction**
- Base: Full 567-cell Amazon network with all edges (min_flow > 1.0 mm)
- Fragmentation levels: 100%, 90%, 75%, 50%, 25%, 10% edge retention
- Edge removal strategy: Random (baseline), then targeted (by betweenness centrality)

**Fragmentation Methods**

| Method | Description | Rationale |
|--------|-------------|-----------|
| Random | Remove edges uniformly at random | Baseline; simulates diffuse deforestation |
| Low-flow first | Remove weakest edges first | Simulates selective clearing of marginal areas |
| High-betweenness first | Remove highest betweenness edges first | Simulates road networks cutting through forest |
| Spatial clustering | Remove edges in contiguous patches | Simulates agricultural frontier expansion |

**Simulation Parameters**
- Noise: Lévy α=1.5, σ=0.06 (cascade-triggering regime)
- Duration: 500 time units
- Time step: 0.5
- Ensemble: N=20 per fragmentation level (increased for statistical power)
- Replicate fragmentation: 5 independent random edge removals per level

**Metrics**
1. Tip/Recovery entropy ratio (primary indicator of buffering loss)
2. Total entropy production
3. Cascade extent (% cells tipped)
4. Network diameter and clustering coefficient at each fragmentation level
5. Time to first cascade event

### Expected Outcomes

```
Fragmentation%  |  Tip/Recovery Ratio  |  Interpretation
----------------|---------------------|------------------
100% (intact)   |  ≈ 1.0              |  Full buffering
90%             |  ≈ 1.0              |  Resilient
75%             |  ≈ 1.0 - 1.2        |  Minor degradation
50%             |  1.5 - 3.0          |  Buffering weakening
25%             |  10 - 100+          |  Buffering collapsed
10%             |  100 - 1000+        |  Isolated cell dynamics
```

### Implementation Notes

```python
def create_fragmented_network(base_network, retention_fraction, method='random'):
    """
    Create fragmented network by removing edges.

    Parameters
    ----------
    base_network : EnergyConstrainedNetwork
        Full 567-cell network
    retention_fraction : float
        Fraction of edges to retain (0.0 to 1.0)
    method : str
        'random', 'low_flow_first', 'high_betweenness_first', 'spatial'

    Returns
    -------
    EnergyConstrainedNetwork
        Network with reduced connectivity
    """
    # Implementation to be added to src/energy_constrained/network.py
```

### Deliverables

- [ ] `create_fragmented_network()` function in network module
- [ ] Notebook section 11: Network Fragmentation Experiments
- [ ] Fragmentation threshold plot (retention % vs tip/recovery ratio)
- [ ] Comparison of fragmentation methods
- [ ] Documentation in phase4_results.md

---

## Experiment 9: Recovery Dynamics with Asymmetric Mechanisms (REVISED)

### Objective

Test whether biologically-realistic asymmetric mechanisms create hysteresis and irreversibility in tipping cascades, aligning our model with observed ecosystem behavior.

### Scientific Rationale (Updated Based on Experiment 8 Findings)

**Experiment 8 Discovery**: Symmetric dynamics produce symmetric outcomes. Our fragmentation experiment showed tip/recovery ratio ≈ 1.0 at ALL retention levels, contradicting expectations of irreversibility.

**Industry Research Shows**: Real ecosystems exhibit strong asymmetry because:
1. **Biological timescales**: Trees grow in decades, burn in hours
2. **State-dependent feedbacks**: Fire begets fire, drought begets drought
3. **Climate trends**: Warming is unidirectional, not symmetric noise
4. **Biodiversity loss**: Recovery capacity degrades after tipping

**This Experiment Tests**: Whether adding these asymmetric mechanisms produces the hysteresis and irreversibility observed in real Amazon research.

### Asymmetric Mechanisms to Implement

#### Mechanism 1: Asymmetric Barrier Heights
Recovery requires more energy than tipping (trees grow slowly, die fast).

```python
# Standard (symmetric): barrier_tip = barrier_recovery = 0.5
# Asymmetric: barrier_recovery = 2.0 × barrier_tip
```

**Implementation**: New `AsymmetricCusp` class with separate barriers.

#### Mechanism 2: State-Dependent Coupling Degradation
Tipped cells provide less support to neighbors (dead forest can't recycle moisture).

```python
# Coupling strength from cell i to j:
# k_ij = k_base * (1.0 if x_i < 0 else degradation_factor)
# degradation_factor = 0.3 means tipped cells provide 30% support
```

**Implementation**: New `StateDependentCoupling` class.

#### Mechanism 3: Climate Trend Forcing
Unidirectional warming, not symmetric noise.

```python
# Add drift term to dynamics:
# c(t) = c_0 + trend_rate * t
# Shifts potential toward tipping over time
```

**Implementation**: Time-varying `c` parameter in cusp dynamics.

#### Mechanism 4: Recovery Capacity Degradation (Biodiversity Loss)
Cells that have tipped have reduced ability to recover (species loss).

```python
# Track tipping history:
# If cell has tipped before: recovery_barrier *= (1 + history_penalty)
```

**Implementation**: Stateful element tracking tipping history.

### Experimental Design

**Protocol: Cascade-then-Recovery with Asymmetric Dynamics**

```
Phase 1 (Cascade): t = 0 to 200
  - Apply Lévy α=1.5, σ=0.06
  - Asymmetric mechanisms ACTIVE
  - Trigger cascade (expect ~50% tipped)

Phase 2 (Recovery): t = 200 to 1000
  - Switch to Gaussian α=2.0, σ=0.02 (sub-tipping noise)
  - Remove climate trend (stabilize forcing)
  - Keep state-dependent coupling and barrier asymmetry
  - Track recovery of tipped cells
```

**Conditions (Factorial Design)**

| Condition | Barrier Asymmetry | Coupling Degradation | Climate Trend | Expected Hysteresis |
|-----------|-------------------|---------------------|---------------|---------------------|
| Symmetric baseline | 1.0× | None | None | ~1.0 (Exp 8 result) |
| Barrier only | 2.0× | None | None | Moderate (1.5-3×) |
| Coupling only | 1.0× | 0.3× | None | Moderate (1.5-3×) |
| Trend only | 1.0× | None | +0.001/t | Strong (>3×) |
| **Full realistic** | 2.0× | 0.3× | +0.001/t | **Very strong (>10×)** |

**Ensemble Size**: 20 runs per condition × 5 conditions = 100 simulations

### Metrics

1. **Tip/Recovery ratio**: PRIMARY metric for hysteresis
2. **Recovery fraction**: % of tipped cells that return to stable state by t=1000
3. **Recovery timescale**: Mean first-passage time from tipped to stable
4. **Permanent tips**: Cells that remain tipped (irreversibility measure)
5. **Path dependence**: Does final state depend on history, not just parameters?

### Expected Outcomes

| Condition | Tip/Recovery Ratio | Recovery Fraction | Interpretation |
|-----------|-------------------|-------------------|----------------|
| Symmetric baseline | ~1.0 | ~50% | Confirms Exp 8 |
| Barrier asymmetry | 1.5-3.0 | 30-40% | Partial hysteresis |
| Coupling degradation | 1.5-3.0 | 35-45% | Feedback loops |
| Climate trend | 3.0-10.0 | 20-30% | Directional forcing |
| **Full realistic** | **>10** | **<20%** | **Matches industry research** |

### Implementation Notes

#### New Classes Required

```python
# 1. AsymmetricBarrierCusp in elements.py
class AsymmetricBarrierCusp(EnergyConstrainedCusp):
    """
    Cusp with different barriers for tipping vs recovery.

    Parameters
    ----------
    barrier_tip : float
        Barrier height for stable → tipped transition
    barrier_recovery : float
        Barrier height for tipped → stable transition
    """
    def __init__(self, barrier_tip=0.5, barrier_recovery=1.0, ...):
        ...

    def potential(self, x: float) -> float:
        """Asymmetric double-well potential."""
        ...

# 2. StateDependentCoupling in couplings.py
class StateDependentCoupling(EnergyCoupling):
    """
    Coupling strength depends on source element state.

    Tipped elements provide reduced support to neighbors.

    Parameters
    ----------
    base_conductivity : float
        Coupling strength when source is stable
    degradation_factor : float
        Multiplier when source is tipped (0 < factor < 1)
    """
    def __init__(self, base_conductivity=0.1, degradation_factor=0.3):
        ...

# 3. TrendForcing in network.py or new module
class TrendForcing:
    """
    Time-varying forcing representing climate trend.

    Parameters
    ----------
    initial_c : float
        Initial forcing parameter
    trend_rate : float
        Rate of change per time unit (positive = toward tipping)
    """
    def get_forcing(self, t: float) -> float:
        return self.initial_c + self.trend_rate * t
```

#### Two-Phase Simulation Protocol

```python
def run_asymmetric_recovery_experiment(
    network,
    cascade_duration=200,
    recovery_duration=800,
    cascade_noise=(0.06, 1.5),  # (σ, α) Lévy
    recovery_noise=(0.02, 2.0),  # (σ, α) Gaussian
    barrier_asymmetry=2.0,       # recovery/tip barrier ratio
    coupling_degradation=0.3,    # tipped cell coupling factor
    climate_trend=0.001,         # forcing drift rate
    n_runs=20
):
    """
    Run cascade-then-recovery with asymmetric mechanisms.

    Returns
    -------
    dict with:
        - trajectories: Full state history
        - tip_recovery_ratio: Entropy ratio
        - recovery_fraction: % recovered
        - recovery_times: Time to recovery for each cell
        - permanent_tips: Cells that didn't recover
    """
```

### Deliverables

- [ ] `AsymmetricBarrierCusp` class in elements.py
- [ ] `StateDependentCoupling` class in couplings.py
- [ ] `TrendForcing` class for climate drift
- [ ] Two-phase simulation protocol in solvers.py
- [ ] Notebook 07: Recovery Dynamics with Asymmetric Mechanisms
- [ ] Comparison plot: symmetric vs asymmetric hysteresis
- [ ] Documentation in phase4_results.md

### Success Criteria

**Primary**: Full realistic condition produces tip/recovery ratio > 10 (matching industry hysteresis observations)

**Secondary**: Clear ranking of mechanism contributions:
- Quantify how much each mechanism contributes to asymmetry
- Identify which mechanism is most important for Amazon-specific dynamics

### References

- Bärtschi (2024): Ecosystem tipping symmetry depends on trait dissimilarity and driver dynamics
- Westen (2023): AMOC hysteresis asymmetry from state-dependent feedbacks
- Nature (2024): Amazon critical transitions require compound stressors
- Scheffer (2012): Anticipating critical transitions - hysteresis and early warnings

---

## Experiment 10: Lévy-Gaussian Transition Mapping (α-Sweep)

### Objective

Map the transition from Lévy to Gaussian behavior with fine resolution to identify critical α values and characterize the transition shape.

### Scientific Rationale

Phase 3 found qualitatively different behavior at α=1.2, 1.5, 1.8, 2.0. A finer sweep will:
1. Identify if there's a sharp critical α or gradual transition
2. Provide interpolation for estimating real-world climate noise α
3. Test universality of the transition across network types

### Experimental Design

**α Values**
- Range: α = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
- 10 values total, Δα = 0.1

**Simulation Parameters**
- Network: 50-cell Amazon subnetwork (faster than 567-cell)
- σ = 0.06 (fixed)
- Duration: 500 time units
- Time step: 0.5
- Ensemble: N=30 per α value (total: 300 simulations)

**Metrics at Each α**
1. Total entropy production
2. Tip/Recovery entropy ratio
3. Number of tipping events
4. % time in tipped state
5. Largest Lyapunov exponent (if time permits)

### Expected Outcomes

```
α    |  Entropy  |  Tip/Rec Ratio  |  Regime
-----|-----------|-----------------|----------
1.1  |  Very high|  >> 1           |  Extreme Lévy
1.2  |  High     |  >> 1           |  Strong Lévy
1.3  |  High     |  > 1            |  Lévy
1.4  |  Medium   |  > 1            |  Lévy
1.5  |  Medium   |  ~1             |  Transition?
1.6  |  Medium   |  ~1             |  Transition?
1.7  |  Low      |  ~1             |  Near-Gaussian
1.8  |  Low      |  ~1             |  Near-Gaussian
1.9  |  Very low |  ~1             |  Gaussian-like
2.0  |  Minimal  |  N/A (no events)|  Gaussian
```

### Implementation Notes

```python
def run_alpha_sweep(
    network,
    alpha_values=np.arange(1.1, 2.05, 0.1),
    sigma=0.06,
    n_runs=30,
    duration=500,
    use_dask=True
):
    """
    Sweep across α values to map Lévy-Gaussian transition.

    Returns DataFrame with metrics at each α.
    """
```

### Deliverables

- [ ] α-sweep analysis function
- [ ] Notebook section 13: Lévy-Gaussian Transition Mapping
- [ ] Transition plot (metrics vs α)
- [ ] Critical α identification (if sharp transition exists)
- [ ] Documentation in phase4_results.md

---

## Experiment 11: Keystone Connection Identification

### Objective

Identify which network connections are disproportionately important for maintaining thermodynamic buffering.

### Scientific Rationale

Not all connections contribute equally to network resilience. Identifying "keystone" connections has:
1. Scientific value: Understanding network structure-function relationships
2. Policy value: Prioritizing conservation of critical forest corridors

### Experimental Design

**Approach: Single-Edge Removal Sensitivity**

For each edge in the top 100 highest-flow connections:
1. Remove that single edge
2. Run ensemble (N=10)
3. Compute tip/recovery ratio
4. Compare to intact network baseline

**Edge Metrics to Correlate**
- Flow magnitude (moisture transport)
- Betweenness centrality (network position)
- Connected cell degrees (hub connections)
- Spatial position (east-west, interior-coastal)

**Simulation Parameters**
- Network: 50-cell subnetwork (tractable for 100 single-edge experiments)
- Noise: Lévy α=1.5, σ=0.06
- Ensemble: N=10 per edge removal
- Total simulations: ~1000

### Expected Outcomes

Identify edges where removal causes:
- Tip/Recovery ratio > 1.5 (significant buffering loss)
- > 20% increase in cascade extent

Correlate with edge properties to develop predictive model.

### Deliverables

- [ ] Single-edge sensitivity analysis function
- [ ] Notebook section 14: Keystone Connection Analysis
- [ ] Ranked list of critical edges
- [ ] Correlation analysis (which edge properties predict importance)
- [ ] Spatial map of keystone corridors
- [ ] Documentation in phase4_results.md

---

## Experiment 12: Publication-Quality Ensembles

### Objective

Run larger ensembles (N=50-100) for key comparisons to achieve statistical significance for publication.

### Target Comparisons

| Comparison | Conditions | N per Condition | Total Sims |
|------------|------------|-----------------|------------|
| Lévy vs Gaussian | α=1.5 vs α=2.0 | 100 | 200 |
| Drought impact | 2003 vs 2010, Lévy α=1.5 | 100 | 200 |
| Network scale | 50-cell vs 567-cell | 50 | 100 |
| Fragmentation threshold | 100%, 50%, 25% retention | 50 | 150 |

**Total: ~650 simulations**

### Statistical Analysis

For each comparison:
1. Two-sample t-test or Mann-Whitney U for means
2. Levene's test for variance differences
3. Effect size (Cohen's d)
4. Bootstrap confidence intervals (95%)

### Deliverables

- [ ] Publication-quality ensemble runs
- [ ] Statistical analysis notebook
- [ ] Summary table with p-values and effect sizes
- [ ] Publication-ready figures

---

## Timeline and Resource Estimates

### Computational Resources

| Experiment | Simulations | Est. Runtime (7 workers) |
|------------|-------------|--------------------------|
| Exp 8: Fragmentation | 600 | ~3 hours |
| Exp 9: Recovery | 80 | ~1 hour |
| Exp 10: α-Sweep | 300 | ~2 hours |
| Exp 11: Keystone | 1000 | ~5 hours |
| Exp 12: Publication | 650 | ~4 hours |
| **Total** | **2630** | **~15 hours** |

### Suggested Order

```
Week 1:
├── Exp 8: Network Fragmentation (highest priority)
└── Exp 9: Recovery Dynamics

Week 2:
├── Exp 10: α-Sweep
└── Exp 11: Keystone Connections

Week 3:
├── Exp 12: Publication Ensembles
└── Documentation and analysis
```

---

## Success Criteria

### Scientific Milestones

| Milestone | Criterion | Status |
|-----------|-----------|--------|
| Fragmentation threshold identified | Clear inflection point in tip/recovery ratio | Pending |
| Recovery dynamics characterized | Quantified difference in 2003 vs 2010 recovery | Pending |
| Lévy-Gaussian transition mapped | Critical α identified with error bounds | Pending |
| Keystone connections identified | Top 10 edges ranked by importance | Pending |
| Publication-ready statistics | p < 0.05 for key comparisons | Pending |

### Technical Milestones

| Milestone | Criterion | Status |
|-----------|-----------|--------|
| Fragmentation functions implemented | Unit tests passing | Pending |
| Two-phase simulation protocol | Validated against single-phase | Pending |
| Dask parallelization stable | No ModuleNotFoundError | ✓ Complete |
| Ensemble automation | Batch submission working | Pending |

---

## Experiment 15: Coupling Restoration Dynamics ⭐ NEW (Based on Exp 9 Findings)

### Objective

Test whether **restoring coupling strength** during recovery is more effective than direct cell forcing, given the finding that coupling degradation is the dominant hysteresis mechanism.

### Scientific Rationale (From Experiment 9)

Experiment 9 revealed a critical finding:
- **Coupling degradation ALONE produces 100% recovery failure** (0% recovery)
- Barrier asymmetry alone produces 68% reduction
- Combined mechanisms produce 87-89% reduction

This means coupling degradation is the **dominant** mechanism. Therefore, interventions that restore coupling may be more effective than direct forcing.

**Physical Interpretation**:
- Direct forcing = planting trees in individual cells
- Coupling restoration = reforestation corridors that reconnect moisture recycling
- Network hub restoration = protecting/restoring high-connectivity regions

### Experimental Design

**Protocol: Cascade → Intervention → Recovery**

```
Phase 1 (Cascade): t = 0 to 200
  - Lévy α=1.5, σ=0.06
  - State-dependent coupling degradation ACTIVE (factor=0.3)
  - Trigger cascade (~50% tipped)

Phase 2 (Intervention): t = 200 to 1000
  - Gaussian α=2.0, σ=0.04 (recovery noise)
  - Compare intervention strategies:
```

**Intervention Conditions**

| Condition | Direct Forcing | Coupling Restoration | Target |
|-----------|---------------|---------------------|--------|
| No intervention | None | None | Baseline (Exp 9) |
| Direct forcing only | f = -0.1 | None | All cells |
| **Coupling restoration only** | None | Restore to 1.0 | All edges |
| Hub coupling restoration | None | Restore to 1.0 | Top 10% betweenness edges |
| Corridor restoration | None | Restore to 1.0 | East-West moisture pathways |
| Combined (direct + coupling) | f = -0.05 | Restore to 0.6 | All |

**Coupling Restoration Implementation**

```python
class RestorableCoupling(StateDependentCoupling):
    """
    Coupling that can be restored during recovery phase.

    Parameters
    ----------
    base_conductivity : float
        Normal coupling strength
    degradation_factor : float
        Multiplier when source is tipped (0.3 = 30% support)
    restoration_factor : float
        Target coupling during intervention (1.0 = full restoration)
    restoration_rate : float
        Rate at which coupling restores toward target
    """

    def restore(self, dt: float):
        """Gradually restore coupling toward restoration_factor."""
        current = self.get_effective_factor()
        target = self.restoration_factor
        self.effective_factor += self.restoration_rate * (target - current) * dt
```

**Simulation Parameters**

| Parameter | Value |
|-----------|-------|
| Network | 50-cell Amazon subnetwork |
| Conditions | 6 |
| Runs per condition | 20 |
| Total simulations | 120 |
| Cascade duration | 200 time units |
| Recovery duration | 800 time units |
| Dask workers | 14 |

### Metrics

1. **Recovery fraction**: Primary outcome (compare to Exp 9 baseline of 6%)
2. **Recovery efficiency**: Recovery fraction per unit intervention cost
3. **Time to 50% recovery**: How quickly does intervention work?
4. **Spatial recovery pattern**: Do hub interventions create spreading recovery?
5. **Intervention cost**: Total forcing energy or coupling restoration effort

### Expected Outcomes

| Condition | Expected Recovery | Rationale |
|-----------|-------------------|-----------|
| No intervention | ~6% | Exp 9 baseline |
| Direct forcing only | ~40% | From Exp 10c |
| **Coupling restoration only** | **>50%?** | If coupling dominates, this should be highly effective |
| Hub coupling restoration | ~40-60% | Targeted efficiency |
| Corridor restoration | ~30-50% | Spatial realism |
| Combined | >70% | Synergistic effects |

### Key Questions

1. **Is coupling restoration more efficient than direct forcing?**
   - Compare recovery per unit cost

2. **Does spatial targeting improve efficiency?**
   - Hub restoration vs uniform restoration

3. **Are there synergies between intervention types?**
   - Combined > sum of parts?

4. **How does restoration rate affect outcome?**
   - Gradual vs immediate restoration

### Implementation Notes

```python
def run_coupling_restoration_experiment(
    network,
    cascade_duration=200,
    recovery_duration=800,
    intervention_type='coupling',  # 'direct', 'coupling', 'combined'
    restoration_target=1.0,        # For coupling restoration
    restoration_rate=0.01,         # Gradual restoration
    direct_forcing=-0.1,           # For direct forcing
    target_edges='all',            # 'all', 'hubs', 'corridors'
    n_runs=20
):
    """
    Run experiment with coupling restoration intervention.
    """
```

### Deliverables

- [ ] `RestorableCoupling` class in couplings.py
- [ ] Hub/corridor edge identification functions
- [ ] Notebook 15: Coupling Restoration Dynamics
- [ ] Comparison plot: intervention effectiveness
- [ ] Cost-effectiveness analysis
- [ ] Policy recommendations for Amazon restoration

### Policy Implications

If coupling restoration is more effective:
1. **Prioritize forest corridors** over isolated reforestation
2. **Protect moisture recycling pathways** even if cells are degraded
3. **Target high-connectivity hubs** for maximum impact
4. **Sequential strategy**: Restore coupling first, then direct forcing

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Fragmentation doesn't show threshold | Medium | High | Try multiple removal strategies |
| Recovery takes too long to simulate | Low | Medium | Truncate at partial recovery |
| α-sweep shows no clear transition | Medium | Medium | Report gradual transition if found |
| Keystone analysis inconclusive | Medium | Low | Focus on other experiments |
| Computational time exceeds estimates | Medium | Low | Prioritize high-impact experiments |
| Coupling restoration ineffective | Low | Medium | Still valuable negative result |

---

## Documentation Plan

### Phase 4 Results Document

Structure for `docs/phase4_results.md`:

```markdown
# Phase 4 Results: Network Resilience and Recovery Dynamics

## Experiment 8: Network Fragmentation
### Configuration
### Results
### Key Findings

## Experiment 9: Recovery Dynamics
...

## Experiment 10: α-Sweep
...

## Experiment 11: Keystone Connections
...

## Experiment 12: Publication Statistics
...

## Synthesis: Integrated Findings
### What We Learned
### Policy Implications
### Limitations
### Future Directions
```

### Notebook Organization

```
notebooks/
├── 04_energy_constrained_exploration.ipynb  # Phase 3 idealized
├── 05_amazon_spatial_energy_tracking.ipynb  # Phase 3 spatial
├── 06_network_fragmentation.ipynb           # Exp 8 (NEW)
├── 07_recovery_dynamics.ipynb               # Exp 9 (NEW)
├── 08_levy_gaussian_transition.ipynb        # Exp 10 (NEW)
├── 09_keystone_connections.ipynb            # Exp 11 (NEW)
└── 10_publication_figures.ipynb             # Exp 12 (NEW)
```

---

## References

### Key Citations for Phase 4

1. **Network resilience theory**: Albert, R., & Barabási, A. L. (2002). Statistical mechanics of complex networks. Reviews of Modern Physics, 74(1), 47.

2. **Lévy flights in ecology**: Viswanathan, G. M., et al. (1999). Optimizing the success of random searches. Nature, 401(6756), 911-914.

3. **Amazon tipping points**: Lovejoy, T. E., & Nobre, C. (2018). Amazon tipping point. Science Advances, 4(2), eaat2340.

4. **Network fragmentation**: Bodin, Ö., & Saura, S. (2010). Ranking individual habitat patches as connectivity providers. Ecological Complexity, 7(2), 176-183.

5. **Recovery dynamics**: Scheffer, M., et al. (2012). Anticipating critical transitions. Science, 338(6105), 344-348.

---

*Document Version 1.0 — December 2025*

*This plan guides Phase 4 experiments building on Phase 3 discoveries.*
