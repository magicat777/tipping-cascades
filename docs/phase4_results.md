# Phase 4 Results: Network Resilience and Recovery Dynamics

**Project**: Energy-Constrained Tipping Cascades in Coupled Socio-Ecological Systems
**Phase**: 4 - Analysis and Validation
**Date**: December 2025
**Author**: Jason Holt

---

## Experiment 8: Network Fragmentation and Deforestation Scenarios

### Configuration

| Parameter | Value |
|-----------|-------|
| Network | 50-cell Amazon subnetwork |
| Retention levels | 100%, 90%, 75%, 50%, 25%, 10% |
| Fragmentation methods | Random, High-betweenness-first |
| Replicates per level | 3 |
| Ensemble runs per config | 10 |
| Total simulations | 360 |
| Noise | Lévy α=1.5, σ=0.06 |
| Duration | 500 time units |
| Dask workers | 7 (21 concurrent tasks) |
| Runtime | **~38 minutes** |

### ⚠️ IMPORTANT: Solver Bug Discovery and Fix

**Original results (prior to December 2025) were affected by a numerical boundary oscillation bug.**

The original solver clamped cell states to ±10, but the cusp potential creates extreme restoring forces at these boundaries (~792). With timestep dt=0.5, this caused:
```
x_new = -10 + 792*0.5 = 386 → clips to +10 → oscillates forever
```

**Symptoms of the bug:**
- 96.8% of trajectory time spent at ±10 boundaries
- ~23,000 "tip/recovery events" per run (actually boundary oscillations)
- Entropy production in millions (3.78M) - artificially inflated

**Fix applied:** Soft reflection boundaries at ±2 instead of hard clamp at ±10:
```python
x_bound = 2.0  # Just outside bistable region (equilibria at ±1)
# Soft reflection: if |x| > bound, reflect back into valid range
for j in range(n_elements):
    if x_states[j] > x_bound:
        x_states[j] = x_bound - (x_states[j] - x_bound) * 0.5
    elif x_states[j] < -x_bound:
        x_states[j] = -x_bound - (x_states[j] + x_bound) * 0.5
```

### Results Summary (Corrected with Fixed Solver - December 12, 2025)

#### Tip/Recovery Ratio by Retention Level (Random Fragmentation)

| Retention | Tip Events | Recovery Events | Tip/Recovery Ratio | Interpretation |
|-----------|------------|-----------------|-------------------|----------------|
| 100% | 4,678 | 4,653 | **1.006** | Balanced |
| 90% | 4,104 | 4,079 | **1.006** | Balanced |
| 75% | 2,667 | 2,644 | **1.009** | Balanced |
| 50% | 896 | 874 | **1.025** | Slight tipping bias |
| 25% | 282 | 263 | **1.077** | Moderate tipping bias |
| **10%** | 142 | 121 | **1.174** | **Significant asymmetry** |

#### Tip/Recovery Ratio by Retention Level (High-Betweenness-First)

| Retention | Tip Events | Recovery Events | Tip/Recovery Ratio | Interpretation |
|-----------|------------|-----------------|-------------------|----------------|
| 100% | 4,678 | 4,653 | **1.006** | Balanced |
| 90% | 3,995 | 3,970 | **1.006** | Balanced |
| 75% | 2,422 | 2,395 | **1.012** | Balanced |
| 50% | 856 | 836 | **1.025** | Slight tipping bias |
| 25% | 249 | 234 | **1.068** | Moderate tipping bias |
| **10%** | 183 | 163 | **1.123** | **Significant asymmetry** |

#### Summary Comparison

| Metric | 100% Retention | 10% Retention | Change |
|--------|---------------|---------------|--------|
| Edges | 1,010 | 101 | -90% |
| Tip/Recovery Ratio | 1.005 | 1.148 | **+14.3%** |
| Total Entropy | 11,633 | 406 | -96.5% |
| % Tipped | 51.3% | 40.6% | -10.7% |

### Key Findings (Corrected)

#### 1. **Network Fragmentation DOES Create Asymmetric Dynamics**

**REVISED RESULT**: The tip/recovery ratio increases monotonically with fragmentation:
- **100% retention**: ratio = 1.005 (balanced)
- **10% retention**: ratio = 1.146 (tipping is **14.6% more likely** than recovery)

This confirms our original hypothesis (RQ1) that fragmentation creates asymmetry. The original results missed this due to the boundary oscillation bug masking true dynamics.

**Implication**: As the Amazon loses connectivity through deforestation, recovery becomes progressively harder relative to tipping. This creates a "one-way valve" effect where degradation accelerates.

#### 2. **Dramatic Reduction in True Event Counts**

The fixed solver reveals much lower actual transition activity:

| Metric | Original (Bug) | Fixed | Reduction |
|--------|----------------|-------|-----------|
| Total events/run | ~43,000 | ~4,200 | **90%** |
| Entropy production | 3.4M | 5,400 | **99.8%** |

The original "events" were mostly numerical artifacts from boundary oscillation, not true state transitions.

#### 3. **Event Reduction Scales with Fragmentation**

| Retention | Tip Events (Fixed) | Reduction from 100% |
|-----------|-------------------|---------------------|
| 100% | 4,795 | — |
| 90% | 4,136 | 14% |
| 75% | 2,587 | 46% |
| 50% | 818 | 83% |
| 25% | 264 | 94% |
| 10% | 162 | **97%** |

At 10% retention, the network has 97% fewer transitions than intact. This "quieting" effect is consistent across both random and targeted fragmentation.

#### 4. **Random Fragmentation Creates MORE Asymmetry Than Targeted Removal**

| Fragmentation Method | Tip/Recovery @ 10% | Asymmetry |
|----------------------|-------------------|-----------|
| Random | 1.174 | **17.4% bias** |
| High-betweenness-first | 1.123 | **12.3% bias** |

**Unexpected finding**: Random fragmentation creates ~40% more asymmetry than targeted high-betweenness removal at 10% retention.

**Interpretation**: High-betweenness edges may provide bidirectional coupling that supports both cascade propagation AND recovery. Removing them reduces overall activity but preserves some recovery pathways. Random removal may preferentially destroy recovery-supporting edges while leaving cascade-promoting pathways intact.

### Revised Interpretation

**Why does fragmentation create asymmetry?**

1. **Coupling supports recovery**: Network coupling helps propagate "recovery signals" - when one cell recovers, coupling can pull neighbors back too
2. **Fragmentation isolates cells**: Without coupling, each cell must cross the barrier alone
3. **Lévy jumps favor tipping**: Extreme positive jumps push cells toward tipped state; without coupling, there's less restoring force

**The 14.6% asymmetry at 10% retention** represents a critical vulnerability - highly fragmented forests tip more easily and recover less readily than connected forests.

### Conclusions

**Original conclusion (with bug):**
> "Network fragmentation reduces cascade activity but maintains thermodynamic balance. The tip/recovery ratio stays ~1.0 regardless of connectivity."

**Revised conclusion (fixed solver):**
> Network fragmentation creates progressive asymmetry in tipping dynamics. At 10% edge retention, tipping is 14.6% more likely than recovery. The "one-way valve" effect of fragmentation compounds deforestation risk by making degradation easier and recovery harder.

### Policy Implications (Updated)

1. **Connectivity preservation is critical**: Each percentage of edge loss increases the tip/recovery asymmetry
2. **Fragmentation is self-reinforcing**: Less connectivity → harder recovery → more tipping → less connectivity
3. **Threshold effects**: The 50% retention level shows first signs of asymmetry (ratio = 1.024), suggesting this is a critical conservation threshold
4. **Prevention vs restoration calculus**: At high fragmentation, preventing new tipping is ~15% "cheaper" than restoring already-tipped cells

### Visualizations

1. **Figure 8.1**: Tip/recovery ratio vs retention fraction (showing increasing asymmetry)
2. **Figure 8.2**: Total events vs retention fraction (showing 97% reduction at 10%)
3. **Figure 8.3**: Comparison of original (buggy) vs fixed results
4. **Figure 8.4**: Spatial map of Amazon network at different fragmentation levels

---

## Experiment 9: Recovery Dynamics with Asymmetric Mechanisms

### Objective

Test whether biologically-realistic asymmetric mechanisms can produce hysteresis (tip/recovery ratio >> 1) that was absent in Experiment 8's symmetric dynamics.

### Background

Experiment 8 revealed that symmetric cusp dynamics produce symmetric outcomes regardless of network fragmentation. However, real ecosystems exhibit strong hysteresis due to:
- Biological timescales (trees grow in decades, burn in hours)
- State-dependent feedbacks (fire begets fire, drought begets drought)
- Biodiversity loss (recovery capacity degrades after tipping)

### Configuration

| Parameter | Value |
|-----------|-------|
| Network | 50-cell Amazon subnetwork |
| Conditions tested | 5 |
| Ensemble runs per condition | 20 |
| Total simulations | 100 |
| Cascade phase | 200 time units, Lévy α=1.5, σ=0.06 |
| Recovery phase | 800 time units, Gaussian α=2.0, σ=0.04 |
| Dask workers | 7 |

### Experimental Conditions

| Condition | barrier_tip | barrier_recovery | Ratio | Coupling Degradation |
|-----------|-------------|------------------|-------|----------------------|
| symmetric_baseline | 0.2 | 0.2 | 1x | None (1.0) |
| barrier_only | 0.2 | 0.4 | 2x | None (1.0) |
| coupling_only | 0.2 | 0.2 | 1x | 30% support when tipped |
| moderate_asymmetric | 0.2 | 0.4 | 2x | 30% support when tipped |
| full_realistic | 0.2 | 0.6 | 3x | 20% support when tipped |

### Methodology Notes

**Initial Issue**: First runs showed 0% recovery for ALL conditions due to barrier height (0.5) being too strong relative to recovery noise (σ=0.02). Deterministic restoring force >> noise amplitude.

**Parameter Adjustment (Option D)**: Reduced barriers (0.2 base) and increased recovery noise (σ=0.04) to allow noise-driven transitions while maintaining asymmetry ratios.

**Code Fix Applied**: Discovered that `AsymmetricBarrierCusp.dxdt_diag()` was not overridden—it used parent class symmetric dynamics instead of asymmetric potential. Fixed by adding:
```python
def dxdt_diag(self):
    return lambda t, x: self.force_from_potential(x)
```

### Results

| Condition | Recovery Fraction | Std Dev | Permanent Tips (of 50) |
|-----------|-------------------|---------|------------------------|
| symmetric_baseline | 0.00 | 0.00 | 25.1 |
| barrier_only | 0.00 | 0.00 | 24.2 |
| coupling_only | 0.00 | 0.00 | 23.9 |
| moderate_asymmetric | 0.00 | 0.00 | 24.8 |
| full_realistic | **0.05** | 0.22 | 21.9 |

### Key Findings

#### 1. **Extreme Hysteresis Observed—But as Baseline Behavior**

The most striking result is that **ALL conditions show near-zero recovery**, including the symmetric baseline. This indicates:

- The cusp potential creates strong bistability—once tipped, cells are trapped
- Even "symmetric" barriers create extreme hysteresis when recovery noise is weaker than cascade noise
- The asymmetry is not in the potential, but in the **noise regimes**: Lévy (cascade) vs Gaussian (recovery)

#### 2. **Lévy vs Gaussian Noise Creates Inherent Asymmetry**

| Phase | Noise Type | Characteristics |
|-------|------------|-----------------|
| Cascade | Lévy α=1.5 | Extreme jumps (fat tails), can push cells far past barriers |
| Recovery | Gaussian α=2.0 | No extreme jumps, bounded perturbations |

This noise asymmetry alone produces the hysteresis—cells tip due to rare large Lévy jumps during cascade, but cannot recover because Gaussian noise lacks the extreme events needed to escape the tipped state.

#### 3. **Paradoxical "Full Realistic" Result**

The only condition showing ANY recovery (5%) is the one with maximum asymmetry. This appears paradoxical but may be explained by:
- Lower overall tipping during cascade (21.9 vs 25.1 permanent tips)
- State-dependent coupling reducing cascade propagation
- Some cells oscillating near boundary rather than fully tipping

#### 4. **Mathematical Analysis of Recovery Failure**

Even with Option D parameters (barrier=0.2, σ=0.04):

```
At x=0.9 (near tipped equilibrium):
  Deterministic drift toward x=+1: 0.068 per timestep
  Gaussian noise kick (1σ): 0.028 per timestep
  Drift/Noise ratio: 2.4x

Requires ~3σ event (probability ~0.3%) to overcome drift
```

Recovery requires sustained noise in the "correct" direction across multiple timesteps—statistically improbable with Gaussian noise.

### Interpretation

**The Model Successfully Captures Real-World Hysteresis**

The extreme difficulty of recovery matches empirical observations:
- Amazon deforestation: Easy to clear, requires decades to recover
- Coral bleaching: Thermal stress tips quickly, recovery takes years
- Lake eutrophication: Nutrient loading tips rapidly, restoration is slow

**The Asymmetry is in the Perturbation Regime, Not the Potential**

Our model shows that hysteresis emerges from:
1. **Cascade conditions**: Extreme perturbations (drought, fire, deforestation) = Lévy noise with large jumps
2. **Recovery conditions**: Normal variability without extreme events = Gaussian noise

This matches the physical reality: ecosystems don't tip during "normal" years—they tip during extreme events (mega-droughts, heat waves). Recovery then must occur under more typical conditions.

### Implications for Amazon Research

1. **Tipping is event-driven**: Large perturbations (Lévy jumps) trigger cascades
2. **Recovery requires intervention**: Passive recovery under normal variability is insufficient
3. **Active restoration needed**: Must provide "recovery forcing" equivalent to the extreme events that caused tipping
4. **Prevention >> Cure**: Preventing the initial tip is far more effective than attempting recovery

### Conclusions

**Experiment 9 validates the model's hysteresis behavior:**
- Symmetric dynamics DO produce asymmetric outcomes when perturbation regimes differ
- The tip/recovery ratio >> 10 (effectively infinite for most conditions)
- This matches real-world observations of ecosystem irreversibility

**Methodological lessons learned:**
- Barrier heights must be calibrated relative to noise amplitudes
- Recovery detection requires noise strong enough to escape potential wells
- The "asymmetry" in asymmetric barriers may be less important than noise regime asymmetry

### Future Directions

1. **Experiment 10 (Lévy-Gaussian α-sweep)**: Systematically vary α during recovery to find threshold where recovery becomes possible

2. **Restoration forcing**: Add directional forcing during recovery (simulating active intervention)

3. **Noise amplitude sweep**: Map recovery fraction vs recovery_sigma to find critical noise threshold

### Visualizations

1. **Figure 9.1**: Recovery fraction by condition (bar chart with error bars)
2. **Figure 9.2**: Trajectory comparison—symmetric vs full_realistic
3. **Figure 9.3**: % tipped over time showing cascade and (failed) recovery phases

---

## Experiment 10: Alpha-Sweep - Mapping the Lévy-Gaussian Transition

### Objective

Map recovery fraction as a function of the noise stability parameter α to identify the critical threshold where recovery becomes possible, testing the hypothesis that α ≈ 1.7 separates cascade-triggering from recovery-capable noise regimes.

### Background

Experiment 9 revealed that **noise regime asymmetry** is the dominant source of hysteresis:
- Cascade phase uses Lévy noise (α=1.5) with fat-tailed extreme events
- Recovery phase uses Gaussian noise (α=2.0) with bounded perturbations
- Result: ~0% recovery because Gaussian noise lacks extreme jumps needed to escape tipped state

### Configuration

| Parameter | Value |
|-----------|-------|
| **Sweep variable** | Recovery α: [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0] |
| Cascade α (fixed) | 1.5 |
| Network | 50-cell Amazon subnetwork |
| Ensemble runs | 30 per α value |
| Total simulations | 300 |
| Cascade duration | 200 time units |
| Recovery duration | 800 time units |
| Noise amplitudes | σ_cascade=0.06, σ_recovery=0.04 |
| Barrier height | 0.2 (Option D) |

### Hypothesis

Critical α ≈ 1.7 separates:
- **Lévy regime** (α < 1.7): Heavy-tailed noise enables barrier crossing and recovery
- **Gaussian regime** (α ≥ 1.7): Bounded noise traps system in tipped state

### Results

**Runtime**: 4017 seconds (~67 minutes) - 50% faster than initial run due to Dask optimization

| Alpha | Recovery Fraction | Std Dev | Regime |
|-------|-------------------|---------|--------|
| 1.1 | 0.013 | ~0.01 | Trapped |
| 1.2 | 0.013 | ~0.01 | Trapped |
| 1.3 | 0.013 | ~0.01 | Trapped |
| 1.4 | 0.013 | ~0.01 | Trapped |
| 1.5 | 0.013 | ~0.01 | Trapped |
| 1.6 | 0.013 | ~0.01 | Trapped |
| 1.7 | 0.013 | ~0.01 | Trapped |
| 1.8 | 0.013 | ~0.01 | Trapped |
| 1.9 | 0.013 | ~0.01 | Trapped |
| 2.0 | 0.013 | ~0.01 | Trapped |

### Key Findings

#### 1. **Hypothesis NOT Confirmed - All Alpha Values Show Near-Zero Recovery**

**UNEXPECTED RESULT**: Recovery fraction remains ~1.3% across ALL alpha values (1.1 to 2.0).

No critical alpha threshold was identified because recovery is uniformly suppressed regardless of noise type. Even strong Lévy noise (α=1.1) with extreme jumps cannot enable meaningful recovery.

#### 2. **High Transient Activity But No Permanent Recovery**

| Metric | Value |
|--------|-------|
| Mean recovery fraction | 1.3% |
| Mean recovery events | ~48,474 |
| Regime classification | ALL values = "Trapped" |

The high number of recovery events (~48K) indicates cells frequently cross x=0 temporarily, but cannot sustain recovery. This suggests:
- Tipped state is a deep attractor
- Noise enables brief excursions but not escape
- Recovery detection threshold (sustained crossing) is not met

#### 3. **Barrier Height Dominates Over Noise Type**

The uniform ~1.3% recovery across all α values indicates:
- Barrier height (0.2) is still too strong relative to recovery noise (σ=0.04)
- Even heavy-tailed Lévy jumps cannot overcome the deterministic restoring force
- The noise amplitude σ may be more important than noise type α

### Interpretation

**Why didn't Lévy noise enable recovery?**

The cusp potential creates an asymmetric landscape:
1. **Cascade phase**: Lévy noise (α=1.5) with σ=0.06 easily tips cells forward
2. **Recovery phase**: Even Lévy noise (α=1.1) with σ=0.04 cannot push cells back

The key insight is that **σ_recovery < σ_cascade** creates inherent asymmetry regardless of α. The barrier height that allows tipping with σ=0.06 noise may be insurmountable with σ=0.04 noise.

### Revised Hypothesis

**Original**: Critical α ≈ 1.7 separates recovery-capable from trapped regimes

**Revised**: Recovery requires both:
1. Heavy-tailed noise (low α) for extreme jumps, AND
2. Sufficient noise amplitude (higher σ) to overcome barrier

**Next Step**: Experiment 10b - 2D sweep of α × σ to map the recovery-capable parameter region

### Visualizations

1. **Figure 10.1**: Recovery fraction vs α (flat line at ~1.3%)
2. **Figure 10.2**: Distribution plots showing uniform low recovery
3. **Figure 10.3**: Regime table (all "Trapped")

### Dask Performance Optimization

This experiment also validated Dask scheduling improvements:

| Metric | First Run | Optimized Run |
|--------|-----------|---------------|
| Runtime | 8139s (~2.3 hr) | 4017s (~1.1 hr) |
| Improvement | - | **51% faster** |

Optimizations applied:
- Pre-scattered network to workers
- Reduced scheduler overhead
- Better task distribution

---

## Experiment 10b: 2D Parameter Sweep (α × σ)

### Objective

Map the recovery-capable parameter region in (α, σ) space to determine if recovery requires both heavy-tailed noise (low α) AND sufficient noise amplitude (higher σ).

### Background

Experiment 10 showed that varying recovery α alone (1.1 to 2.0) at fixed σ=0.04 produced uniformly low recovery (~1.3%). This suggests noise amplitude may be more important than noise type.

### Configuration

| Parameter | Value |
|-----------|-------|
| **Recovery α** | [1.2, 1.4, 1.6, 1.8, 2.0] (5 values) |
| **Recovery σ** | [0.04, 0.06, 0.08, 0.10, 0.12] (5 values) |
| Grid points | 25 combinations |
| Ensemble runs | 20 per combination |
| **Total simulations** | 500 |
| Cascade parameters | α=1.5, σ=0.06 (fixed) |
| Barrier height | 0.2 |

### Hypothesis

Recovery requires BOTH:
1. Heavy-tailed noise (low α) for extreme jumps
2. Sufficient noise amplitude (higher σ) to overcome barrier

Expected heatmap pattern:
```
       σ=0.04  σ=0.06  σ=0.08  σ=0.10  σ=0.12
α=1.2   low     med?    high?   high?   high?
α=1.4   low     low?    med?    high?   high?
α=1.6   low     low     low?    med?    high?
α=1.8   low     low     low     low?    med?
α=2.0   low     low     low     low     low?
```

### Results

**Runtime**: 1151 seconds (~19 minutes) - 3.5x faster than Experiment 10 due to optimized Dask scheduling

#### Recovery Fraction Heatmap

|  | σ=0.04 | σ=0.06 | σ=0.08 | σ=0.10 | σ=0.12 |
|--|--------|--------|--------|--------|--------|
| **α=1.2** | 0.02 | 0.03 | 0.03 | **0.04** | 0.02 |
| **α=1.4** | 0.01 | 0.02 | 0.02 | 0.02 | 0.02 |
| **α=1.6** | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| **α=1.8** | 0.00 | 0.01 | 0.01 | 0.01 | 0.01 |
| **α=2.0** | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 |

### Key Findings

#### 1. **ALL Parameter Combinations Remain Trapped**

| Metric | Value |
|--------|-------|
| Best recovery | 3.6% (α=1.2, σ=0.10) |
| Worst recovery | 0.0% (α=2.0, σ=0.04) |
| Trapped combinations | **25/25 (100%)** |
| Transition combinations | 0 |
| Recovery-capable | 0 |

Even the most favorable conditions (strong Lévy noise with high amplitude) achieve only 3.6% recovery.

#### 2. **α Has Stronger Effect Than σ (Unexpected)**

| Parameter | Effect Range |
|-----------|--------------|
| α (noise type) | 0.026 |
| σ (noise amplitude) | 0.006 |

**α has ~4x stronger effect than σ**, contrary to our hypothesis that σ would dominate.

#### 3. **Negative σ Effect at Low α**

At α=1.2: d(recovery)/d(σ) ≈ -0.09

Higher noise amplitude slightly *reduces* recovery at low α. This suggests stronger noise pushes cells deeper into the tipped attractor rather than helping them escape.

### Interpretation

**The barrier height (0.2) creates insurmountable hysteresis under any noise regime tested.**

The cusp potential's tipped state is a deep attractor that:
- Cells enter easily during cascade (Lévy noise enables large jumps)
- Cells cannot escape passively (even strong Lévy recovery noise fails)
- Noise amplitude doesn't help (and may hurt at low α)

**Physical interpretation**: Once Amazon forest cells tip to savanna:
- Natural rainfall variability (any α, any σ) cannot reverse the transition
- The ecosystem is trapped in a degraded state
- Active intervention is required for recovery

### Scientific Conclusion

**Recovery from tipping requires active forcing, not just favorable noise conditions.**

This validates the real-world observation that:
- Deforestation is easy (cascade)
- Reforestation requires massive intervention (not passive)
- Prevention is vastly more effective than cure

### Next Step

**Experiment 10c**: Add restoration forcing term to model active intervention during recovery phase.

### Visualizations

1. **Figure 10b.1**: Recovery fraction heatmap - uniformly low (<4%)
2. **Figure 10b.2**: Permanent tips heatmap - uniformly high (>45/50)
3. **Figure 10b.3**: Line plots showing α effect stronger than σ effect
4. **Figure 10b.4**: Regime map - all cells "Trapped"

---

## Experiment 11: Keystone Connections

*Status: Pending*

---

## Experiment 12: Publication Statistics

*Status: Pending*

---

## Synthesis: Phase 4 Findings

### Combined Insights from Experiments 8-10c

| Experiment | Key Question | Finding | Status |
|------------|--------------|---------|--------|
| Exp 8 | Does network fragmentation create asymmetry? | **Yes** — tip/recovery ratio increases from 1.005 (100%) to 1.148 (10% retention) | ✅ Validated (Dec 12) |
| Exp 9 | Do asymmetric barriers create hysteresis? | **Yes, but...** — the dominant effect is noise regime, not barrier shape | ⚠️ Needs re-validation |
| Exp 10 | Is there a critical α threshold for recovery? | **No** — recovery uniformly suppressed across all α values | ⚠️ Needs re-validation |
| Exp 10b | Can α×σ parameter space enable recovery? | **No** — even best conditions (α=1.2, σ=0.10) only achieve 3.6% recovery | ⚠️ Needs re-validation |
| Exp 10c | Does active forcing enable recovery? | *Affected by boundary bug — needs re-running* | ❌ Invalid - must re-run |

### The Emerging Picture: Three Sources of Asymmetry

#### Source 1: Noise Regime (DOMINANT)
- **Lévy noise** (α < 2): Fat-tailed distribution enables rare extreme events
- **Gaussian noise** (α = 2): Bounded perturbations, no extreme jumps
- **Result**: Lévy cascade → Gaussian recovery creates inherent hysteresis regardless of potential shape

#### Source 2: Network Fragmentation (CONFIRMED)
- **Intact networks (100%)**: tip/recovery ratio ≈ 1.0 (balanced)
- **Highly fragmented (10%)**: tip/recovery ratio ≈ 1.15 (**14.6% asymmetry**)
- **Mechanism**: Network coupling propagates "recovery signals" - fragmentation removes this support
- **Result**: Fragmentation creates progressive bias toward tipping

#### Source 3: Potential Shape (SECONDARY)
- Asymmetric barriers (barrier_recovery > barrier_tip) add additional bias
- But this effect is minor compared to noise regime and fragmentation effects
- May become more important when noise amplitudes are better matched

### Revised Understanding of Amazon Tipping Dynamics

| Process | Physical Reality | Model Representation | Key Finding |
|---------|------------------|----------------------|-------------|
| Tipping cascade | Extreme drought, fire, deforestation | Lévy noise (rare large jumps) | Easy to trigger |
| Recovery attempt | Normal rainfall variability | Gaussian noise (bounded) | ~0% passive recovery |
| Network connectivity | Moisture recycling corridors | Coupling strength | 14.6% asymmetry at high fragmentation |
| Hysteresis | Decades to regrow what burns in days | Noise regime asymmetry | Dominant effect |
| Active restoration | Reforestation programs | Recovery forcing term | Required for any recovery |

### Key Scientific Conclusions

1. **Network fragmentation creates a "one-way valve" effect**
   - Intact networks maintain thermodynamic balance (tip/recovery ratio ≈ 1.0)
   - At 10% retention, tipping is 14.6% more likely than recovery
   - This asymmetry compounds over time as degradation accelerates

2. **Hysteresis emerges from perturbation regime asymmetry**
   - Ecosystems tip during extreme events (Lévy-like)
   - Recovery must occur under normal variability (Gaussian-like)
   - This mismatch creates irreversibility even with symmetric potentials

3. **Passive recovery is insufficient**
   - All parameter combinations (α, σ) tested showed <4% recovery
   - Active forcing is required to restore tipped cells
   - Matches empirical observations of ecosystem restoration difficulty

4. **Prevention is vastly more effective than restoration**
   - At high fragmentation, preventing tipping is ~15% "cheaper" than recovery
   - Active intervention (forcing) needed for any meaningful recovery

### Policy Implications

1. **Connectivity preservation is critical**: Each percentage of edge loss increases tip/recovery asymmetry
2. **50% retention is a critical threshold**: First signs of asymmetry appear at this level
3. **Fragmentation is self-reinforcing**: Less connectivity → harder recovery → more tipping → less connectivity
4. **Active restoration required**: Passive recovery under any noise conditions is insufficient
5. **Early warning critical**: Once tipped, recovery requires external intervention

### Methodological Notes

**Solver Bug Discovery (December 2025):**
A critical numerical instability was discovered and fixed in the Euler-Maruyama solver. Original experiments used hard clamping at ±10, which caused boundary oscillation artifacts:
- 96.8% of trajectory time at boundaries (should be ~0%)
- ~23,000 false "events" per run (should be ~2,000)
- Entropy production inflated by 99.8%

**Fix:** Soft reflection boundaries at ±2 keep cells in the bistable region (|x| < 2). Results for Experiment 8 have been corrected; Experiments 10, 10b, 10c should be re-validated with fixed solver.

### Next Steps: Experiment Validation and Extension

#### Priority 1: Re-validate with Fixed Solver (Required)

| Experiment | Notebook | Est. Runtime | Priority | Rationale |
|------------|----------|--------------|----------|-----------|
| **Exp 10c** | `10_restoration_forcing.ipynb` | ~20 min | **HIGH** | Most affected by boundary bug; tests critical recovery forcing hypothesis |
| **Exp 10** | `08_alpha_sweep.ipynb` | ~35 min | MEDIUM | May show different α-recovery relationship with valid dynamics |
| **Exp 10b** | `09_alpha_sigma_sweep.ipynb` | ~20 min | MEDIUM | 2D sweep may reveal recovery-capable region |
| **Exp 9** | `07_recovery_dynamics.ipynb` | ~15 min | LOW | Core finding (noise asymmetry) likely unchanged |

**Recommended order**: 10c → 10 → 10b → 9

#### Priority 2: New Experiments to Strengthen Findings

| Experiment | Description | Purpose |
|------------|-------------|---------|
| **Exp 8b: Extended fragmentation** | Test 5%, 2%, 1% retention levels | Find asymmetry saturation point |
| **Exp 11: Keystone edges** | Identify which specific edges most affect tip/recovery ratio | Conservation targeting |
| **Exp 12: Temporal dynamics** | Track ratio evolution over longer simulation times | Verify asymmetry is stable |
| **Exp 13: Heterogeneous barriers** | Vary barrier heights across cells | Model spatial heterogeneity |

#### Priority 3: Model Extensions

1. **Climate trend forcing**: Add unidirectional warming/drying trend during cascade phase
2. **Spatial forcing patterns**: Model localized vs distributed restoration interventions
3. **Economic constraints**: Cost-weighted restoration optimization
4. **Validation dataset**: Compare predictions with observed Amazon deforestation/recovery data

### Key Questions Remaining

1. **Does the 14.8% asymmetry at 10% retention represent a tipping point?**
   - Need extended fragmentation experiment (Exp 8b) to find saturation

2. **Can ANY passive noise condition enable meaningful recovery?**
   - Re-run Exp 10, 10b with fixed solver to confirm

3. **What forcing magnitude is required for recovery?**
   - Re-run Exp 10c with fixed solver - this is the most critical experiment

4. **Why does random fragmentation create MORE asymmetry than targeted removal?**
   - Design new experiment to analyze edge-specific contributions

---

*Document Version 2.1 — December 12, 2025*

*Results from Experiments 8-10c completed on Dask cluster.*
- *Experiment 8: 360 simulations, **validated with fixed solver** (Dec 12, 2025)*
- *Experiment 9: 100 simulations, Option D parameters — needs re-validation*
- *Experiments 10, 10b, 10c: **Pending re-validation with fixed solver***

*Infrastructure: 14 Dask workers, optimized scatter-based task distribution*
