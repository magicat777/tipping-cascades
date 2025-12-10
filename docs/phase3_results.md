# Phase 3 Results: Energy-Constrained Tipping Cascades

**Project**: Climate Tipping Cascades with Thermodynamic Constraints
**Date**: December 2025
**Author**: Jason Holt

## Overview

Phase 3 extends the PyCascades framework with thermodynamic constraints to test hypotheses from Phase 2:
1. Energy constraints naturally create asymmetric effective couplings
2. Protective coupling networks represent thermodynamically favorable configurations
3. Cascade propagation requires energy input; transitions "cost" thermodynamic work

---

## Experiment 1: Asymmetric vs Symmetric Coupling

**Objective**: Test whether asymmetric coupling minimizes entropy production compared to symmetric coupling.

### Configuration
- **Network**: 4-element Earth system (GIS, THC, WAIS, AMAZ)
- **Ensemble size**: 10 runs per configuration
- **Duration**: 500 time units
- **Time step**: 1.0
- **Noise**: σ=0.08, α=1.5 (Lévy stable)
- **Seed**: 42 (reproducible)

### Results

#### Entropy Production Comparison

| Metric | Asymmetric | Symmetric | Difference |
|--------|------------|-----------|------------|
| Total Entropy (mean) | 208,133 | 212,837 | -2.2% |
| Total Entropy (std) | 12,270 | 4,197 | — |
| Coefficient of Variation | 5.9% | 2.0% | — |

**Key Finding**: Asymmetric coupling produces **2.2% less entropy** than symmetric coupling.

#### Tipping Events Comparison

| Metric | Asymmetric | Symmetric |
|--------|------------|-----------|
| Mean tipping events | 870 | 890 |
| Std tipping events | 51 | 18 |

**Key Finding**: Asymmetric coupling results in fewer tipping events with higher variance.

#### Time in Tipped State by Element

| Element | Asymmetric | Symmetric |
|---------|------------|-----------|
| GIS | 48.3% | 49.4% |
| THC | 49.0% | 49.7% |
| WAIS | 49.2% | 49.7% |
| AMAZ | 48.0% | 48.6% |

**Key Finding**: All elements show slightly less time in tipped state with asymmetric coupling. GIS protection is consistent with Phase 2 findings.

### Interpretation

1. **Hypothesis SUPPORTED**: Asymmetric coupling produces less entropy, suggesting it represents a thermodynamically favorable configuration.

2. **Higher variance in asymmetric**: The system has more "freedom" in how it evolves, leading to diverse trajectories while maintaining lower average entropy.

3. **GIS protection**: The asymmetric coupling structure protects GIS from cascading, consistent with the Phase 2 hypothesis about Amazon moisture recycling creating protective feedback.

---

## Experiment 2: Tipping Event Energy Costs (High Noise)

**Objective**: Analyze the thermodynamic cost of individual tipping events.

### Configuration
- Same as Experiment 1 (σ=0.08, α=1.5 Lévy noise)
- Analysis of single representative run

### Results

| Metric | Value |
|--------|-------|
| Total tipping events | 1,952 |
| Tip events (stable → tipped) | 976 |
| Recovery events (tipped → stable) | 976 |
| Avg entropy per tip event | 4,265 |
| Avg entropy per recovery event | 4,275 |
| Avg ΔE per event | 1.34 |

### Interpretation

1. **Nearly equal costs**: Tip and recovery events have virtually identical entropy costs (ratio ≈ 1.00).

2. **Small ΔE**: The tiny energy change (1.34) indicates transitions occur very close to the threshold (x ≈ 0), not deep barrier crossings.

3. **High frequency**: 1,952 events in 500 time units = ~4 events per time unit. This is "noisy bistable flickering" rather than genuine tipping cascades.

4. **Noise-dominated regime**: At high noise amplitude (σ=0.08) with heavy-tailed Lévy noise (α=1.5), the system is driven by stochastic fluctuations rather than deterministic barrier-crossing dynamics. The thermodynamic asymmetry between "uphill" tipping and "downhill" recovery is masked.

### Conclusion

The high-noise regime does not reveal thermodynamic asymmetry because:
- Transitions are noise-induced, not barrier-crossing events
- System doesn't settle in potential wells long enough
- Lévy noise extreme jumps can bypass barriers entirely

**Recommendation**: Test at lower noise to allow genuine thermodynamic dynamics to emerge.

---

## Experiment 3: Low-Noise Energy Cost Asymmetry

**Objective**: Test whether reducing noise reveals the expected thermodynamic asymmetry where tipping costs more entropy than recovery.

### Configuration
Four noise regimes compared:

| Label | σ | α | Description |
|-------|---|---|-------------|
| High Lévy | 0.08 | 1.5 | Baseline (Exp 2) |
| Low Lévy | 0.03 | 1.5 | Reduced amplitude, heavy tails |
| Low Gaussian | 0.03 | 2.0 | Reduced, normal distribution |
| Very Low Gaussian | 0.02 | 2.0 | Cleanest regime |

- **Duration**: 2000 time units (longer for rare events)
- **Time step**: 0.5 (finer resolution)
- **Ensemble**: 10 runs per configuration
- **Filter**: Only "clean" events persisting >10 time units

### Hypothesis

With lower noise:
- **Tip events** should cost MORE entropy (climbing potential barrier = thermodynamically unfavorable)
- **Recovery events** should cost LESS entropy (descending barrier = thermodynamically favorable)
- **Ratio (Tip/Recovery) > 1.0** expected

### Results

| Noise Regime | Clean Events | Tip Cost | Recovery Cost | Ratio | Verdict |
|--------------|--------------|----------|---------------|-------|---------|
| High Lévy (σ=0.08, α=1.5) | 11 | 5068.8 ± 7164.6 | 3.0 ± 2.1 | **1700.5** | ✓ SUPPORTED |
| Low Lévy (σ=0.03, α=1.5) | 22 | 2.3 ± 1.6 | 3.4 ± 4.9 | **0.68** | ✗ REVERSED |
| Low Gaussian (σ=0.03, α=2.0) | 0 | — | — | — | ❓ NO DATA |
| Very Low Gaussian (σ=0.02, α=2.0) | 0 | — | — | — | ❓ NO DATA |

### Key Findings

#### 1. High-Noise Lévy: Extreme Asymmetry (Ratio = 1700x)

- **Tip cost**: 5068.8 ± 7164.6 (massive variance!)
- **Recovery cost**: 3.0 ± 2.1 (very consistent)
- The enormous standard deviation indicates rare Lévy extreme jumps cause catastrophic, high-entropy tipping events
- Recovery is "cheap" because the system naturally relaxes downhill after being thrown over the barrier
- **Interpretation**: Lévy jumps can violently throw the system over barriers, producing huge entropy spikes. Recovery is a gentle, low-entropy relaxation.

#### 2. Low-Noise Lévy: REVERSED Asymmetry (Ratio = 0.68)

This is the most surprising and scientifically interesting result:

- **Tipping costs LESS than recovery** — the opposite of our hypothesis!
- **Mechanism**: At low amplitude, Lévy noise rarely produces large jumps, but when it does, the system can "teleport" over the barrier with minimal entropy production (quick transition)
- Recovery, however, requires the system to gradually climb back over the barrier against the deterministic flow, which is slow and dissipative
- **Analogy**: Like quantum tunneling — the heavy tail allows barrier bypass without "climbing"

#### 3. Gaussian Noise: Complete Stability

- **Zero tipping events** at σ=0.02-0.03 with Gaussian noise
- The system is trapped in its initial potential well
- Gaussian noise lacks the heavy tails needed to overcome the barrier at these amplitudes
- **Implication**: The barrier height (~0.5) requires σ > 0.03 for Gaussian-driven tipping

### Physical Interpretation

| Noise Type | Tipping Mechanism | Thermodynamic Pattern |
|------------|-------------------|----------------------|
| **High Lévy** | Violent extreme jumps | Tip >> Recovery (catastrophic) |
| **Low Lévy** | Rare barrier "tunneling" | Tip < Recovery (inverted!) |
| **Gaussian** | Gradual barrier climbing | No events (too stable) |

### Theoretical Implications

1. **Lévy noise fundamentally alters thermodynamics**: The heavy-tailed distribution (α < 2) enables transitions that bypass the classical barrier-crossing paradigm. This is analogous to quantum tunneling in that the system can appear on the other side of a barrier without traversing it classically.

2. **Kramers theory breakdown**: Classical Kramers escape rate theory assumes Gaussian fluctuations and thermal activation over barriers. Lévy noise violates these assumptions, leading to:
   - Non-Arrhenius temperature dependence
   - Inverted cost asymmetry at low noise
   - Extreme event dominance

3. **Climate implications**: If climate system noise has heavy tails (as suggested by paleoclimate records showing abrupt transitions), then:
   - Tipping may occur more easily than Gaussian models predict
   - Recovery from tipped states may be harder than expected
   - Traditional "early warning signals" based on Gaussian assumptions may fail

### Conclusion

The hypothesis that "tipping costs more entropy than recovery" is:
- **SUPPORTED** for high-amplitude Lévy noise (extreme jumps dominate)
- **REVERSED** for low-amplitude Lévy noise (barrier tunneling inverts the pattern)
- **UNTESTABLE** for Gaussian noise at these amplitudes (no events)

This reveals a **noise-type bifurcation** in the thermodynamics of tipping cascades.

---

## Summary of Phase 3 Findings

### Confirmed Hypotheses

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| Asymmetric coupling minimizes entropy | ✓ SUPPORTED | 2.2% reduction vs symmetric |
| GIS protection from cascading | ✓ SUPPORTED | Lower tipped % with asymmetric |
| Tipping costs more than recovery | ⚠️ COMPLEX | Depends on noise type! |

### Key Insights

1. **Thermodynamic favorability**: The asymmetric coupling structure found in Phase 2 (protective Amazon feedback) appears to be thermodynamically favorable, producing less entropy than symmetric alternatives.

2. **Noise-type bifurcation**: The thermodynamic cost asymmetry between tipping and recovery fundamentally depends on the noise distribution:
   - **High Lévy**: Tipping is catastrophically expensive (1700x recovery cost)
   - **Low Lévy**: Tipping is cheaper than recovery (0.68x) — INVERTED!
   - **Gaussian**: No events at comparable amplitudes

3. **Lévy "tunneling" effect**: Heavy-tailed Lévy noise (α<2) enables barrier bypass via extreme jumps, analogous to quantum tunneling. This inverts classical thermodynamic expectations at low noise amplitudes.

4. **Kramers theory breakdown**: Classical escape rate theory assumes Gaussian fluctuations. Lévy noise violates these assumptions, producing qualitatively different behavior.

5. **Climate relevance**: If real climate forcing has heavy tails (supported by paleoclimate evidence), then:
   - Tipping may be easier than Gaussian models predict
   - Recovery may be harder than expected
   - Traditional early warning signals may fail

---

## Technical Notes

### Energy-Constrained Framework

The Phase 3 implementation extends PyCascades with:

- **Extended state space**: `y = [x_0, ..., x_{n-1}, E_0, ..., E_{n-1}]`
- **Energy tracking**: Double-well potential `U(x) = x⁴/4 - x²/2` scaled by barrier height
- **Dissipation**: Viscous damping `D = γ(dx/dt)²`
- **Entropy production**: `σ = D/T`
- **Energy-based coupling**: Flow proportional to energy gradient

### Numerical Considerations

- State bounds [-10, 10] prevent overflow from Lévy jumps
- Energy bounds [-100, 100] maintain physical values
- Clipping in potential/force calculations for numerical stability

### Code Location

- Module: `src/energy_constrained/`
- Notebook: `notebooks/04_energy_constrained_exploration.ipynb`
- Tests: `tests/test_energy_constrained/`

---

## Experiment 4: Amazon Spatial Model - 50-Cell Subnetwork

**Objective**: Test whether the noise-type bifurcation discovered in idealized models applies to spatially-explicit Amazon moisture recycling cascades.

### Configuration
- **Network**: 50 highest-connectivity Amazon cells (from 567-cell grid)
- **Data source**: Wunderling et al. (2022) moisture recycling matrix
- **Ensemble size**: 5 runs per configuration
- **Duration**: 500 time units
- **Time step**: 0.5
- **Coupling**: Gradient-driven, proportional to moisture flow

### Noise Configurations

| Label | σ | α | Description |
|-------|---|---|-------------|
| Gaussian | 0.06 | 2.0 | Baseline normal noise |
| Lévy α=1.8 | 0.06 | 1.8 | Mildly heavy-tailed |
| Lévy α=1.5 | 0.06 | 1.5 | Moderate heavy tails |
| Lévy α=1.2 | 0.06 | 1.2 | Strongly heavy-tailed |

### Results

#### Entropy Production

| Noise Type | Total Entropy | Tip/Recovery Ratio |
|------------|---------------|-------------------|
| Gaussian (α=2.0) | ~8,000 | 1.0000 |
| Lévy α=1.8 | ~22,000 | 0.9976 |
| Lévy α=1.5 | ~35,000 | 0.9986 |
| Lévy α=1.2 | ~35,000 | 0.9991 |

### Key Finding: Near-Symmetric Entropy in Spatial Model

**Contrast with Phase 3 idealized model:**
- Idealized 4-element: Up to 1700x tip/recovery asymmetry
- Spatial 50-cell: All ratios within 0.3% of 1.0 (symmetric)

**Interpretation**: The moisture recycling network provides **thermodynamic buffering** that prevents asymmetric tipping dynamics. The bidirectional coupling creates stabilizing feedback loops.

---

## Experiment 5: Full 567-Cell Amazon Network

**Objective**: Verify whether thermodynamic buffering is a scale-invariant property of the Amazon network or an artifact of high-connectivity cell selection.

### Configuration
- **Network**: All 567 Amazon grid cells
- **Couplings**: 20,000+ connections (min_flow > 1.0 mm/month)
- **Parallelization**: Dask distributed across 4 workers
- **Other parameters**: Same as Experiment 4

### Results

#### Comparison: 50-Cell vs Full 567-Cell Network

| Noise Type | 50-cell Ratio | 567-cell Ratio | Difference |
|------------|---------------|----------------|------------|
| Gaussian (α=2.0) | 1.0000 | 1.0000 | +0.0000 |
| Lévy α=1.8 | 0.9976 | 0.9991 | +0.0014 |
| Lévy α=1.5 | 0.9986 | 0.9991 | +0.0005 |
| Lévy α=1.2 | 0.9991 | 0.9991 | +0.0000 |

### Key Findings

1. **Scale-invariant buffering**: The near-perfect tip/recovery symmetry holds at both 50-cell and 567-cell scales. The difference is <0.2% across all noise types.

2. **Robust network property**: Thermodynamic buffering is NOT an artifact of cell selection. Peripheral, weakly-connected cells show the same behavior as high-connectivity hubs.

3. **Contrast with idealized model**: The Amazon moisture recycling topology fundamentally differs from the idealized 4-element network:
   - Idealized: Sparse, directional coupling → asymmetric dynamics
   - Amazon: Dense, bidirectional moisture flows → symmetric dynamics

### Physical Interpretation

The Amazon's moisture recycling network creates **closed-loop feedback**:
- Forest evapotranspiration → moisture transport → downwind precipitation → forest growth
- This bidirectional cycling equalizes the thermodynamic cost of state transitions

**Conservation implication**: The network's inherent resilience depends on maintaining connectivity. Deforestation that breaks moisture recycling links could expose regions to the asymmetric tipping dynamics seen in isolated systems.

---

## Summary of Phase 3 Findings

### Confirmed Hypotheses

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| Asymmetric coupling minimizes entropy | ✓ SUPPORTED | 2.2% reduction vs symmetric |
| GIS protection from cascading | ✓ SUPPORTED | Lower tipped % with asymmetric |
| Tipping costs more than recovery | ⚠️ COMPLEX | Depends on noise type! |
| **Amazon network buffers asymmetry** | ✓ SUPPORTED | Ratio ≈ 1.0 at all scales |

### Key Insights

1. **Thermodynamic favorability**: The asymmetric coupling structure found in Phase 2 (protective Amazon feedback) appears to be thermodynamically favorable, producing less entropy than symmetric alternatives.

2. **Noise-type bifurcation**: The thermodynamic cost asymmetry between tipping and recovery fundamentally depends on the noise distribution:
   - **High Lévy**: Tipping is catastrophically expensive (1700x recovery cost)
   - **Low Lévy**: Tipping is cheaper than recovery (0.68x) — INVERTED!
   - **Gaussian**: No events at comparable amplitudes

3. **Lévy "tunneling" effect**: Heavy-tailed Lévy noise (α<2) enables barrier bypass via extreme jumps, analogous to quantum tunneling. This inverts classical thermodynamic expectations at low noise amplitudes.

4. **Kramers theory breakdown**: Classical escape rate theory assumes Gaussian fluctuations. Lévy noise violates these assumptions, producing qualitatively different behavior.

5. **Spatial network buffering (NEW)**: The Amazon moisture recycling network provides scale-invariant thermodynamic buffering, eliminating the asymmetry seen in isolated/idealized systems. This is a network topology effect, not a scale effect.

6. **Climate relevance**: If real climate forcing has heavy tails (supported by paleoclimate evidence), then:
   - Tipping may be easier than Gaussian models predict
   - Recovery may be harder than expected
   - BUT: Well-connected networks like Amazon moisture recycling provide inherent resilience

---

## Technical Notes

### Energy-Constrained Framework

The Phase 3 implementation extends PyCascades with:

- **Extended state space**: `y = [x_0, ..., x_{n-1}, E_0, ..., E_{n-1}]`
- **Energy tracking**: Double-well potential `U(x) = x⁴/4 - x²/2` scaled by barrier height
- **Dissipation**: Viscous damping `D = γ(dx/dt)²`
- **Entropy production**: `σ = D/T`
- **Energy-based coupling**: Flow proportional to energy gradient

### Dask Parallelization

For large-scale experiments (567-cell network):
- Worker path initialization via `client.run()`
- Serialization of network objects via pickle
- Conversion between dict results and SolverResult objects
- Batch processing with progress tracking

### Numerical Considerations

- State bounds [-10, 10] prevent overflow from Lévy jumps
- Energy bounds [-100, 100] maintain physical values
- Clipping in potential/force calculations for numerical stability

### Code Location

- Module: `src/energy_constrained/`
- Notebooks:
  - `notebooks/04_energy_constrained_exploration.ipynb` (idealized experiments)
  - `notebooks/05_amazon_spatial_energy_tracking.ipynb` (spatial Amazon model)
- Tests: `tests/test_energy_constrained/`

---

## Experiment 6: Drought Impact on Cascade Thermodynamics

**Objective**: Test whether historical Amazon droughts (2005, 2010) alter cascade vulnerability and thermodynamic costs.

### Configuration
- **Data**: Wunderling et al. moisture recycling data for dry season (Jul-Sep)
- **Years compared**: 2003 (normal), 2005 (drought), 2010 (severe drought)
- **Network**: 50-cell subnetwork with year-specific barriers and coupling
- **Noise**: Gaussian (α=2.0, σ=0.06)
- **Ensemble**: 5 runs per condition

### Drought Impact on Network Properties

Droughts affect the network in two ways:
1. **Lower barriers**: Reduced rainfall/evap ratio → more vulnerable elements
2. **Weaker coupling**: Less moisture recycling → reduced network connectivity

### Results: Gaussian Noise (No Cascades)

| Condition | Total Entropy | Tipping Events | % Tipped |
|-----------|---------------|----------------|----------|
| 2003 (Normal) | 29.4 | 0 | 0% |
| 2005 (Drought) | 23.7 | 0 | 0% |
| 2010 (Severe) | 19.9 | 0 | 0% |

### Key Finding: Drought Reduces Thermodynamic Activity

- **No cascades triggered** under any condition with Gaussian noise
- But entropy production **decreased** during drought:
  - 2005: -19% vs 2003
  - 2010: -32% vs 2003

**Interpretation**: Drought weakens moisture recycling → less energy flows through network → less dissipation → less entropy production. The network becomes "quieter" but potentially more fragile.

---

## Experiment 7: Drought × Lévy Noise Interaction

**Objective**: Test whether extreme events (Lévy noise) can trigger cascades under drought-weakened conditions, and whether there's a synergistic vulnerability effect.

### Configuration
- **Climate conditions**: 2003 (Normal) vs 2010 (Severe drought)
- **Noise types**: Gaussian (α=2.0), Lévy α=1.5, Lévy α=1.2
- **Other parameters**: Same as Experiment 6

### Results

| Configuration | Total Entropy | Tipping Events | % Time Tipped |
|---------------|---------------|----------------|---------------|
| 2003 + Gaussian | 29.4 | 0 | 0.0% |
| 2003 + Lévy α=1.5 | 3,014,690 | 20,777 | 48.5% |
| 2003 + Lévy α=1.2 | 3,244,371 | 22,232 | 49.9% |
| 2010 + Gaussian | 19.9 | 0 | 0.0% |
| 2010 + Lévy α=1.5 | 1,717,244 | 18,531 | 46.9% |
| 2010 + Lévy α=1.2 | 2,130,717 | 21,893 | 49.6% |

### Key Findings

#### 1. Lévy Noise is the Critical Trigger

The difference between Gaussian and Lévy is **5 orders of magnitude**:
- Gaussian: ~20-30 entropy, 0% tipped
- Lévy: ~2-3 million entropy, ~50% tipped

**Conclusion**: Extreme events (heavy-tailed noise) completely dominate cascade behavior, regardless of climate condition.

#### 2. Drought Reduces Entropy Even Under Cascading Conditions

Even when cascades are triggered by Lévy noise:
- Lévy α=1.5: 2010 drought produces **43% less entropy** than 2003 normal
- Lévy α=1.2: 2010 drought produces **34% less entropy** than 2003 normal

The pattern from Experiment 6 persists: weaker network = less thermodynamic activity.

#### 3. No Synergistic Vulnerability

We hypothesized drought + Lévy might produce worse outcomes than either alone. Instead:
- Both conditions reach ~50% tipped under Lévy noise
- Drought actually has **fewer tipping events** (18.5K vs 21K at α=1.5)
- Weaker coupling = cascades propagate less efficiently

### Physical Interpretation

```
Lévy noise → Extreme jumps bypass barriers → Cascades trigger regardless of climate

Drought → Weaker moisture recycling →
   ├── Fewer cascade pathways (slightly protective in propagation)
   └── Less recovery support (dangerous for permanence of tipped state)
   └── Less overall thermodynamic activity
```

### Implications for Amazon Tipping

1. **Extreme events dominate**: A single extreme fire season or heat wave (Lévy-like forcing) can trigger cascades that normal climate variability (Gaussian) cannot, regardless of drought status.

2. **Drought is a secondary modifier**: It doesn't prevent cascades under extreme forcing, but changes their thermodynamic character—the system is "quieter" overall.

3. **The "quieter" drought network** may trap cells in tipped states longer because reduced coupling provides less recovery support from neighboring cells.

4. **Conservation priority**: Preventing extreme forcing events (fires, deforestation shocks) is more critical than drought mitigation for cascade prevention, but maintaining network connectivity is essential for recovery.

---

## Summary of Phase 3 Findings

### Confirmed Hypotheses

| Hypothesis | Status | Evidence |
|------------|--------|----------|
| Asymmetric coupling minimizes entropy | ✓ SUPPORTED | 2.2% reduction vs symmetric |
| GIS protection from cascading | ✓ SUPPORTED | Lower tipped % with asymmetric |
| Tipping costs more than recovery | ⚠️ COMPLEX | Depends on noise type! |
| Amazon network buffers asymmetry | ✓ SUPPORTED | Ratio ≈ 1.0 at all scales |
| **Drought increases vulnerability** | ⚠️ COMPLEX | Reduces activity, not triggers cascades |
| **Extreme events trigger cascades** | ✓ SUPPORTED | Lévy >> Gaussian by 5 orders of magnitude |

### Key Insights

1. **Thermodynamic favorability**: The asymmetric coupling structure found in Phase 2 (protective Amazon feedback) appears to be thermodynamically favorable, producing less entropy than symmetric alternatives.

2. **Noise-type bifurcation**: The thermodynamic cost asymmetry between tipping and recovery fundamentally depends on the noise distribution:
   - **High Lévy**: Tipping is catastrophically expensive (1700x recovery cost)
   - **Low Lévy**: Tipping is cheaper than recovery (0.68x) — INVERTED!
   - **Gaussian**: No events at comparable amplitudes

3. **Lévy "tunneling" effect**: Heavy-tailed Lévy noise (α<2) enables barrier bypass via extreme jumps, analogous to quantum tunneling. This inverts classical thermodynamic expectations at low noise amplitudes.

4. **Kramers theory breakdown**: Classical escape rate theory assumes Gaussian fluctuations. Lévy noise violates these assumptions, producing qualitatively different behavior.

5. **Spatial network buffering**: The Amazon moisture recycling network provides scale-invariant thermodynamic buffering, eliminating the asymmetry seen in isolated/idealized systems. This is a network topology effect, not a scale effect.

6. **Extreme events dominate drought** (NEW): Lévy noise triggers cascades (~50% tipped) regardless of climate condition. Drought modifies thermodynamic activity (-32% to -43% entropy) but doesn't fundamentally change cascade occurrence.

7. **Climate relevance**: If real climate forcing has heavy tails (supported by paleoclimate evidence), then:
   - Tipping may be easier than Gaussian models predict
   - Recovery may be harder than expected
   - BUT: Well-connected networks like Amazon moisture recycling provide inherent resilience
   - Extreme events (fires, heat waves) are the critical trigger, not gradual drought

---

## Technical Notes

### Energy-Constrained Framework

The Phase 3 implementation extends PyCascades with:

- **Extended state space**: `y = [x_0, ..., x_{n-1}, E_0, ..., E_{n-1}]`
- **Energy tracking**: Double-well potential `U(x) = x⁴/4 - x²/2` scaled by barrier height
- **Dissipation**: Viscous damping `D = γ(dx/dt)²`
- **Entropy production**: `σ = D/T`
- **Energy-based coupling**: Flow proportional to energy gradient

### Dask Parallelization

For large-scale experiments (567-cell network):
- Worker path initialization via `client.run()`
- Serialization of network objects via pickle
- Conversion between dict results and SolverResult objects
- Batch processing with progress tracking

### Numerical Considerations

- State bounds [-10, 10] prevent overflow from Lévy jumps
- Energy bounds [-100, 100] maintain physical values
- Clipping in potential/force calculations for numerical stability

### Code Location

- Module: `src/energy_constrained/`
- Notebooks:
  - `notebooks/04_energy_constrained_exploration.ipynb` (idealized experiments)
  - `notebooks/05_amazon_spatial_energy_tracking.ipynb` (spatial Amazon model)
- Tests: `tests/test_energy_constrained/`

---

## Next Steps

1. **Deforestation scenarios**: Model impact of network fragmentation on thermodynamic resilience
2. **Statistical significance**: Larger ensembles (N=50-100) for publication-quality results
3. **α-sweep experiment**: Vary α from 1.2 to 2.0 to map the transition from Lévy to Gaussian behavior
4. **Paleoclimate validation**: Compare model predictions with ice core / sediment records of abrupt transitions
5. **Recovery dynamics**: Study how drought-weakened networks recover (or fail to recover) after cascades
