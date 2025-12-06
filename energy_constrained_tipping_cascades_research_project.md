# Energy-Constrained Tipping Cascades in Coupled Socio-Ecological Systems

**Personal Science Research Project**

*Jason Holt — December 2025*

---

> **Executive Summary**
>
> This project extends the PyCascades tipping cascade modeling framework to incorporate energy flow constraints as a novel parameterization of tipping element interactions. Drawing from thermodynamic principles that treat human systems as dissipative structures requiring constant energy throughput, this research models how cascade events disrupt energy availability across interconnected climate, biosphere, and human settlement systems, triggering secondary failures through energy deprivation rather than direct forcing alone.
>
> The approach addresses a recognized gap in coupled human-environment system (CHES) modeling: the absence of explicit energy accounting in tipping cascade dynamics. By parameterizing interaction strengths as functions of energy flow capacity, this framework enables investigation of infrastructure-mediated cascade propagation mechanisms not captured by existing models.

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [Research Objectives](#2-research-objectives)
3. [System Configuration](#3-system-configuration)
   - 3.1 [Hardware Platform](#31-hardware-platform)
   - 3.2 [Software Environment Setup](#32-software-environment-setup)
   - 3.3 [Verification](#33-verification)
   - 3.4 [k3s Research Infrastructure](#34-k3s-research-infrastructure)
   - 3.5 [Operational Resilience](#35-operational-resilience)
   - 3.6 [AI-Assisted Research Workflow](#36-ai-assisted-research-workflow)
4. [Research Phases](#4-research-phases)
5. [Project Timeline](#5-project-timeline)
6. [Key References](#6-key-references)
7. [Project Resources](#7-project-resources)

---

## 1. Background and Motivation

### 1.1 The Tipping Cascade Problem

Earth system tipping elements—critical thresholds in climate subsystems where small perturbations can trigger qualitative state changes—do not operate in isolation. The Greenland Ice Sheet, Amazon rainforest, Atlantic Meridional Overturning Circulation (AMOC), West Antarctic Ice Sheet, and other major tipping elements interact through teleconnections, moisture transport, ocean circulation, and atmospheric dynamics. Recent research demonstrates that these interactions can produce domino effects: one element crossing its threshold can push others toward or beyond theirs.

The Global Tipping Points Report 2025 identifies approximately 24 Earth system components that may reach tipping points, with tropical coral reefs already having crossed their threshold at approximately 1.2°C warming. Modeling studies using Monte Carlo approaches with millions of simulations have found that roughly one-third of realizations produce cascading effects between tipping elements. This cascade risk is particularly sensitive to polar ice sheet dynamics—at 1.5°C warming, neglecting ice sheet interactions alters expected tipped elements by more than a factor of two.

### 1.2 The Missing Human Dimension

Current tipping cascade models primarily address climate-climate and climate-biosphere interactions. However, human systems—settlements, infrastructure, economies, governance structures—also exhibit tipping behavior and couple bidirectionally with both climate and ecological systems. Mathematical models that couple human behavior to environmental processes reveal that changes in human system parameters can abruptly lead to desirable or undesirable new states. Yet there are far fewer coupled human-environment system (CHES) models than models of uncoupled ecosystems.

The World Bank projects 216 million people will move within their countries by 2050 due to climate impacts. Cities receiving the most climate migration—Bogota, Sao Paulo, Dhaka, Karachi, Accra, Freetown—are already under infrastructure stress. This displacement creates cascade dynamics: populations carry energy demands to new locations while abandoning embodied energy in infrastructure; receiving regions must expand capacity or experience degradation. Above approximately 2.3°C warming, climate damages may increase adaptation investment needs enough to trigger private-debt tipping points and cascading financial defaults.

### 1.3 Human Systems as Dissipative Structures

A thermodynamic perspective provides conceptual clarity: human settlements and infrastructure networks are dissipative structures in the Prigogine sense—they maintain organized states far from thermodynamic equilibrium only through continuous energy throughput. When that energy flow is disrupted, the system cannot maintain its organized state and degrades toward higher entropy configurations. This is not merely metaphorical; cities literally require constant energy input for water treatment, climate control, food storage, transportation, communication, and the maintenance activities that prevent infrastructure decay.

This framing suggests a novel approach to modeling tipping cascades: parameterize the interaction strengths between tipping elements as functions of energy flow capacity. When one element tips, it may not directly force another element toward its threshold; instead, it may disrupt the energy flows that the second element requires to maintain its stable state. A cascade event in the power grid does not apply mechanical force to water treatment plants—it removes the energy throughput they require for operation.

### 1.4 Research Gap and Contribution

The PyCascades software package from the Potsdam Institute for Climate Impact Research (PIK) provides a robust framework for modeling interacting tipping elements on complex networks. It supports various bifurcation types (cusp, Hopf), network topologies (Erdös-Rényi, Barabási-Albert, Watts-Strogatz), and stochastic processes (Gaussian noise, Lévy, Cauchy). Published applications include Amazon moisture recycling networks, interacting Earth system tipping elements, and global trade network cascades.

However, the current framework does not explicitly model energy flow constraints on interaction dynamics. Coupling strengths are typically parameterized as static values or simple functions of system state. This project extends PyCascades to incorporate energy throughput as a fundamental variable governing both system stability and inter-system coupling. The contribution addresses a recognized need for integrated socio-ecological modeling that captures the mechanistic pathways through which cascade events propagate across system boundaries.

---

## 2. Research Objectives

### 2.1 Primary Objective

Develop and validate an extension to the PyCascades framework that parameterizes tipping element interactions with energy flow constraints, enabling investigation of cascade propagation mechanisms mediated by energy availability disruption.

### 2.2 Specific Objectives

1. Define a formal energy accounting framework for tipping elements, specifying energy throughput requirements for state maintenance and energy flow dependencies in coupling terms.

2. Implement new tipping element classes that incorporate energy state variables and energy-dependent stability thresholds.

3. Develop energy-mediated coupling functions that modulate interaction strength based on available energy capacity rather than static parameters.

4. Construct a multi-layer network model connecting climate tipping elements, biosphere regime shifts, and human settlement stability through energy flow pathways.

5. Conduct parameter sensitivity analysis to identify energy flow parameters with greatest influence on cascade risk.

6. Compare cascade dynamics between energy-constrained and standard (non-energy-constrained) model configurations to quantify the contribution of energy flow mechanisms.

7. Document findings and contribute code extensions to the open-source PyCascades repository.

### 2.3 Success Criteria

- Working implementation of energy-constrained tipping elements that integrates with existing PyCascades infrastructure.
- Reproducible ensemble simulations demonstrating distinct cascade behavior under energy constraints versus standard parameterization.
- Sensitivity analysis identifying key energy parameters influencing cascade risk metrics.
- Documentation sufficient for other researchers to use and extend the framework.
- Engagement with PIK COPAN research group for feedback and potential collaboration.

---

## 3. System Configuration

### 3.1 Hardware Platform

| Component | Specification |
|-----------|---------------|
| System | 2023 Razer Blade 18 |
| RAM | 64GB DDR5 |
| GPU VRAM | 12GB (NVIDIA RTX) |
| Operating System | Ubuntu Linux (dual-boot) |
| Container Platform | k3s Kubernetes |

**Hardware Capability Assessment:** This configuration supports ensemble sizes of 500K–2M simulations overnight, network sizes exceeding 100 nodes with full coupling, and 10,000+ stochastic realizations per parameter configuration. These scales are comparable to published research and sufficient for novel scientific contributions.

### 3.2 Software Environment Setup

**Step 1: Create Isolated Conda Environment**

```bash
conda create -n tipping-research python=3.9
conda deactivate
conda activate tipping-research
```

**Step 2: Install Dependencies via Mamba**

```bash
conda install -c conda-forge mamba
mamba install -c conda-forge numpy scipy matplotlib cartopy seaborn \
    netCDF4 networkx ipykernel numba
```

**Step 3: Install PyCascades-Specific Dependencies**

```bash
pip install sdeint PyPDF2 pyDOE SALib
```

**Step 4: Clone Research Repositories**

```bash
mkdir ~/climate-research && cd ~/climate-research
git clone https://github.com/pik-copan/pycascades.git
git clone https://github.com/pik-copan/pyunicorn.git
git clone https://github.com/pik-copan/pycopancore.git
```

**Step 5: Install Packages in Development Mode**

```bash
cd pycascades && pip install -e .
cd ../pyunicorn && pip install -e .
cd ../pycopancore && pip install -e .
```

### 3.3 Verification

```bash
python -c "import pycascades as pc; print('PyCascades:', pc.__file__)"
python -c "import pyunicorn; print('pyunicorn:', pyunicorn.__file__)"
python -c "from SALib.sample import saltelli; print('SALib ready')"
```

### 3.4 k3s Research Infrastructure

This project leverages the existing ODIN k3s cluster for scalable, GPU-accelerated simulation workloads. The infrastructure follows a hybrid deployment model:

- **Research artifacts** (code, notebooks, data, papers): `Projects/cascades/`
- **k3s deployment manifests**: `ODIN/k3s-configs/namespaces/cascades/`

#### 3.4.1 Hardware Resource Allocation

| Resource | Total Available | System Reserved | Available for k3s |
|----------|-----------------|-----------------|-------------------|
| CPU Cores | ~16 physical (32 threads) | 4 threads | ~28 threads |
| RAM | 64 GB | 8 GB | ~56 GB |
| GPU VRAM | 12 GB | 2 GB (display) | ~10 GB |
| GPU Compute | 1× RTX 4080 | Shared | Time-sliced |

#### 3.4.2 Pod Architecture

**Tier 1: Core Infrastructure (Always Running)**

| Pod | Purpose | CPU | RAM | GPU |
|-----|---------|-----|-----|-----|
| `jupyterlab` | Interactive development, notebooks | 2 cores | 8 GB | Optional |
| `mlflow-server` | Experiment tracking, metrics | 0.5 cores | 1 GB | None |
| `dask-scheduler` | Coordinates distributed workers | 0.5 cores | 1 GB | None |
| `redis` | Task state persistence, job queue | 0.5 cores | 1 GB | None |
| **Subtotal** | | **3.5 cores** | **11 GB** | **0** |

**Tier 2: Scalable Compute Workers (Auto-scaling)**

| Pod Type | Min | Max | Per-Pod Resources | Purpose |
|----------|-----|-----|-------------------|---------|
| `dask-worker-cpu` | 0 | 12 | 2 cores, 4 GB RAM | CPU-bound simulations |
| `dask-worker-gpu` | 0 | 2 | 2 cores, 4 GB RAM, 4 GB VRAM | GPU-accelerated SDE integration |
| **Peak Subtotal** | | **14** | **28 cores, 56 GB** | |

**Tier 3: Optional Services (Phase-dependent)**

| Pod | Purpose | CPU | RAM | GPU | When Needed |
|-----|---------|-----|-----|-----|-------------|
| `ollama-llm` | Local LLM for literature processing | 2 cores | 8 GB | 6 GB VRAM | Phase 1-2 literature review |
| `neo4j` | Graph database for network analysis | 1 core | 2 GB | None | Phase 4 pathway analysis |

#### 3.4.3 Pod Count Summary

| Scenario | Pod Count | Resource Usage |
|----------|-----------|----------------|
| **Idle/Development** | 4-5 pods | JupyterLab + MLflow + Scheduler + Redis |
| **Light Analysis** | 7-9 pods | + 3-4 CPU workers |
| **Heavy Ensemble (CPU)** | 16-17 pods | + 12 CPU workers |
| **Heavy Ensemble (GPU)** | 6-7 pods | + 2 GPU workers (VRAM limited) |
| **Maximum (mixed)** | ~19 pods | Full utilization |

#### 3.4.4 Namespace Structure

```
cascades namespace
├── Core Services (always on): 4 pods
│   ├── jupyterlab (1)
│   ├── mlflow-server (1)
│   ├── dask-scheduler (1)
│   └── redis (1)
│
├── Workers (auto-scaling): 0-14 pods
│   ├── dask-worker-cpu (0-12)
│   │   └── HorizontalPodAutoscaler based on queue depth
│   └── dask-worker-gpu (0-2)
│       └── Manual scaling (GPU contention management)
│
├── Optional (phase-dependent): 0-2 pods
│   ├── ollama-llm (0-1)
│   └── neo4j (0-1)
│
└── Persistent Storage
    ├── cascades-code-pvc (5 GB) - notebooks, scripts
    ├── cascades-data-pvc (100 GB) - simulation outputs
    ├── cascades-checkpoints-pvc (10 GB) - job checkpoints
    └── mlflow-artifacts-pvc (20 GB) - experiment artifacts
```

### 3.5 Operational Resilience

Large-scale ensemble simulations (500K-2M runs) require fault-tolerant architecture to prevent catastrophic progress loss from system interruptions.

#### 3.5.1 Risk Assessment (Laptop Environment)

| Risk | Likelihood | Impact Without Mitigation |
|------|------------|---------------------------|
| Thermal throttling | High | Slows execution |
| Thermal shutdown | Medium | **100% progress lost** |
| Power disconnection | Medium | Battery drain → shutdown |
| NVIDIA driver crash | Medium | GPU workers lost |
| Accidental lid close | Low-Medium | Sleep = frozen state |
| Ubuntu kernel update | Low | Potential auto-reboot |
| OOM killer | Low (64GB) | Worker termination |

#### 3.5.2 Resilience Strategies

**Strategy 1: Chunked Job Architecture**

Instead of monolithic simulation runs, work is decomposed into small, resumable units:

```
Monolithic (fragile):
└── 1 job: 500,000 simulations → 12 hours
    └── Crash at hour 11 = lose 11 hours

Chunked (resilient):
└── 500 jobs: 1,000 simulations each → ~1.5 min per job
    ├── Job completes → results written to PVC immediately
    ├── System crashes → lose at most 1 job (1.5 min of work)
    └── System restarts → remaining jobs resume from queue
```

**Strategy 2: Incremental Result Persistence**

Results are written as they complete, not batched at the end:

```python
# Resilient pattern
for batch in simulation_batches:
    result = run_batch(batch)
    append_to_netcdf(result)      # Persisted immediately
    mark_batch_complete(batch.id)  # Progress tracked in Redis
```

**Strategy 3: Kubernetes Job Restart Policy**

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: ensemble-batch-001
spec:
  backoffLimit: 3  # Retry failed jobs up to 3 times
  template:
    spec:
      restartPolicy: OnFailure
```

**Strategy 4: Simulation-Level Checkpointing**

For individual simulations exceeding 1 minute runtime:

```python
def run_simulation_with_checkpoint(sim_id, params):
    checkpoint_file = f"checkpoints/{sim_id}.pkl"

    if exists(checkpoint_file):
        state = load_checkpoint(checkpoint_file)  # Resume
    else:
        state = initialize(params)

    while not state.complete:
        state = integrate_step(state)
        if step % 1000 == 0:
            save_checkpoint(state, checkpoint_file)

    return state.results
```

#### 3.5.3 Resilience Layer Summary

| Layer | Mechanism | Maximum Loss |
|-------|-----------|--------------|
| Simulation | Checkpoint every N integration steps | < 1 minute |
| Batch | 1000 sims per job, results persisted | < 2 minutes |
| Worker | Kubernetes restart policy | Auto-restart ~30 sec |
| Scheduler | Dask state persisted to Redis | Queue rebuilt ~1 min |
| System | Progress tracked in MLflow | Resume from last batch |

#### 3.5.4 Operational Procedures for Long Runs

**Pre-Run Checklist:**

- [ ] Power adapter connected and verified
- [ ] Lid close behavior set to "do nothing" (`gnome-tweaks` or `systemd-logind`)
- [ ] Automatic sleep/suspend disabled
- [ ] `unattended-upgrades` auto-reboot disabled
- [ ] Thermal headroom verified (`sensors` command)
- [ ] PVC storage space confirmed sufficient
- [ ] Grafana monitoring dashboard active
- [ ] (Optional) Alert webhook configured for failure notification

**Post-Crash Recovery:**

```bash
# Check job status
kubectl get jobs -n cascades

# View completed experiments
mlflow ui  # or check MLflow pod logs

# Resume: Dask scheduler automatically requeues pending work
# Typical recovery time: < 5 minutes to resume
```

#### 3.5.5 Expected Outcomes

| Scenario | Without Resilience | With Resilience |
|----------|-------------------|-----------------|
| 12-hour run, crash at hour 11 | **Lose 11 hours** | **Lose ~2 minutes** |
| Overnight unattended run | Requires monitoring | Safe to run |
| Driver crash mid-ensemble | Full restart needed | Auto-recovery |

### 3.6 AI-Assisted Research Workflow

This project employs a collaborative human-AI research model where Claude Code serves as the primary technical implementer while the human researcher provides direction, review, and strategic decisions.

#### 3.6.1 Role Distribution

| Responsibility | Claude Code | Human Researcher |
|----------------|-------------|------------------|
| **Research Lead** | Design experiments, write code, run simulations | Review approach, ask questions, approve direction |
| **Technical Implementation** | All coding, deployment, debugging | Observe, learn, provide feedback |
| **Concept Translation** | Explain scientific concepts in accessible terms | Flag anything unclear |
| **Documentation** | Generate technical reports and papers | Review and co-author |
| **Decision Points** | Present options with tradeoffs | Make strategic choices |

#### 3.6.2 Claude Code Capabilities Applied

| Research Phase | Claude Code Function |
|----------------|---------------------|
| Phase 0: Infrastructure | Write k3s manifests, Dockerfiles, configure cluster |
| Phase 1: Foundation | Execute notebooks, explain PyCascades architecture |
| Phase 2: Theory | Literature synthesis, mathematical derivation, parameter research |
| Phase 3: Implementation | Write Python classes, tests, documentation |
| Phase 4: Analysis | Run ensembles, statistical analysis, visualization |
| Phase 5: Documentation | Write technical report, prepare PR, draft communications |

#### 3.6.3 Interaction Patterns

**Concept Translation Example:**

> *Claude Code explains:* "The fold bifurcation `dx/dt = a + bx - x³ - c` behaves like a ball rolling on a landscape. When parameter `c` increases past a critical value, one of the stable valleys disappears—the ball has no choice but to roll into a different basin. This is the 'tipping' event. Our energy extension makes `c` depend on energy availability: when power is cut, the landscape deforms and the system tips more easily."

**Decision Point Example:**

> *Claude Code presents:* "For GPU acceleration, we have three options:
> 1. **Numba CUDA** - Direct GPU kernels, most control, more complex
> 2. **CuPy** - NumPy drop-in replacement, easiest migration
> 3. **JAX** - Automatic differentiation + GPU, useful if we want gradients later
>
> Given our SDE integration workload, I recommend Option 2 (CuPy) for simplicity. Do you want me to proceed with this approach?"

#### 3.6.4 Optional: Local LLM Deployment

For offline literature processing or when Claude API is unavailable:

```yaml
# Ollama deployment for local inference
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ollama-llm
  namespace: cascades
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: ollama
        image: ollama/ollama:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

**Use cases:**
- Batch processing of PDF literature for key parameter extraction
- Generating summaries of simulation results
- Drafting documentation sections

**Note:** Local LLM competes for GPU resources with simulation workers. Schedule literature processing during low-compute phases.

---

## 4. Research Phases

### 4.0 Phase 0: Infrastructure Setup

**Objective:** Establish the k3s-based research environment with GPU support, distributed computing capabilities, and operational resilience.

**Tasks:**

1. Create project directory structure in `Projects/cascades/`
2. Initialize git repository for version control
3. Create k3s namespace and resource quotas in `ODIN/k3s-configs/namespaces/cascades/`
4. Configure NVIDIA GPU operator access for the namespace
5. Create persistent volume claims for code, data, checkpoints, and artifacts
6. Build containerized research environment (Dockerfile with conda + all dependencies)
7. Deploy JupyterLab with GPU access
8. Deploy MLflow for experiment tracking
9. Deploy Dask scheduler and worker templates
10. Deploy Redis for task state persistence
11. Configure HorizontalPodAutoscaler for CPU workers
12. Verify end-to-end pipeline: submit test job → execute on worker → persist results
13. Document operational procedures for long-running ensembles

**Deliverables:**

- Complete k3s manifest set in `ODIN/k3s-configs/namespaces/cascades/`
- Working Dockerfile and container image in local registry
- Operational JupyterLab accessible via browser
- MLflow tracking server with web UI
- Dask cluster capable of scaling 0-12 CPU workers
- Pre-run checklist and recovery procedures documented

### 4.1 Phase 1: Foundation and Replication (Weeks 1–4)

**Objective:** Establish working familiarity with PyCascades by reproducing published results and understanding the codebase architecture.

**Tasks:**

1. Work through all three example notebooks in order: (a) Amazon moisture recycling network, (b) Interacting climate tipping elements, (c) Global trade network cascades.

2. Clone and execute the cryosphere tipping elements analysis code from Jonathan Rosser's repository, verifying reproduction of Sobol sensitivity analysis results.

3. Document the class hierarchy: tipping_element base class, coupling class, network builder utilities, integration routines.

4. Read the PyCascades description paper (Wunderling et al., European Physical Journal Special Topics, 2021) and the associated Nature Climate Change overshoot paper.

5. Experiment with parameter variations to develop intuition for model behavior: coupling strengths, threshold distributions, network topologies.

**Deliverables:**

- Annotated Jupyter notebooks with reproduced results and commentary.
- Architecture diagram of PyCascades class structure.
- Technical notes on extension points for custom tipping element types.

### 4.2 Phase 2: Theoretical Framework Development (Weeks 5–8)

**Objective:** Formalize the energy accounting framework and design the mathematical structure for energy-constrained tipping dynamics.

**Core Concepts to Formalize:**

- **Energy State Variable (E):** Each tipping element i has an associated energy availability Eᵢ(t) representing the power throughput accessible to maintain system function.

- **Energy-Dependent Stability:** The critical threshold parameter c_crit becomes a function of energy availability: c_crit(E). As E decreases, the system becomes easier to tip (threshold decreases).

- **Energy-Mediated Coupling:** Coupling strength κᵢⱼ between elements i and j is modulated by energy flow capacity: κᵢⱼ(E) = κ⁰ᵢⱼ · f(Eᵢ, Eⱼ) where f is a saturation function.

- **Energy Cascade Dynamics:** When element i tips, it may disrupt energy flows to connected elements: dEⱼ/dt includes terms representing energy supply from element i.

**Mathematical Framework:**

The standard PyCascades fold-bifurcation form is:

```
dx/dt = a + bx - x³ - c
```

The energy-extended form becomes:

```
dx/dt = a + bx - x³ - c(E)
dE/dt = Φ_in(t) - Φ_out(x, E) - Σⱼ Φᵢⱼ(xⱼ)
```

where Φ_in is external energy supply, Φ_out is energy consumption (dependent on system state), and Φᵢⱼ represents energy transfer disruption when connected elements tip.

**Tasks:**

1. Review literature on dissipative structures and far-from-equilibrium thermodynamics (Prigogine, Nicolis).

2. Derive functional forms for c(E), κ(E), and energy flow dynamics that satisfy physical constraints.

3. Establish parameter ranges from empirical data: urban energy consumption per capita, infrastructure energy requirements, grid interconnection capacities.

4. Design the class interface for EnergyConstrainedTippingElement to integrate with existing PyCascades infrastructure.

**Deliverables:**

- Technical specification document for energy-constrained tipping element mathematics.
- Parameter estimation table with literature sources.
- Class interface design document.

### 4.3 Phase 3: Implementation (Weeks 9–14)

**Objective:** Implement the energy-constrained tipping element classes and coupling functions within the PyCascades framework.

**Implementation Components:**

**A. EnergyConstrainedTippingElement Class**

```python
class EnergyConstrainedTippingElement(pc.tipping_element):
    """Tipping element with energy state variable."""
    
    def __init__(self, E_nominal, E_critical, decay_rate, **kwargs):
        super().__init__(**kwargs)
        self.E_nominal = E_nominal   # Normal operating energy
        self.E_critical = E_critical # Energy below which system tips
        self.decay_rate = decay_rate # Energy loss rate when supply disrupted
    
    def c_threshold(self, E):
        """Energy-dependent threshold function."""
        return self.c_base * (E / self.E_nominal) ** self.sensitivity
```

**B. EnergyMediatedCoupling Class**

```python
class EnergyMediatedCoupling(pc.coupling):
    """Coupling strength modulated by energy availability."""
    
    def effective_strength(self, E_source, E_target):
        saturation = min(E_source, E_target) / self.E_reference
        return self.base_strength * np.tanh(saturation)
```

**C. HumanSettlementElement Class**

A specialized tipping element representing urban infrastructure stability under climate and energy stress, with parameters for carrying capacity, infrastructure degradation thresholds, and adaptation rates.

**Tasks:**

1. Implement EnergyConstrainedTippingElement with unit tests.
2. Implement EnergyMediatedCoupling with unit tests.
3. Implement HumanSettlementElement specialized class.
4. Create integration tests verifying compatibility with existing PyCascades network builders.
5. Develop visualization utilities for energy state evolution alongside standard tipping dynamics.
6. Document API following PyCascades conventions.

**Deliverables:**

- Working Python module: `pycascades/tipping_elements/energy_constrained.py`
- Test suite with >90% code coverage.
- API documentation and usage examples.

### 4.4 Phase 4: Analysis and Validation (Weeks 15–20)

**Objective:** Conduct systematic analysis comparing energy-constrained and standard model configurations to characterize the contribution of energy flow mechanisms to cascade dynamics.

**Analysis Components:**

**A. Sensitivity Analysis**

Apply Sobol global sensitivity analysis to identify parameters with greatest influence on cascade outcomes. Key parameters include: E_nominal, E_critical, decay_rate, coupling_saturation, energy_sensitivity.

```python
problem = {
    'num_vars': 8,
    'names': ['E_nominal', 'E_critical', 'decay_rate', 'coupling_sat',
             'energy_sens', 'threshold_base', 'warming_rate', 'network_density'],
    'bounds': [[...], [...], ...]  # From literature
}
samples = saltelli.sample(problem, 2**14)  # ~500K simulations
```

**B. Comparative Cascade Analysis**

Run matched ensembles with identical initial conditions and random seeds under: (1) standard PyCascades parameterization, (2) energy-constrained parameterization with high energy availability, (3) energy-constrained parameterization with stressed energy supply.

Metrics: cascade probability, cascade size distribution, time to cascade completion, pathway diversity.

**C. Multi-Layer Network Construction**

Build a three-layer network connecting:
- **Layer 1 (Climate):** GIS, WAIS, AMOC, Amazon, permafrost, monsoons.
- **Layer 2 (Biosphere):** ecosystem service providers, species range shifts, agricultural productivity.
- **Layer 3 (Human):** major urban agglomerations, energy infrastructure nodes, financial centers.

Inter-layer connections parameterized by energy flow pathways identified in literature.

**Tasks:**

1. Define parameter distributions from literature for Sobol analysis.
2. Execute sensitivity analysis ensemble (~500K simulations, estimated 48-72 hours).
3. Compute and visualize Sobol indices (total effect, first-order, second-order interactions).
4. Execute comparative ensemble under matched conditions.
5. Construct multi-layer network with energy flow constraints.
6. Analyze cascade pathways in multi-layer configuration.
7. Statistical analysis of energy constraint contribution to cascade risk.

**Deliverables:**

- Sobol sensitivity analysis results with visualizations.
- Comparative analysis quantifying energy constraint effects.
- Multi-layer network model with documentation.
- Reproducible analysis scripts and data.

### 4.5 Phase 5: Documentation and Contribution (Weeks 21–24)

**Objective:** Prepare findings for dissemination and contribute code extensions to the open-source ecosystem.

**Tasks:**

1. Write comprehensive documentation for energy-constrained modules.
2. Create tutorial notebook demonstrating energy-constrained cascade analysis.
3. Prepare technical report summarizing methodology and findings.
4. Open pull request to pik-copan/pycascades repository.
5. Contact PIK COPAN team (core@pik-potsdam.de) with summary of work.
6. Assess potential for conference presentation (AGU, EGU) or journal submission.

**Deliverables:**

- GitHub repository with complete codebase, tests, and documentation.
- Technical report (target: 15-20 pages with figures).
- Pull request to upstream PyCascades repository.
- Communication with PIK researchers documenting engagement.

---

## 5. Project Timeline

| Phase | Activities | Duration |
|-------|------------|----------|
| Phase 0 | Infrastructure Setup | Week 0 (Prerequisites) |
| Phase 1 | Foundation and Replication | Weeks 1–4 |
| Phase 2 | Theoretical Framework Development | Weeks 5–8 |
| Phase 3 | Implementation | Weeks 9–14 |
| Phase 4 | Analysis and Validation | Weeks 15–20 |
| Phase 5 | Documentation and Contribution | Weeks 21–24 |
| **Total** | **Complete Project Cycle** | **~6 months** |

*Note: Timeline assumes part-time effort (~10-15 hours/week) alongside professional responsibilities. Phases may overlap or extend based on findings and available time. Phase 0 is a prerequisite that establishes the computational infrastructure before formal research begins.*

---

## 6. Key References

1. Wunderling, N., Krönke, J., Wohlfarth, V., et al. (2021). Modelling nonlinear dynamics of interacting tipping elements on complex networks: the PyCascades package. *European Physical Journal Special Topics*.

2. Wunderling, N., Winkelmann, R., et al. (2024). Global warming overshoots increase risks of climate tipping cascades in a network model. *Nature Climate Change*.

3. Donges, J.F., Heitzig, J., et al. (2020). Earth system modeling with endogenous and dynamic human societies: the copan:CORE open World-Earth modeling framework. *Earth System Dynamics*, 11, 395-413.

4. Farahbakhsh, I., Bauch, C.T., Anand, M. (2022). Modelling coupled human-environment complexity for the future of the biosphere. *Philosophical Transactions B*.

5. Yumashev, D., et al. (2024). Tipping points in coupled human-environment system models: a review. *Earth System Dynamics*, 15, 947.

6. Lenton, T.M., et al. (2023). Global Tipping Points Report 2023. University of Exeter.

7. Prigogine, I., Nicolis, G. (1977). *Self-Organization in Non-Equilibrium Systems*. Wiley.

8. World Bank (2021). Groundswell Part 2: Acting on Internal Climate Migration.

---

## 7. Project Resources

**Code Repositories:**

- PyCascades: https://github.com/pik-copan/pycascades
- pyunicorn: https://github.com/pik-copan/pyunicorn
- pycopancore: https://github.com/pik-copan/pycopancore
- Cryosphere analysis: https://github.com/JonathanRosser/Cryosphere-tipping-elements

**PIK COPAN Contact:**

- Email: core@pik-potsdam.de
- Website: https://www.pik-potsdam.de/copan

**Documentation:**

- PyCascades paper: DOI 10.1140/epjs/s11734-021-00155-4
- copan:CORE paper: DOI 10.5194/esd-11-395-2020
- pyunicorn documentation: https://github.com/pik-copan/pyunicorn

---

*Document Version 1.1 — December 2025*

*Revision history:*
- *v1.1: Added sections 3.4 (k3s Infrastructure), 3.5 (Operational Resilience), 3.6 (AI-Assisted Workflow), and Phase 0*
- *v1.0: Initial research proposal*

*This is a living document. Updates will track project progress and incorporate lessons learned.*
