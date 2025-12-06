# Foundations Reading List

**Building Conceptual Fluency for AI-Assisted Climate Systems Research**

*Companion to: Energy-Constrained Tipping Cascades Research Project*

---

> **How to Use This Reading List**
>
> Your role in this research is that of an informed director—someone who understands **what** the science is trying to accomplish, **why** it matters, and **whether** the AI's outputs make sense, even if you don't personally derive every equation. Think of it like being a film director who doesn't operate the camera but knows exactly what shot they need.
>
> This list is organized by conceptual tracks rather than strict sequence. Start with Track 1 (Climate Tipping Points) since it's the domain context for everything else. Then explore other tracks based on what feels most relevant to your current phase of work. You don't need to complete all readings before starting—build knowledge in parallel with the research.
>
> **Difficulty levels:**
> - **◆ Accessible** — Popular science, no prerequisites
> - **◆◆ Intermediate** — Some technical content, concepts explained
> - **◆◆◆ Technical** — Research papers, math present but skimmable

---

## Table of Contents

1. [Track 1: Climate Tipping Points](#track-1-climate-tipping-points)
2. [Track 2: Complex Systems and Nonlinear Dynamics](#track-2-complex-systems-and-nonlinear-dynamics)
3. [Track 3: Thermodynamics and Dissipative Structures](#track-3-thermodynamics-and-dissipative-structures)
4. [Track 4: Network Science](#track-4-network-science)
5. [Track 5: Coupled Human-Environment Systems](#track-5-coupled-human-environment-systems)
6. [Track 6: Technical Methods](#track-6-technical-methods)
7. [Supplementary: Video and Audio Resources](#supplementary-video-and-audio-resources)
8. [Suggested Reading Order](#suggested-reading-order)

---

## Track 1: Climate Tipping Points

*Core domain knowledge.* Start here to understand what tipping points are, why they matter, and how scientists think about them. This is the "what are we modeling" foundation.

### 1.1 Essential Starting Points

---

#### Global Tipping Points Report 2023 — Executive Summary

| | |
|---|---|
| **Difficulty** | ◆ Accessible |
| **Time** | ~3 hours |
| **Priority** | ⭐ START HERE |
| **Author** | University of Exeter Global Systems Institute |

**What you'll learn:** The current scientific consensus on which Earth systems are approaching tipping points, what "tipping" actually means, and the interconnections between different tipping elements. This report is the authoritative reference for the field as of 2023.

**Why it matters for your project:** Provides the catalog of tipping elements you'll be modeling and their estimated thresholds. The discussion of cascading risks directly motivates your energy-flow extension.

**Access:** https://global-tipping-points.org (free PDF download)

---

#### "Climate Tipping Points — Too Risky to Bet Against"

| | |
|---|---|
| **Difficulty** | ◆ Accessible |
| **Time** | ~45 minutes |
| **Author** | Lenton, Rockström, et al. — Nature Comment (2019) |

**What you'll learn:** A concise overview of tipping point science from the researchers who defined the field. Introduces the concept of cascading tipping points and why interactions between elements matter.

**Why it matters:** This paper is the conceptual ancestor of PyCascades. Understanding its argument helps you see what the modeling framework is trying to capture.

**Access:** DOI 10.1038/d41586-019-03595-0 (open access)

---

#### The Tipping Point (Climate Context)

| | |
|---|---|
| **Difficulty** | ◆ Accessible |
| **Time** | ~4 hours (full book, skim-friendly) |
| **Author** | David Spratt & Ian Dunlop — "What Lies Beneath" (2018) |

**What you'll learn:** A policy-oriented synthesis of tipping point risks written for non-scientists. Good for understanding why this research matters beyond academic interest.

**Access:** Free PDF from Breakthrough National Centre for Climate Restoration

---

### 1.2 Deeper Understanding

---

#### "Tipping Elements in the Earth's Climate System"

| | |
|---|---|
| **Difficulty** | ◆◆ Intermediate |
| **Time** | ~2 hours |
| **Author** | Lenton et al. — PNAS (2008) |

**What you'll learn:** The foundational paper that defined "tipping elements" as a scientific concept. Establishes the framework for identifying and characterizing potential tipping points in the climate system.

**Reading strategy:** Focus on the conceptual definitions and Table 1 (the catalog of tipping elements). The mathematical details can be skimmed.

**Access:** DOI 10.1073/pnas.0705414105 (open access)

---

#### "Exceeding 1.5°C Global Warming Could Trigger Multiple Climate Tipping Points"

| | |
|---|---|
| **Difficulty** | ◆◆ Intermediate |
| **Time** | ~1.5 hours |
| **Author** | Armstrong McKay et al. — Science (2022) |

**What you'll learn:** Updated assessment of tipping point thresholds based on latest evidence. Shows which elements may already be approaching tipping and how thresholds have been revised downward.

**Why it matters:** Provides current best estimates for threshold parameters you'll use in modeling.

**Access:** DOI 10.1126/science.abn7950

---

#### "Global Warming Overshoots Increase Risks of Climate Tipping Cascades"

| | |
|---|---|
| **Difficulty** | ◆◆◆ Technical |
| **Time** | ~2 hours |
| **Priority** | ⭐ PROJECT CORE PAPER |
| **Author** | Wunderling et al. — Nature Climate Change (2023) |

**What you'll learn:** How PyCascades is used in practice for published research. This paper shows the methodology you'll be extending—network-based cascade modeling with Monte Carlo analysis.

**Reading strategy:** Focus on Figures 1-3 and the Methods section. The supplementary information has the technical details you can reference later.

**Access:** DOI 10.1038/s41558-023-01645-0

---

## Track 2: Complex Systems and Nonlinear Dynamics

*The mathematical worldview.* Tipping points are a specific type of phenomenon in complex systems. This track builds intuition for how systems can change suddenly, why small causes can have large effects, and what "nonlinear" really means.

### 2.1 Conceptual Foundations

---

#### Thinking in Systems: A Primer

| | |
|---|---|
| **Difficulty** | ◆ Accessible |
| **Time** | ~6 hours (book) |
| **Priority** | ⭐ HIGHLY RECOMMENDED |
| **Author** | Donella Meadows (2008) |

**What you'll learn:** The clearest introduction to systems thinking ever written. Covers stocks, flows, feedback loops, delays, and system archetypes. No math required—pure conceptual understanding.

**Why it matters:** This book will fundamentally change how you think about interconnected systems. The mental models it provides are exactly what you need to direct AI-assisted research effectively.

**Access:** Book (~$15). Also available in many libraries. Worth owning.

---

#### Sync: How Order Emerges from Chaos

| | |
|---|---|
| **Difficulty** | ◆ Accessible |
| **Time** | ~8 hours (book, skim-friendly) |
| **Author** | Steven Strogatz (2003) |

**What you'll learn:** How synchronized behavior emerges in complex systems—from fireflies to neurons to power grids. Builds intuition for coupled oscillators and collective dynamics without equations.

**Why it matters:** The coupling mechanisms in your tipping cascade model are fundamentally about synchronization and propagation. Strogatz makes these concepts viscerally understandable.

**Access:** Book (~$18). Popular science—enjoyable read.

---

#### Complexity: A Guided Tour

| | |
|---|---|
| **Difficulty** | ◆ Accessible |
| **Time** | ~5 hours (book) |
| **Author** | Melanie Mitchell (2009) |

**What you'll learn:** Broad overview of complexity science—cellular automata, genetic algorithms, network theory, scaling laws. Accessible introduction to the Santa Fe Institute worldview.

**Access:** Book (~$18). Also available at most libraries.

---

### 2.2 Bifurcations and Phase Transitions

---

#### "Early Warning Signals for Critical Transitions"

| | |
|---|---|
| **Difficulty** | ◆◆ Intermediate |
| **Time** | ~1 hour |
| **Priority** | ⭐ KEY CONCEPT |
| **Author** | Scheffer et al. — Nature (2009) |

**What you'll learn:** How systems behave as they approach tipping points—critical slowing down, increased variance, flickering. These are the signatures that indicate a system is losing stability.

**Why it matters:** Understanding early warning signals helps you interpret what the model outputs mean. When your simulations show certain patterns, you'll know what they indicate.

**Access:** DOI 10.1038/nature08227

---

#### Critical Transitions in Nature and Society

| | |
|---|---|
| **Difficulty** | ◆◆ Intermediate |
| **Time** | ~3 hours (selected chapters) |
| **Author** | Marten Scheffer (2009) |

**What you'll learn:** Deep dive into regime shifts—how lakes flip from clear to turbid, how deserts expand, how societies collapse. The book that established the modern framework for thinking about tipping points.

**Reading strategy:** Focus on Chapters 1-4 (conceptual framework) and Chapter 10 (societal transitions). Other chapters are excellent but optional.

**Access:** Book (~$45). May be available through library.

---

#### "Bifurcation" — Complexity Explorer Tutorial

| | |
|---|---|
| **Difficulty** | ◆ Accessible |
| **Time** | ~30 minutes |
| **Priority** | ⭐ VISUAL LEARNING |
| **Author** | Santa Fe Institute — Online Course Module |

**What you'll learn:** Visual, interactive introduction to bifurcations—the mathematical structures underlying tipping points. See how systems can suddenly switch between stable states.

**Why it matters:** PyCascades models "cusp" and "Hopf" bifurcations. This will give you visual intuition for what those terms mean.

**Access:** https://www.complexityexplorer.org (free, requires registration)

---

## Track 3: Thermodynamics and Dissipative Structures

*The energy-flow foundation.* Your project's key innovation is treating human systems as dissipative structures—organized systems that require constant energy flow to maintain their organization. This track provides the conceptual basis for that framing.

### 3.1 Core Concepts

---

#### Order Out of Chaos: Man's New Dialogue with Nature

| | |
|---|---|
| **Difficulty** | ◆ Accessible |
| **Time** | ~6 hours (book) |
| **Priority** | ⭐ THEORETICAL FOUNDATION |
| **Author** | Ilya Prigogine & Isabelle Stengers (1984) |

**What you'll learn:** The foundational text on dissipative structures, written by the Nobel laureate who developed the concept. Explains how ordered structures can emerge and persist in systems far from equilibrium—but only through continuous energy throughput.

**Why it matters:** This is THE source for your project's central insight—that cities and infrastructure are dissipative structures. When you parameterize energy requirements for maintaining system stability, you're operationalizing Prigogine's framework.

**Reading strategy:** Focus on Parts II and III. Part I is historical context (skimmable). The prose is philosophical but accessible.

**Access:** Book (~$20). A classic worth owning.

---

#### Into the Cool: Energy Flow, Thermodynamics, and Life

| | |
|---|---|
| **Difficulty** | ◆ Accessible |
| **Time** | ~5 hours (book) |
| **Author** | Eric Schneider & Dorion Sagan (2005) |

**What you'll learn:** Accessible introduction to non-equilibrium thermodynamics and its application to living systems and ecosystems. Explains energy gradients, entropy production, and why complex structures form.

**Why it matters:** Provides intuitive understanding of energy flows in complex systems. Will help you conceptualize the energy state variables in your model.

**Access:** Book (~$25). Engaging popular science.

---

### 3.2 Energy in Human Systems

---

#### "Cities as Dissipative Structures"

| | |
|---|---|
| **Difficulty** | ◆◆ Intermediate |
| **Time** | ~1.5 hours |
| **Author** | Portugali — Chapter in "Complexity, Cognition and the City" (2011) |

**What you'll learn:** Direct application of dissipative structure theory to urban systems. How cities self-organize, require energy throughput, and can undergo phase transitions.

**Why it matters:** This is precisely the framing your HumanSettlementElement class will operationalize.

**Access:** Chapter in edited volume. May require library access or purchase.

---

#### "Energy and the Wealth of Nations" — Selected Chapters

| | |
|---|---|
| **Difficulty** | ◆◆ Intermediate |
| **Time** | ~1 hour |
| **Author** | Charles Hall & Kent Klitgaard (2018) |

**What you'll learn:** How energy flows underpin economic activity. Introduction to EROI (Energy Return on Investment) and biophysical economics. Argues that economic systems are fundamentally energy transformation systems.

**Reading strategy:** Chapters 1-5 provide the core concepts. Later chapters apply to specific cases.

**Access:** Book (~$50). Consider library access or selected chapters.

---

#### "The Metabolism of Cities"

| | |
|---|---|
| **Difficulty** | ◆◆ Intermediate |
| **Time** | ~45 minutes |
| **Author** | Abel Wolman — Scientific American (1965) |

**What you'll learn:** Classic paper treating cities as organisms with measurable inputs (water, food, fuel) and outputs (sewage, garbage, emissions). Foundational for urban metabolism studies.

**Why it matters:** Provides concrete data on urban energy/material flows that can inform parameter estimation.

**Access:** Scientific American archives or widely cited/excerpted online.

---

## Track 4: Network Science

*The structural framework.* PyCascades models tipping elements as nodes in networks, with cascades propagating along edges. This track provides intuition for network concepts you'll encounter.

### 4.1 Foundations

---

#### Linked: The New Science of Networks

| | |
|---|---|
| **Difficulty** | ◆ Accessible |
| **Time** | ~6 hours (book) |
| **Priority** | ⭐ HIGHLY RECOMMENDED |
| **Author** | Albert-László Barabási (2002) |

**What you'll learn:** The foundational popular introduction to network science. Covers small-world networks, scale-free networks, hubs, cascading failures—all concepts you'll encounter in PyCascades.

**Why it matters:** When you specify network topologies (Erdös-Rényi, Barabási-Albert, Watts-Strogatz), this book gives you intuitive understanding of what those structures mean and why they matter for cascade dynamics.

**Access:** Book (~$18). Engaging, accessible. A classic.

---

#### Six Degrees: The Science of a Connected Age

| | |
|---|---|
| **Difficulty** | ◆ Accessible |
| **Time** | ~5 hours (book) |
| **Author** | Duncan Watts (2003) |

**What you'll learn:** Complementary perspective on network science from one of its founders. Strong on cascade dynamics, epidemics on networks, and how structure affects propagation.

**Why it matters:** Watts-Strogatz small-world networks are one of the topologies in PyCascades. This gives you the background story.

**Access:** Book (~$18).

---

### 4.2 Cascades on Networks

---

#### "Cascade Control and Defense in Complex Networks"

| | |
|---|---|
| **Difficulty** | ◆◆ Intermediate |
| **Time** | ~1 hour |
| **Priority** | ⭐ DIRECTLY RELEVANT |
| **Author** | Motter — Physical Review Letters (2004) |

**What you'll learn:** How cascading failures propagate through networks—power grids, communication networks, and similar infrastructure. Introduction to the mathematics of cascade dynamics.

**Reading strategy:** Focus on the conceptual setup and interpretation. The math formalizes cascade propagation rules similar to what PyCascades implements.

**Access:** DOI 10.1103/PhysRevLett.93.098701

---

#### "Multilayer Networks" — Review Article

| | |
|---|---|
| **Difficulty** | ◆◆ Intermediate |
| **Time** | ~1.5 hours |
| **Author** | Kivelä et al. — Journal of Complex Networks (2014) |

**What you'll learn:** How to model systems with multiple types of connections or multiple interacting layers. Directly relevant to your three-layer (climate-biosphere-human) network design.

**Reading strategy:** Focus on the conceptual framework and examples. Mathematical formalism can be referenced as needed.

**Access:** DOI 10.1093/comnet/cnu016 (open access)

---

## Track 5: Coupled Human-Environment Systems

*The integration challenge.* Your project bridges climate tipping points with human systems. This track covers the emerging field of coupled human-environment system modeling and the specific challenges of integrating social dynamics.

### 5.1 Foundational Framework

---

#### "Earth System Modeling with Endogenous and Dynamic Human Societies: copan:CORE"

| | |
|---|---|
| **Difficulty** | ◆◆ Intermediate |
| **Time** | ~1.5 hours |
| **Priority** | ⭐ PROJECT FRAMEWORK |
| **Author** | Donges, Heitzig et al. — Earth System Dynamics (2020) |

**What you'll learn:** The design philosophy of the copan:CORE World-Earth modeling framework—how to integrate natural and social processes in a coherent modeling architecture.

**Why it matters:** This is the broader framework within which PyCascades sits. Understanding copan:CORE helps you see how your extension could integrate with the larger ecosystem.

**Access:** DOI 10.5194/esd-11-395-2020 (open access)

---

#### "Modelling Coupled Human-Environment Complexity for the Future of the Biosphere"

| | |
|---|---|
| **Difficulty** | ◆◆ Intermediate |
| **Time** | ~1 hour |
| **Author** | Farahbakhsh, Bauch & Anand — Philosophical Transactions B (2022) |

**What you'll learn:** Review of CHES modeling approaches and the challenges of integrating human behavior with ecological dynamics. Identifies the gap your project addresses—far fewer CHES models than uncoupled models.

**Access:** DOI 10.1098/rstb.2021.0382 (open access)

---

### 5.2 Social Tipping Points

---

#### "Social Tipping Dynamics for Stabilizing Earth's Climate by 2050"

| | |
|---|---|
| **Difficulty** | ◆◆ Intermediate |
| **Time** | ~1 hour |
| **Author** | Otto et al. — PNAS (2020) |

**What you'll learn:** How tipping point thinking applies to social systems—identifying social tipping elements that could trigger rapid decarbonization. Shows that tipping dynamics aren't just physical phenomena.

**Why it matters:** Your human settlement elements are social tipping elements. This paper provides conceptual grounding for modeling them.

**Access:** DOI 10.1073/pnas.1900577117 (open access)

---

#### Groundswell Part 2: Acting on Internal Climate Migration

| | |
|---|---|
| **Difficulty** | ◆ Accessible |
| **Time** | ~2 hours |
| **Author** | World Bank (2021) |

**What you'll learn:** Projections for climate-driven internal migration through 2050—216 million people across six regions. Identifies hotspots of out-migration and in-migration, with implications for urban systems.

**Why it matters:** Provides empirical grounding for human displacement as a cascade mechanism. The cities identified as receiving climate migrants are candidates for your human settlement tipping elements.

**Access:** Free PDF from World Bank Open Knowledge Repository

---

## Track 6: Technical Methods

*Working with the tools.* While AI will handle implementation details, understanding these methods helps you direct the work effectively and validate outputs.

### 6.1 PyCascades Documentation

---

#### PyCascades Description Paper

| | |
|---|---|
| **Difficulty** | ◆◆◆ Technical |
| **Time** | ~2 hours |
| **Priority** | ⭐ REQUIRED |
| **Author** | Wunderling et al. — European Physical Journal Special Topics (2021) |

**What you'll learn:** The official documentation for PyCascades—its mathematical foundations, class structure, and capabilities. This is the technical reference for everything you'll build on.

**Reading strategy:** Read through once to understand the overall architecture. Return to specific sections as needed during implementation. The equations formalize the dynamics; focus on understanding what they represent conceptually.

**Access:** DOI 10.1140/epjs/s11734-021-00155-4

---

#### PyCascades Example Notebooks

| | |
|---|---|
| **Difficulty** | ◆◆ Intermediate |
| **Time** | ~3 hours (hands-on) |
| **Author** | GitHub Repository — /examples/ |

**What you'll learn:** Practical experience with PyCascades through working examples: Amazon moisture recycling, climate tipping element interactions, economic cascades.

**How to use:** Work through these with Claude Code. Ask Claude to explain each cell, modify parameters, and observe how outputs change. Build intuition through experimentation.

**Access:** https://github.com/pik-copan/pycascades/tree/master/examples

---

### 6.2 Sensitivity Analysis

---

#### "Sensitivity Analysis: A Review of Recent Advances"

| | |
|---|---|
| **Difficulty** | ◆◆ Intermediate |
| **Time** | ~1 hour |
| **Author** | Saltelli et al. — European Journal of Operational Research (2020) |

**What you'll learn:** Overview of sensitivity analysis methods, especially Sobol indices. Explains what first-order effects, total effects, and interactions mean.

**Why it matters:** Phase 4 of your project involves Sobol sensitivity analysis. Understanding what the outputs mean helps you interpret and present results.

**Access:** DOI 10.1016/j.ejor.2020.01.037

---

#### SALib Documentation and Tutorials

| | |
|---|---|
| **Difficulty** | ◆◆ Intermediate |
| **Time** | ~1 hour (hands-on) |
| **Author** | SALib Python Package |

**What you'll learn:** Practical use of the SALib package for sensitivity analysis—the same tool used in the cryosphere tipping elements paper you'll reproduce.

**How to use:** Work through the getting started guide with Claude Code. Run a simple Sobol analysis on a toy problem to build familiarity before applying to PyCascades.

**Access:** https://salib.readthedocs.io

---

### 6.3 Python Scientific Computing

---

#### Python Data Science Handbook — Selected Chapters

| | |
|---|---|
| **Difficulty** | ◆ Accessible |
| **Time** | ~4 hours (reference) |
| **Author** | Jake VanderPlas (2016) |

**What you'll learn:** Practical NumPy and Matplotlib skills for working with simulation outputs. Focus on array operations and visualization.

**How to use:** Reference as needed. When Claude Code produces visualizations or data manipulation, you can ask it to explain using concepts from this book.

**Access:** Free online at https://jakevdp.github.io/PythonDataScienceHandbook/

---

## Supplementary: Video and Audio Resources

*Alternative learning formats.* These resources cover similar material in video/podcast form—useful for learning while commuting or taking a break from reading.

### Video Lectures

1. **"Tipping Points in the Climate System"** — Tim Lenton, Royal Society lecture (YouTube, ~1 hour). Accessible overview from the leading researcher.

2. **"Introduction to Complexity"** — Santa Fe Institute, Complexity Explorer course (free, ~15 hours total). Comprehensive introduction to complex systems with interactive elements.

3. **"Network Science"** — Albert-László Barabási, online course (free video lectures). Visual introduction to networks from the author of Linked.

4. **"The Era of Global Tipping Points"** — Potsdam Institute presentations (YouTube). See how PIK researchers present this work.

5. **"Nonlinear Dynamics and Chaos"** — Steven Strogatz, Cornell lectures (YouTube, full course). Deeper mathematical treatment if interested.

### Podcasts

1. **Complexity Podcast** — Santa Fe Institute. Episodes on tipping points, networks, and complex systems.

2. **Tipping Point** — BBC Radio 4. Series on climate tipping points featuring Lenton and other researchers.

3. **Outrage + Optimism** — Climate podcast with episodes on tipping points and systems change.

---

## Suggested Reading Order

Here's a phased approach that builds knowledge in parallel with your project phases:

### Before Starting (Foundation)

- Global Tipping Points Report 2023 — Executive Summary (Track 1)
- Thinking in Systems — Donella Meadows (Track 2)
- Linked — Barabási, Chapters 1-8 (Track 4)

### During Phase 1 (Weeks 1-4)

- PyCascades Description Paper (Track 6)
- Lenton 2008 — Tipping Elements paper (Track 1)
- Wunderling 2023 — Overshoots paper (Track 1)
- Scheffer 2009 — Early Warning Signals (Track 2)

### During Phase 2 (Weeks 5-8)

- Order Out of Chaos — Prigogine (Track 3)
- Cities as Dissipative Structures — Portugali (Track 3)
- copan:CORE paper — Donges et al. (Track 5)
- Multilayer Networks review (Track 4)

### During Phase 3-4 (Weeks 9-20)

- SALib documentation and tutorials (Track 6)
- Sensitivity Analysis review — Saltelli (Track 6)
- Social Tipping Dynamics — Otto et al. (Track 5)
- Groundswell Report — World Bank (Track 5)

### Ongoing / As Needed

- Into the Cool — deeper thermodynamics (Track 3)
- Critical Transitions in Nature and Society — Scheffer (Track 2)
- Six Degrees — Watts (Track 4)
- Complexity: A Guided Tour — Mitchell (Track 2)

---

> **A Note on Reading Strategy**
>
> You don't need to master everything before starting. The goal is fluency, not expertise. Read enough to ask good questions, understand what Claude Code is doing, and recognize when outputs don't make sense.
>
> When working with AI on implementation:
> - Ask Claude to explain concepts in plain language
> - Request analogies to systems you understand (networks, infrastructure)
> - Have Claude walk through code logic before running it
> - Ask "what would we expect to see if this is working correctly?"
>
> Your role is director and validator. The reading builds the judgment to play that role effectively.

---

*Reading List Version 1.0 — December 2025*

*Companion to: Energy-Constrained Tipping Cascades in Coupled Socio-Ecological Systems*
