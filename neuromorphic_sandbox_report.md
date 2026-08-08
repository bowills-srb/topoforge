# Spatial Topology as a First-Class Design Variable: An Empirical Investigation on a Minimal Spiking-Network Testbed

**Author:** Bo Wills (with Claude as research collaborator)
**Date:** August 8, 2026
**Environment:** Pure-NumPy SNN simulator, 1,000-node networks, Windows laptop, single evening session

---

## Abstract

We built a minimal simulation testbed to investigate a single question: how much does the physical placement of computational elements matter, holding the logical workload constant? Across eleven experiments on 1,000-neuron leaky integrate-and-fire (LIF) networks, we measured communication energy (wire-length-squared model), thermal density penalties, traffic congestion under shortest-path routing, and spiking dynamics. The testbed independently reproduced, from first principles, several load-bearing results of modern interconnect and accelerator architecture: a ~79x communication-energy spread attributable to placement alone; constraint-boundary-riding optima under thermal limits; the existential (not merely performance) role of sparse long-range links; the 2–3x advantage of structured over random global wiring; gateway congestion and bottleneck migration; the thermal infeasibility of homogeneous logic-on-logic 3D stacking alongside the viability of heterogeneous memory-on-logic; and the necessity of inhibition for stable network dynamics. None of these results is novel to the field. What is notable is the cost of obtaining them: a laptop, open-source tooling, and roughly one working session. We argue this has implications for how architecture intuition can be built and how early-stage design-space exploration can be democratized.

---

## 1. Motivation and Framing

The starting hypothesis, stated informally: in biological neural systems, physical proximity encodes logical proximity — cortical wiring is dominated by short local connections, with sparse, deliberate long-range projections — and this co-location of related computation is a primary source of the brain's ~20 W energy budget. Contemporary AI accelerators, by contrast, largely decouple logical relationships from physical location: weights live in memory arrays, compute happens in separate units, and the energy cost of shuttling data between them (the von Neumann overhead) now dominates the power budget of transformer inference and training.

The question we set out to test empirically: **if the workload (netlist) is held fixed, how large is the design space opened up by placement, and what constraints shape its optimum?**

## 2. Methods

All experiments use a common substrate:

- **Network:** 1,000 point neurons, organized into 5 clusters of 200. Cluster generation: uniform-random placement within a disc of radius *r* around each of 5 cluster centers, centers uniform in a square of side *S* (both swept as experimental variables).
- **Local connectivity:** *k*-nearest-neighbor (k=10) directed edges, i.e., a fixed logical fan-in per node — the "netlist" held constant across placements.
- **Communication energy model:** E = Σ d² per edge, motivated by wire capacitance scaling linearly with length and switching energy as CV². Vertical (inter-layer) distance is charged a 3x multiplier as a coarse TSV-overhead proxy.
- **Thermal model:** spatial binning into unit cells; superlinear (quadratic) penalty on occupancy/power above a per-cell limit, modeling the nonlinear cost of hotspot mitigation. Two variants proved decisive (see §5).
- **Traffic model:** BFS shortest-path routing of uniformly random source-destination message pairs; per-node transit counts as the congestion metric.
- **Dynamics:** discrete-time LIF (leak 0.90/step, threshold 1.0, 3-step refractory period, background noise), with and without a 20% inhibitory subpopulation at 2x synaptic magnitude.

**Limitations, stated up front.** This is a toy model in every dimension: n=1,000 is six orders of magnitude below relevant scale; the energy model omits repeaters, routing congestion on wires, and technology-dependent wire hierarchies; the thermal model is a caricature of heat diffusion; shortest-path routing ignores queueing and flow control; single seeds per configuration, no variance estimates. The claims below are therefore qualitative and directional — about the *shape* of trade-off structure, not about numeric transferability. As §5 demonstrates, several conclusions are exquisitely sensitive to modeling assumptions, which we treat as a finding in its own right.

## 3. Placement Dominates Communication Energy (Experiments 1–3)

Initial sweeps using proximity-threshold connectivity conflated placement quality with network density (clustered layouts admitted ~10x more edges within threshold, inflating total energy while deflating per-edge energy). The clean experiment fixes the netlist — k=10 nearest neighbors, 10,000 edges for every layout — and measures placement alone:

| Placement | Total energy | Mean wire length |
|---|---|---|
| 2D grid (32×32) | 21,196 | 1.40 |
| Columnar 3D (10×10×10) | 14,752 | 1.20 |
| Loose clusters (10, r=3) | 2,851 | 0.46 |
| Tight clusters (5, r=1) | 269 | 0.14 |

**A 79x spread from placement alone.** With energy quadratic in distance, the 10x wire-length reduction from clustering compounds accordingly. This is the quantitative core of the case for physical locality — the same argument underlying chiplet partitioning, near-memory compute, and dataflow-aware floorplanning.

## 4. Density Is Thermally Capped; the Optimum Rides the Constraint (Experiment 4)

Unconstrained, "cluster tighter" wins without bound. Adding the thermal penalty and sweeping cluster radius (with cluster centers spread proportionally to avoid overlap artifacts) produced the expected U-curve:

| Radius | Comm. energy | Thermal penalty | Total |
|---|---|---|---|
| 4 | 4,046 | 85,300 | 89,346 |
| 6 | 9,103 | 22,700 | 31,803 |
| **8** | **16,183** | **3,450** | **19,633** |
| 10 | 25,286 | 250 | 25,536 |
| 14 | 49,561 | 0 | 49,561 |
| 40 | 404,576 | 0 | 404,576 |

Two structural observations. First, the optimum (r=8) *accepts a residual thermal violation* — peak cell density 14 against a limit of 8 — because retreating to full thermal compliance costs more in wiring than it recovers in heat. Optimal designs operate at the constraint boundary, which is the formal version of "chips boost until they throttle." Second, the cost curve is asymmetric: over-dense designs incur steep but bounded penalties, while over-sparse designs incur unbounded wire growth — rationalizing the industry posture of erring dense and managing heat actively (throttling, dark silicon) rather than spreading out.

## 5. Model Assumptions Are the Result: The 3D Stacking Reversal (Experiments 9/9B)

The most epistemically important result of the session. Folding the design into 1/2/4/8 stacked layers (footprint shrinking as √(1/L), constant total area) was evaluated under two thermal models:

- **Per-layer model** (each layer's 2D density penalized independently): monotone improvement with stacking; 8 layers optimal, with thermal penalty falling to *zero* — because 125 neurons/layer trivially satisfies any per-layer density limit.
- **Shared-heatsink column model** (power summed over the (x,y) column across all layers, with a stack-depth multiplier for buried-layer insulation): complete reversal. 1 layer optimal; 8 layers **55x worse** (19,633 → 1,081,743).

The physical design was identical in both cases; the conclusion inverted on one assumption about heat egress. The wire savings available from stacking this workload capped at 14% — never close to paying a realistic thermal bill for homogeneous logic-on-logic. This reproduces, in miniature, why the 3D products that shipped at scale are heterogeneous (HBM, V-Cache: cool die on hot die) while logic-on-logic remains largely unshipped.

**Experiment 10** made the heterogeneous case explicit. Pairing each compute node with a memory node (at 0.1 relative power) under three placements:

| Configuration | Access energy | Thermal | Total |
|---|---|---|---|
| Memory beside (separate region) | 2,025,000 | 3,450 | 2,028,450 |
| Memory interleaved (in-plane) | 645 | 5,342 | 5,987 |
| Memory stacked (one layer up) | 3,000 | 8,588 | 11,588 |

The separated ("von Neumann") configuration loses by **~350x** — a directional quantification of why memory movement dominates accelerator power. Interleaved narrowly beat stacked, but the comparison is not fully fair: the interleaved layout was not charged the footprint expansion (and consequent lengthening of all compute-compute wires) that in-plane memory actually costs. Both winners correspond to shipping technology families — compute-in-memory and HBM respectively — and the model correctly identifies the loser.

## 6. Global Connectivity Is Existential, and Structure Beats Randomness (Experiments 5–6)

The r=8, k=10 clustered optimum from §4 harbored a defect invisible to energy metrics: it was **five disconnected components**. BFS from random sources found a mean of 587/1,000 nodes unreachable. Purely local connectivity does not yield a degraded communication fabric; it yields no fabric.

Adding as few as 5 random inter-cluster edges (0.05% edge-count increase) restored full connectivity. But random long links are expensive under d² costing (+44% total energy for 5 links). Replacing them with **hub-routed backbones** — one router per cluster (centroid-nearest node), connected in star/ring/mesh patterns — dominated random placement at every price point:

| Backbone | Links | Energy vs. base | Mean hops |
|---|---|---|---|
| Star | 4 | +26.8% | 9.17 |
| Ring | 5 | +42.3% | 9.04 |
| Mesh | 10 | +77.0% | 8.72 |
| 5 random | 5 | +44.2% | 10.98 |
| 25 random | 25 | +223% | 7.41 |

Ring strictly dominates 5-random at equal link count; mesh at 10 links approaches 25-random hop performance at one-third the energy. Deliberate topology is worth roughly 2–3x over unstructured global wiring — consistent with the fact that no production NoC uses random long wires.

## 7. Congestion, Gateway Provisioning, and Bottleneck Migration (Experiments 7–8)

Static metrics flatter the star. Routing 2,000 random messages exposed it: the star hub transited **71% of all traffic** (1,415/2,000 messages; 66x mean node load) — a latency queue, thermal hotspot, and single point of failure co-located. Ring and mesh halved worst-case load (~700; 33–35x), with mesh only marginally ahead of ring — because the binding constraint had become the one-gateway-per-cluster funnel, not the backbone pattern.

Sweeping routers-per-cluster (spread within each cluster by farthest-point selection; full inter-cluster router mesh):

| Routers/cluster | Links | Wire energy | Hops | Max router load | Max non-router load |
|---|---|---|---|---|---|
| 1 | 10 | 28,853 | 9.23 | 670 | 199 |
| 2 | 40 | 65,046 | 7.77 | 453 | 133 |
| 3 | 90 | 135,510 | 6.63 | 366 | 117 |
| 4 | 160 | 250,178 | 5.83 | 266 | **121** |

Two findings. First, **bottleneck migration captured directly**: non-router load floors at ~120 between 3 and 4 routers/cluster — congestion relocates from gateways to the local links feeding them, and further gateway provisioning cannot help. Second, backbone wire cost grows as R² (15.5x the entire local fabric at R=4) against sub-linear returns. The knee is at **2 routers/cluster**: ~30% improvement on every metric at 2.3x wire cost, before either the migration floor or the quadratic cost bites. The correspondence to multi-ported chiplet interfaces and parallel thalamic relay nuclei is noted without ceremony.

## 8. Dynamics: Excitation Alone Is Pathological (Experiment 11)

Running LIF dynamics on the final fabric (r=8 clusters, 2 routers/cluster, mesh backbone, flat die) with purely excitatory synapses (w=0.30, fan-in ≥10): a 5-step stimulus to one cluster propagated correctly through the router backbone (ignition of neighboring clusters within 2–6 steps — the fabric works as a signaling substrate) and then **saturated permanently**. Total activity: 26,880 spikes in 100 steps against a refractory-limited ceiling of ~33,000 — the network locked at ~80% of physical maximum. With fan-in × weight = 3.0 ≫ threshold, ignition is self-sustaining once any neighborhood exceeds ~40% activation; the refractory rotation guarantees a standing pool of recruitable nodes. In energy terms, this is worst-case switching activity on every node indefinitely — the dynamical equivalent of the thermal violations of §4, and a faithful miniature of runaway excitation in biological tissue.

Introducing a 20% inhibitory subpopulation (outgoing weights −0.60, i.e., 2x magnitude, loosely matching cortical E/I ratios) reduced total activity 73% (to 7,211 spikes, ~27% of ceiling) and qualitatively changed the regime: stimulus response, near-quiescence, then **sustained population oscillations** — per-cluster waxing/waning cycles of excitation building until inhibition overtakes it. The network neither dies nor saturates; it rhythms. This is the expected signature of E/I-balanced recurrent networks and the dynamical argument for why "machinery that makes things stop" (inhibition; clock/power gating) is not overhead but a precondition of computation.

## 9. Synthesis: A Derived Design Rulebook

Read as a single arc, the experiments compose into an ordered set of design principles, each empirically forced by the failure mode of the previous one:

1. **Co-locate communicating computation.** Placement alone spans ~79x in communication energy at fixed netlist.
2. **Densify to the thermal boundary, not past it** — and expect the optimum to sit *on* the constraint, tolerating marginal violations where the wiring gradient exceeds the thermal gradient.
3. **Pure locality fragments the system.** Sparse global links (≪1% of edges) are existential, not an optimization.
4. **Never scatter global wires; structure them.** Hub-routed backbones dominate random long links 2–3x per unit cost.
5. **Never provision a single gateway** (71% traffic capture); provision approximately two per cluster, beyond which the bottleneck migrates into the local fabric while backbone cost grows quadratically.
6. **Stack heterogeneously or not at all.** Under shared-heatsink physics, homogeneous logic stacking is thermally dominated (55x); low-power-on-high-power stacking converts a 350x memory-access penalty into a small constant.
7. **Provision inhibition/gating as a first-class resource.** An excitation-only fabric with working interconnect saturates into a pathological attractor; ~20% inhibitory capacity buys stable, structured, low-duty-cycle dynamics.

The composite — dense local clusters, thermally bounded, bridged by a small number of multi-ported structured gateways, with heterogeneous vertical integration and explicit stopping machinery — is recognizably both cortical architecture (columns, white matter, parallel thalamic relays, E/I balance) and contemporary accelerator architecture (chiplets, NoC fabrics, multi-port D2D interfaces, HBM, power gating). That two independent optimization processes (evolution; the semiconductor industry) converged here, and that a laptop-scale model recovers the same structure in an evening, suggests these are properties of the underlying constraint geometry rather than artifacts of either lineage.

## 10. What This Session Actually Demonstrates

The technical results are, individually, well known. Three meta-results are the genuine takeaways:

**1. The design space is legible at toy scale.** Every trade-off above emerged clearly at n=1,000 with O(n²) NumPy and no specialized tooling. The *structure* of placement/thermal/congestion trade-offs — U-curves, constraint-riding optima, bottleneck migration, phase transitions in dynamics — is scale-invariant enough to build correct intuition cheaply before engaging production EDA flows.

**2. Model interrogation is the core skill.** The 3D stacking reversal (§5) is the session's most important single datum: an identical design scored best and worst under two defensible-sounding thermal models. The practitioner's job is not running simulations but auditing what simulations silently assume — and the fastest way to learn that skill is to be personally burned by it on a Friday night, at zero cost.

**3. The collaboration model works at the architecture level.** This session was conducted by a domain non-specialist (construction operations background, self-taught full-stack developer) working conversationally with an LLM: the human supplying direction, skepticism, and the originating physical intuition (that spatial arrangement of computation is a first-class variable); the model supplying implementation, domain framing, and prediction-making — with a running record of predictions that were *wrong* (energy cost of random highways; 2-layer stacking optimum; "return to quiet" vs. oscillation) and corrected by the data. The prerequisite qualifications for contributing to architecture exploration may be shifting from credential-gated knowledge toward the ability to pose falsifiable questions and audit models — which is a different, and more widely distributed, skill.

## 11. Open Directions

Ranked by expected insight per unit effort on the existing testbed:

- **Adaptive/congestion-aware routing** vs. shortest-path: how much of the gateway provisioning of §7 can software recover? (The classic hardware-vs-routing trade.)
- **Traffic locality distributions:** replace uniform random pairs with power-law/locality-biased traffic (realistic for both cortex and NN inference) and re-derive the gateway and backbone optima.
- **Quiescence and criticality:** tune noise floor and E/I balance toward the critical regime; measure dynamic range and information transmission as a function of distance from criticality.
- **Learning on the fabric:** local plasticity rules (STDP-like) on the clustered topology — does the physical structure bias what is learnable?
- **Vectorization and scale:** the current O(n²) loops cap practical size; a sparse-matrix rewrite reaches n=10⁵ on the same laptop, enough to study scaling laws of the §9 rulebook.
- **Heterogeneous placement with honest area accounting:** charge the interleaved-memory configuration of §5 its true footprint cost and settle the interleaved-vs-stacked comparison properly.

---

*All code, notebooks, and experimental records are version-controlled in the project repository (`neuromorphic-sandbox`, 11 experiments, single session, August 8, 2026).*
