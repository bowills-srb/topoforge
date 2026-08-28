# Placement Determines Learnability: Local Connectivity as a First-Class Design Variable for Plastic Neuromorphic Hardware

**Bo Wills** (Independent Researcher)

*Provisional patent filed: USPTO #64/134,008 (August 14, 2026)*

---

## Abstract

Neuromorphic mapping tools place neurons on physical cores to minimize
communication energy and latency during frozen inference. We show that
on hardware with active structural plasticity — where connectivity
continues to change after deployment — the same placement decision
also determines *what the network can learn*, independent of its
effect on energy. Across simulated substrates from N=900 to N=5,000,
placing functionally distinct neuron populations in contiguous,
wire-length-optimal blocks ("segregated" placement) produces a
3.8-4.0x learning penalty relative to placement that interleaves
those populations, at *identical* energy cost during frozen operation
(bit-identical, since the conditions differ only in which type sits at
which position) and a 1.4% post-plasticity energy premium.
We show this effect survives three independent stress tests: removal
of physical geometry entirely (the effect persists at 2.81x under
pure graph-locality constraints, indicating the mechanism is local-
versus-global connectivity rather than spatial embedding specifically);
replication on a network sized to a published hardware architecture's
own proportions; and, most consequentially, replication on real
spike-encoded human speech (Spiking Heidelberg Digits), where an
adjacency-matched, twelve-seed comparison shows a segregated-placement
learning penalty of roughly 3.6x-7.3x depending on training-exposure
regime (7.25x, 95% CI [5.05x, 11.48x], at the validated regime;
Welch p = 3.2×10⁻¹⁴). The penalty is not confined to a
baseline of our own construction: a from-scratch reimplementation of
a published mapping tool (SpiNeMap), given the connectivity graph it
actually has at map time — one that cannot contain associations
plasticity has yet to build — produces a placement that learns 3.3x
less than interleaved and sits only 8% of the way from our segregated
baseline toward interleaved. The same optimizer becomes harmless when
the to-be-learned associations are declared to it up front, which
locates the problem in the graph a mapper is given rather than in
wire-length optimization as such. Sweeping the fabric's core pitch
against the plasticity radius shows the penalty saturating to its full
magnitude beyond a pitch of roughly 1.5 radii, while an
association-aware mapper is unharmed at every density tested and, on
sparse fabrics, outperforms uniform interleaving by 1.46x. We report effect
sizes, statistical tests,
and — in a dedicated Limitations section — the specific respects in
which this evidence should and should not be trusted at its current
stage.

---

## 1. Introduction

Every published neuromorphic mapping tool we are aware of — SpiNeMap,
NEUTRAMS, and comparable systems — optimizes neuron-to-core placement
for one objective: minimizing the energy and latency cost of
communication during inference on a network whose weights are frozen
after training. This objective is well-motivated for the hardware
generation these tools target. It is not well-motivated for a newer
class of neuromorphic hardware in which structural plasticity —
physical rewiring of which neurons communicate with which, not merely
adjustment of connection weights — continues after deployment.

On such hardware, placement is not just a communication-cost decision.
It is a decision about which pairs of neurons can ever come to
influence each other's learning at all, because structural plasticity
rules are necessarily local: a rule that decides whether to grow or
prune a connection can only evaluate candidates within some bounded
neighborhood, whether that neighborhood is defined by physical
distance, graph-hop count, or another locality measure. If two
functionally related neuron populations are placed outside each
other's reach under whatever locality measure governs plasticity, no
local rule can ever discover the relationship between them —
regardless of how much data the network subsequently observes.

This paper's central claim: **placement determines learnability, not
merely energy**, and the magnitude of this effect is large enough
that we believe it constitutes an overlooked design variable in
current neuromorphic hardware-mapping practice.

### 1.1 Why has this not been previously reported?

The most likely explanation is straightforward. The neuromorphic
mapping literature has focused almost exclusively on frozen,
pre-trained inference workloads, in which connectivity does not
change after deployment and a placement-learning interaction cannot
arise by construction. As on-chip structural plasticity becomes more
common in newer hardware generations, this interaction becomes newly
relevant rather than newly discovered.

---

## 2. Related Work

**Communication-aware mapping.** SpiNeMap and comparable tools solve
placement as an optimization problem over communication cost, given a
fixed, pre-trained network topology. NEUTRAMS extends this to
account for hardware resource constraints. Both frameworks assume the
mapped network's connectivity is static after deployment; neither
evaluates placement's effect on post-deployment learning, because
their target hardware does not support it.

**Neuromorphic hardware with structural plasticity.** Recent hardware
generations, including academic and industrial neuromorphic chips,
have begun to expose structural plasticity as a first-class feature
rather than a purely software-simulated mechanism. This shift is
precisely what creates the placement-learning interaction studied
here; on hardware where connectivity is fixed after fabrication or
after a training phase, the interaction cannot manifest.

**Interleaving effects in learning, more broadly.** The finding that
interleaved practice or interleaved exposure to different categories
of information produces more durable learning than blocked exposure
is well established in the cognitive-science literature on the
spacing and interleaving effects. We note the resonance between that
literature and the present findings as a connection worth further
investigation, while being explicit that we have not verified the
specific citations for that body of work to the standard we hold the
neuromorphic-hardware citations above; a reader pursuing this
connection should consult the interleaving-effect literature directly
rather than treat this paper as an authoritative citation of it.

**Self-powered analogue neuromorphic sensing.** Kim, Zhao, Sud, Xu,
Zhao, Liao, Zhang, Midya, Qiu, and Yang (2026) demonstrate a
memristor-based analogue system performing multimodal sensing,
spike-encoding, and hetero-synaptic learning in a single self-powered
circuit, with a measured Gaussian-shaped dependence of synaptic
strengthening on inter-channel timing coincidence. We use this
measured functional form, rather than our own invented learning rule,
in one of the generalization tests reported in Section 4.4, to check
whether our core finding depends on the specific plasticity mechanism
we designed or generalizes to a mechanism independently measured in
different hardware (Kim et al., *Nature Sensors* 1, 535-544, 2026,
DOI: 10.1038/s44460-026-00067-7).

**Biological network optimization under local rules.** Tero et al.
(2010) show that the slime mold *Physarum polycephalum*, using
purely local reinforce-or-dissolve rules with no central
coordination, constructs network topologies closely resembling the
Tokyo rail system when food sources are placed at station locations.
Their mathematical model of tube conductance dynamics is structurally
analogous to the value-gated structural plasticity rule used
throughout this work (Tero, Takagaki, Saigusa, Ito, Bebber, Fricker,
Yumiki, Kobayashi, and Nakagaki, *Science* 327, 439-442, 2010).

---

## 3. Architecture and Methods

### 3.1 Substrate

Neurons are placed at fixed or migratable coordinates on a
two-dimensional substrate (or, in the geometry-free variant described
in Section 4.3, assigned to abstract graph communities with no
coordinates at all). Each neuron follows leaky integrate-and-fire
(LIF) dynamics with a refractory period. Connectivity is initialized
randomly and evolves under structural plasticity: connections are
periodically pruned based on accumulated value and replaced by
candidates drawn from a locally-scored pool.

### 3.2 Structural plasticity rule

Three decaying pairwise state variables are tracked per candidate
connection: **C**, a fast-decaying co-firing correlation; **E**, an
eligibility trace; and **V**, a slow-decaying value signal updated by
reward-prediction error (RPE) and gated to only accrue for pairs
within a bounded locality (a fixed physical radius in the spatial
condition, or graph-community co-membership in the geometry-free
condition). All three decay lazily — `value(t) = stored *
decay^(t - last_touched)` — which keeps the implementation tractable
at scale without ever materializing a dense N x N state matrix.

### 3.3 Placement strategies compared

- **Segregated**: neurons of each functional type occupy a
  contiguous region (the output our wire-length-minimizing heuristic
  converges toward; a real tool need not — see Section 6).
- **Interleaved**: functional types are spatially mixed.
- **Random**: type assignment is independent of position (a control).

We are explicit that our "segregated" condition is our own
implementation of a wire-length-minimizing heuristic rather than a
published tool's output. Section 4.6 closes that gap directly by
reimplementing SpiNeMap's published algorithm and running it through
the same benchmark; we have not benchmarked against NEUTRAMS or other
tools' actual output, and the scope of what Section 4.6 does and does
not settle is discussed in Section 6.

### 3.4 Statistical methodology

**Computational reproducibility.** The rewiring step selects which
connections to replace by sorting a score array in which most entries
are exactly zero, using NumPy's default (unstable) sort. Which of the
tied-at-zero connections gets replaced therefore depends on sort
implementation details, and those changed between NumPy 1.26.4 and
2.5.1: on the benchmark of Section 4.1, seed 0, the two versions give
2955 and 2979 for interleaved placement and 682 and 724 for
segregated — a 6.2% difference on the segregated condition, and a
headline ratio of 4.33x versus 4.03x. Forcing a stable sort makes the
two versions agree exactly (2978 and 702, ratio 4.24x), confirming
tie-breaking as the sole cause. We report this rather than quietly
pinning it because it bounds how precisely any single magnitude in
this paper should be read. Every contrast we report is computed
within a single environment, so the comparisons are unaffected; but a
reader reproducing an absolute number on a different NumPy build
should expect agreement to a few percent, not to the digit. Directions
and orderings are robust to this; third significant figures are not.


Learning-quality differences between placement conditions are tested
with Welch's two-sample t-test (unequal variances assumed), reporting
t-statistic, degrees of freedom, p-value, and Cohen's d. Where
effect sizes are unusually large (d > 10, well beyond the
conventional "huge" threshold of d ~ 2.0), we flag this explicitly:
such magnitudes are a signature of low between-seed variance in a
fully controlled synthetic simulation, not a claim about the
magnitude of variability expected on physical hardware or real
workloads.

---

## 4. Results

### 4.1 The Placement-Learning Benchmark (PLB)

At N=900, ten seeds per condition: segregated placement achieves a
mean learning-quality score of 731.8 +/- 6.0; interleaved placement
achieves 2952.9 +/- 29.1 — a 4.04x ratio. Welch's t-test: t(18) =
236.7, p = 1.1x10^-19, Cohen's d = 105.9. At N=5,000: segregated
4032.8 +/- 26.8, interleaved 15349.6 +/- 216.7 — a 3.81x ratio; t(18) =
163.9, p = 2.3x10^-17, d = 73.3. The effect is consistent across a
5.5x increase in scale. As noted in Section 3.4, the very large
Cohen's d values reflect the low noise of a fully controlled
simulation and should not be read as a prediction of effect
magnitude on physical hardware.

During frozen (non-plastic) operation, energy cost differs
negligibly between placement strategies — the entire learning penalty
manifests only when plasticity is active, consistent with the
proposed mechanism (a communication-cost objective and a
learnability objective are simply different objectives, and
optimizing one does not incidentally optimize the other).

### 4.2 Persistence and detection findings (brief)

Beyond the placement-learning result, this substrate exhibits two
further properties relevant to hardware applications, reported in
full in the project's supplementary experimental record: learned
structure resists erosion from parameter perturbation in a way that
is not explained by any single tunable constant (decay rate,
correlation weight, or turnover rate individually), suggesting
persistence is a property of the value hierarchy rather than of any
one parameter; and spatially coherent anomalies produce
substantially more local correlation structure than scattered noise
(a 4.7x ratio in one benchmark), suggesting the substrate's own
geometry can perform anomaly discrimination without a dedicated
detection algorithm. These findings are secondary to the placement
result and are described here only to indicate the substrate's
broader behavior; they are not the focus of the statistical and
replication work in the remainder of this paper.

### 4.3 Generalization test 1: removing geometry entirely

The placement-learning result is compatible with two different
explanations: that spatial embedding specifically enables the
effect, or that any local-versus-global connectivity constraint
would produce it, with spatial embedding simply being the physically
realizable case. To distinguish these, we replaced physical
coordinates with pure graph-community membership — no coordinates,
no distance computation anywhere in the implementation — and
re-ran the segregated-versus-interleaved comparison with the
identical two-cluster reward structure, ten seeds per condition.

Segregated: 726 +/- 9. Interleaved: 2037 +/- 18. Ratio: 2.81x — smaller
than the 4.04x observed with spatial embedding at the same scale, but
clearly non-trivial and highly significant. We conclude the mechanism
is local-versus-global connectivity generically; spatial embedding is
a sufficient but not necessary implementation of locality, and adds
an additional effect beyond generic locality — plausibly because
physical geometry also constrains wire cost, creating a further
selection pressure that abstract graph locality does not. We revise
our framing accordingly: the operative variable is *local connectivity
structure*, of which physical placement is the practically relevant,
hardware-realizable instance.

### 4.4 Generalization test 2: an externally measured learning rule

All results above use a reward-prediction-error learning rule we
designed. To test whether the placement effect depends on the
specific rule rather than on placement itself, we replaced it with
the Gaussian inter-channel-timing-dependent plasticity (ITDP) curve
measured by Kim et al. (2026) in physical memristor hardware (Section
2), with two channels required to co-fire within a coincidence
window for fusion to accrue.

Across five seeds, placement beyond the local deposit radius produced
exactly zero fusion mass in every trial (0.00 +/- 0.00) — a complete,
mechanism-level confirmation that locality gates learning under this
externally sourced rule, not only under our own. A secondary,
unresolved finding from the same experimental arc — an inverted-U
relationship between timing-jitter and learning quality, in which
moderate jitter outperformed both perfect synchrony and high jitter —
survived three independent mechanism hypotheses being tested and
individually refuted (targeting diversity, refractory-period
resonance, and candidate-pool exhaustion were each ruled out by
direct measurement). We report this honestly as solid, reproducible,
and currently unexplained, rather than omit it or force a
premature explanation.

### 4.5 Generalization test 3: real spike-encoded speech data

All results above use synthetic, hand-designed stimulus patterns on a
fixed schedule. We consider this the single most important limitation
to address, since it leaves open whether the effect is an artifact of
synthetic pattern structure. We replicated the placement comparison
using the Spiking Heidelberg Digits (SHD) dataset (Cramer, Stradmann,
Schemmel, and Zenke, *IEEE TNNLS*, 2022) — real spike-encoded human
speech, 700 raw channels pooled to 140, real recording durations,
real inter-speaker and inter-utterance variability.

A direct, naive replication of the synthetic experimental design
produced misleading results for two identifiable, mechanistic
reasons, both instructive and reported for transparency rather than
omitted: (1) the value-decay rate, tuned for synthetic lives roughly
800-1,200 steps long, decays to numerical insignificance over a real
life approximately 8,600 steps long, producing spuriously exact
zero results that reflected a decay-timescale mismatch rather than
absent learning; (2) an initial placement geometry that separated
segregated-condition clusters far enough apart also produced a
57-fold difference in raw input-output adjacency between conditions,
so an initial large ratio (over 1,000x) reflected mismatched
opportunity rather than a difference in learning arising from
arrangement alone.

Correcting both issues — matching the decay rate to the life length, and placing both conditions in matched geometry (same disc, same neuron density, differing only in whether functional types occupy contiguous wedges or are shuffled) — yields an adjacency-matched comparison. At the validated life-length/decay regime (approximately 8,600 steps, V retention 0.651), segregated placement produces a mean bridge mass of 1161 ± 822 across twelve seeds; interleaved placement produces 8423 ± 1089. Ratio: 7.25x, bootstrap 95% CI [5.05x, 11.48x]. Welch's t-test: p = 3.2×10⁻¹⁴; a non-parametric Wilcoxon signed-rank test, which does not assume normality, independently confirms significance (p = 4.9×10⁻⁴); Cohen's d = 7.53.

To isolate whether this magnitude depends on the number of distinct real samples used, we compared two configurations at the identical validated regime: four samples per class repeated across five exposure epochs (the original configuration, reproducing at 7.22x on independent re-verification) against twenty distinct samples per class seen once each. The two configurations produce statistically indistinguishable ratios (7.22x versus 7.25x), directly showing that sample diversity is not the driver of the effect — a five-fold increase in distinct real examples changes nothing.

The magnitude does depend on the training-exposure regime itself: extending exposure to two epochs (approximately 17,200 steps, V retention 0.42) compresses the ratio to 3.62x, 95% CI [2.70x, 5.42x] (still highly significant: p = 1.3×10⁻⁷, Wilcoxon p confirms, d = 4.9). We report both regimes rather than selecting one: the honest range for this effect on real speech data is approximately 3.6x to 7.3x depending on training exposure, with both endpoints comfortably bracketing the 3.8-4.0x observed on synthetic data in Section 4.1.

We report one further honest caveat, quantified more precisely now than in earlier drafts of this work. Segregated-condition variance remains high (relative standard deviation 60-70%) regardless of seed count — adding more seeds revealed this variance rather than shrinking it, and two of twelve seeds in the longer-exposure regime produced bridge mass of essentially zero. We read this as a real, reportable property of segregated placement rather than measurement noise: segregation does not merely reduce average learning, it makes learning outcomes substantially less reliable across runs. A hardware designer choosing between placement strategies should weigh this reliability difference alongside the mean effect.

### 4.6 Generalization test 4: a real mapping tool (SpiNeMap)

The preceding three tests stress the effect itself; this one addresses a different objection — that our "segregated" condition is a baseline we constructed rather than the output of a mapping tool anyone deploys. SpiNeMap (Balaji et al., arXiv:1909.01843) releases no code, so we implemented its published two-step algorithm from scratch: **SpiNeCluster**, a Kernighan-Lin partitioning that minimizes inter-cluster (global) synapse count with clusters sized to a core's capacity; and **SpiNePlacer**, a particle-swarm optimization that assigns clusters to physical cores to minimize communication cost. We then ran SpiNeMap's placement through the identical Placement-Learning Benchmark, ten seeds, alongside our three hand-built conditions.

Because a plastic network's cross-type associations do not exist as synapses until they are learned, the connectivity graph a mapper is given matters, and we tested both bounds: a **population** graph (intra-type synapses only — the structure realistically known before plasticity runs) and a **functional** graph (additionally including the to-be-learned association synapses — an upper bound rarely available in practice).

*Correction note.* An earlier version of this section reported that both SpiNeMap variants landed near interleaved placement. That result was wrong, and the error is instructive enough to state plainly. Our partitioner was validated on type-*shuffled* neuron labels, where it reaches the exact theoretical minimum cut; the placement path calls it with type-*sorted* labels, and on that input the greedy stalled 6.4% above the optimum — at essentially the random-partition cut (KL/random = 0.9965), producing clusters of mean type-purity 0.38 with 1 of 60 cores pure. The clustering step was, in effect, not running, so what we had benchmarked was a near-random placement rather than SpiNeMap. Two properties of this graph made the failure easy to miss: the audit used an input distribution the deployed path never sees, and the cut has narrow dynamic range (even the exact optimum lies only 6% below random), so the absolute cut value looked plausible on its own. The partitioner now searches swaps exhaustively within each cluster pair and randomizes node order internally; it attains the exact optimum on both orderings, both graph modes, and every seed, with balanced clusters and type-purity 1.000 on the population graph. The audit now runs the deployed ordering and asserts a zero gap against the theoretical optimum. All numbers below are from the corrected implementation.

A structural audit before any learning (Table 3) shows what the corrected mapper actually does. Minimizing inter-cluster synapse count on the population graph makes cores type-*pure* — that is precisely the optimal partition when the only edges are intra-type — so a neuron in a core's interior has no associated partner within the plasticity radius at all, and only neurons near a core boundary do. On the functional graph, where the associations are declared, the optimal partition instead packs associated types into the same core, and every neuron has partners in reach.

**Table 3.** Fraction of source-type neurons with an associated-partner neuron within the plasticity radius, before learning (N=900).

| Placement | 0→3 | 1→4 |
|---|---|---|
| Random | 100.0% | 100.0% |
| Segregated (ours) | 0.0% | 0.0% |
| Interleaved (ours) | 100.0% | 100.0% |
| SpiNeMap — population graph | 71.1% | 66.7% |
| SpiNeMap — functional graph | 100.0% | 100.0% |

Learning quality follows that structure, and the two graph conditions fall on opposite sides of the benchmark (Table 4). On the population graph — the honest map-time case — SpiNeMap lands 7.7% of the way from our segregated baseline toward interleaved: it learns 3.27x less than interleaved placement (903.8 ± 61.2 vs 2952.9 ± 29.1; Welch t(12.0) = −95.6, p = 9.9×10⁻²⁰, d = 42.8), statistically distinguishable from our hand-rolled segregated condition but only 1.24x above it. On the functional graph it lands at 101.2% — indistinguishable from interleaved for practical purposes (2980.6 ± 16.6, a 0.9% difference that is nominally significant at p = 0.02 only because of the simulation's very low noise).

**Table 4.** Learning quality (taught mass after the plastic phase), mean ± s.d. over ten seeds, N=900. Position is on the segregated → interleaved axis, where 0% is our segregated baseline and 100% is interleaved.

| Placement | Taught mass | vs. segregated | Position |
|---|---|---|---|
| Segregated (ours) | 731.8 ± 6.0 | — | 0% |
| SpiNeMap — population graph | 903.8 ± 61.2 | +23% | 7.7% |
| Random | 2733.2 ± 25.9 | +274% | 90.1% |
| Interleaved (ours) | 2952.9 ± 29.1 | +304% | 100% |
| SpiNeMap — functional graph | 2980.6 ± 16.6 | +307% | 101.2% |

The conclusion is the opposite of what we previously reported, and it strengthens rather than weakens the paper's central claim. A published, communication-minimizing mapping tool, given the connectivity graph it actually has at map time, **does** produce the pathological placement: our hand-rolled segregated baseline is not a strawman but a close stand-in for what such a tool emits on a plastic substrate. What rescues SpiNeMap is not its objective but its input — declaring the to-be-learned associations up front moves it the entire way to interleaved. This is a sharper and more actionable statement of the problem than "wire-length optimization is harmful": the harm comes from optimizing a communication objective over a graph that omits the associations plasticity has yet to build, and the same optimizer becomes harmless the moment those associations are in the graph it is given. Since a plastic system's whole purpose is to learn associations that were not known at map time, the population-graph case is the one that matters in practice.

(One internal inconsistency is also corrected here: an earlier Table 4 reported 686 ± 5 and 2926 ± 26 for the segregated and interleaved conditions, which disagreed with the same conditions in Section 4.1. The values above were freshly reproduced and match Section 4.1 exactly.)

### 4.7 Dose-response and mediation: the reachable correlated fraction

Every result to this point is a two-point contrast (segregated versus interleaved). To test whether placement acts *causally* through a single measurable quantity rather than through some property peculiar to one geometry, we interpolated continuously between the two conditions. A knob α scatters each neuron from its segregated position to a random disc position with probability α, so α=0 reproduces the segregated placement and α=1 the interleaved one. Before any learning we measure a placement-only mediator — the mean number of input neurons within the plasticity radius of each output neuron, the structural "reach" that permits input→output bridges to form at all. We swept α on the real-data task at both K=2 and K=4 (eight seeds each, the same 8,600-step regime).

Two predictions were registered before running. First, a **dose-response**: total learned bridge mass should increase monotonically with α. It does — Spearman ρ = 1.00 at K=4 and ρ = 0.89 at K=2 (the K=2 curve has a mild non-monotonic wobble in the low-α region, within seed noise), with learning rising 3.2× (K=4) and 3.9× (K=2) from α=0 to α=1. Second, **mediation**: if reach is the variable placement acts through, normalized learning should be a single function of reach, with the K=2 and K=4 sweeps falling on one curve rather than two. This holds — pooling both K, reach predicts normalized learning with Pearson R² = 0.90 (Spearman ρ = 0.93), and the two sweeps *interleave* along the reach axis (Table 5) rather than separating.

**Table 5.** Dose-response pooled across K and α, sorted by the pre-learning reach metric. Learning is normalized to each K's interleaved (α=1) ceiling; K=2 and K=4 rows interleave rather than forming two curves.

| K | α | mean reach | normalized learning |
|---|---|---|---|
| 2 | 0.00 | 24.1 | 0.26 |
| 2 | 0.15 | 25.8 | 0.50 |
| 4 | 0.00 | 26.9 | 0.32 |
| 2 | 0.30 | 28.4 | 0.43 |
| 2 | 0.45 | 31.1 | 0.50 |
| 2 | 0.60 | 33.8 | 0.71 |
| 4 | 0.33 | 34.0 | 0.68 |
| 2 | 0.80 | 34.6 | 0.90 |
| 4 | 0.67 | 37.0 | 0.90 |
| 2 | 1.00 | 37.3 | 1.00 |
| 4 | 1.00 | 38.7 | 1.00 |

We read this as evidence that placement affects learning through a single mediating quantity — the fraction of correlated pairs the local plasticity rule can physically reach — rather than through anything specific to a given geometry, task, or class count. (Section 4.8 revisits how that quantity should be *measured*: across a wider set of placement families, the mean-count form used here fails, and a coverage form succeeds. The mediation claim survives; the metric is refined.) On one curve it accounts for why removing geometry still leaves an effect (Section 4.3), why a communication-minimizing mapper that compacts populations lands near interleaved (Section 4.6), and why the penalty's magnitude shifts with regime (Section 4.5): each changes reach. One honest limit remains at this point: in the sweep, reach is *measured*, not manipulated while holding all other geometric properties fixed, so the evidence so far is correlational. We resolve that next with a controlled dissociation.

To turn the mediation from correlational to causal, we manipulated reach and the most obvious confound — whether output classes are spatially clustered or mixed — as two independent factors. Reach was set by rigidly translating the output population a distance from the input population, which changes input–output distances while preserving input–input and output–output structure exactly. Class arrangement was set by assigning class labels to a *fixed* set of output positions either in contiguous spatial blocks or shuffled: a pure relabeling that leaves the geometry, and therefore reach, byte-identical between the two (confirmed to machine precision at every separation). Across eight seeds and four separations (mean reach 71 → 57 → 23 → 2.6 input neighbours per output neuron), learning tracked reach monotonically and steeply — a 77-fold change from highest to lowest reach, Spearman ρ = 1.00 at fixed arrangement, pooled R² = 0.98 — while class arrangement had no significant effect at any matched reach (paired *p* = 0.09–0.28; Table 6). Reach is therefore not merely correlated with learning but the variable placement acts *through*; the spatial clustering of classes that nominally distinguishes "segregated" from "interleaved" is, on its own and with reach held fixed, inert.

**Table 6.** Controlled dissociation. At each separation, class arrangement (clustered vs mixed) is a pure relabeling of identical output positions, so reach is byte-identical across the two columns; only the input–output separation changes reach. Total bridge mass, mean ± s.d. over eight seeds.

| separation | mean reach | clustered | mixed | arrangement *p* (paired) |
|---|---|---|---|---|
| 0 | 71.0 | 5663 ± 1701 | 6346 ± 2591 | 0.11 |
| 5 | 57.0 | 4754 ± 804 | 4993 ± 561 | 0.09 |
| 10 | 22.5 | 1180 ± 914 | 819 ± 324 | 0.27 |
| 15 | 2.6 | 73 ± 59 | 83 ± 66 | 0.28 |

The arrangement comparisons are non-significant throughout, though a weak, non-significant trend favouring mixed at high reach (*p* ≈ 0.09–0.11) means we claim only that any arrangement effect is small relative to the 77-fold reach effect, not that it is exactly zero.

### 4.8 Fabric sparsity: when the penalty saturates, and what protects against it

Section 4.6 shows a real mapping tool incurring the penalty at this benchmark's geometry, where the core pitch happens to equal the plasticity radius. Since Section 4.7 identifies reach as the operative variable, the natural question is how the result moves when that coincidence is broken. We swept the fabric's core pitch from 0.5x to 3x the plasticity radius — the radius being a property of the substrate's plasticity mechanism and the pitch a property of the floorplan — holding everything else fixed. Placements are not regenerated per pitch: each strategy's placement is produced once by the same functions used in Sections 4.1 and 4.6, and each core's disc is then rigidly translated to the new pitch, which preserves within-core geometry exactly and is bit-identical to the original at ρ = 1 (both verified). Eight seeds per cell.

**Table 7.** Learning (taught mass) versus fabric sparsity ρ = core pitch / plasticity radius. Mean ± s.d. over eight seeds, N=900. The segregated baseline is included as the floor.

| ρ | interleaved | SpiNeMap — population | SpiNeMap — functional | segregated |
|---|---|---|---|---|
| 0.5 | 3029 ± 21 | 1595 ± 90 | 3017 ± 35 | 752 ± 11 |
| 1.0 | 2958 ± 30 | 904 ± 69 | 2980 ± 19 | 733 ± 6 |
| 1.25 | 2279 ± 23 | 753 ± 19 | 2982 ± 22 | 731 ± 5 |
| 1.5 | 2035 ± 18 | 733 ± 3 | 2987 ± 15 | 730 ± 5 |
| 2.0 | 2039 ± 8 | 730 ± 10 | 2974 ± 18 | 732 ± 5 |
| 3.0 | 2039 ± 8 | 730 ± 10 | 2974 ± 18 | 732 ± 5 |

Three results, one of which contradicts a prediction we registered before running.

**The penalty saturates rather than switching on.** Population-graph SpiNeMap already carries a penalty at the densest fabric (1.90x below interleaved at ρ = 0.5) and degrades monotonically until, at ρ ≥ 1.5, it is statistically indistinguishable from the segregated baseline itself (733 ± 3 vs 730 ± 5). So sparsity does not create the penalty — the map-time graph does — but sparsity determines how complete it becomes. Beyond ρ ≈ 1.5 a communication-minimizing placement built on the pre-plasticity graph has lost the ability to form cross-type associations entirely, and no amount of further spreading changes that.

**Declaring the associations protects at every fabric density.** Functional-graph SpiNeMap is flat across the whole sweep (3017 → 2974, a 1.4% change over a 6x change in pitch). More striking, beyond ρ = 1.25 it *exceeds* interleaved placement — 2974 vs 2039, a 1.46x advantage — because it deliberately co-locates the specific populations that must associate, whereas interleaving mixes all types uniformly and therefore spends core capacity on neurons that have no reason to be adjacent. On a sparse fabric, uniform interleaving is a blunt instrument; targeted co-location of correlated populations is strictly better. This is the paper's most directly actionable result for tool builders: the fix is not to abandon communication-aware mapping but to give it the association structure.

**The interleaved control, and a prediction that failed.** We predicted interleaved placement would be flat in pitch, on the reasoning that its cross-type partners sit inside the same core disc (radius 1.5, well under the plasticity radius) and respacing cores cannot remove them. That is wrong: interleaved also draws partners from *neighbouring* cores at low ρ, and loses them, falling 1.49x from ρ = 0.5 to ρ = 1.5. What the sweep does deliver is a stronger control than the predicted flatness would have been. Interleaved learning is identical at ρ = 1.5, 2.0 and 3.0 (2035, 2039, 2039) across a 4x increase in mean squared wire distance, and its reach is likewise pinned at 3.00 across those points. The plasticity rule's rewiring score contains a distance discount, 1/(1 + 0.05d²), so a plausible alternative explanation for every result in this paper is that distance *per se* penalizes candidate pairs. That explanation predicts continued decline from ρ = 1.5 to 3.0. It does not occur, in any of the four placement families. Distance matters only through whether it puts a correlated partner inside the radius.

**Refining the mediator (exploratory).** Section 4.7 measured reach as a mean *count* of correlated partners within the radius, on a placement family where every reasonable form of that measure co-varies. This sweep breaks the tie, and the count form does not survive it: pooled across all four placement families and six pitches, count explains R² = 0.266 of the variance in learning. The failure is systematic, not noisy — population-graph SpiNeMap at ρ = 0.5 has a *higher* mean count than interleaved (36.8 vs 29.6) while learning half as much, because type-pure cores concentrate the reachable pool onto boundary neurons and leave core interiors with nothing in reach. Substituting *coverage* — the fraction of correlated neurons with at least one partner within the radius — raises this to R² = 0.832 (Spearman ρ = 0.922), and five pairs of conditions with matched count (6.3–8.1) but different coverage all separate as coverage predicts: coverage 1.00 gives ≈2980 taught, coverage 0.61 gives 904. Adding count back on top of coverage buys +0.004 R², while coverage adds +0.172 over count alone. Pool size is not irrelevant, but its effect is strongly saturating: within interleaved placement, where coverage is pinned at 1.000 at every pitch, a 9.9x reduction in mean count costs only 1.49x in learning, against roughly 4x for losing coverage altogether.

**Confirming it on a family built to discriminate.** Because coverage was formulated after seeing the data above, we ran a separate confirmatory test on three placements constructed in advance to make the rival accounts predict different orderings, with the predictions and decision rules registered before the run. All 900 neurons are placed in blobs of constant density on a lattice spaced so that every intra-blob pair lies inside the plasticity radius and every inter-blob pair outside it, so a blob's composition fixes its members' reach by construction rather than by measurement (verified: zero inter-blob neighbours, maximum intra-blob distance 2.95-4.14 against a radius of 5.0). C1 uses 60 blobs of 15; C2 uses 30 blobs of 30 with all five types mixed; C3 uses 30 blobs of 30 with the partner types concentrated into a few blobs and the remaining neurons parked in blobs whose types are not associated. C1→C2 changes partner share 2.1x at fixed count and coverage; C2→C3 changes coverage 2.4x at matched count and partner share. The design is adversarial to the coverage account: C3 has the *highest* mean count (6.25 against 6.00), so the count account predicts C3 wins outright rather than merely tying.

**Table 8.** Confirmatory test. Mean ± s.d. over eight seeds, N=900. Metrics are measured on the built placements, not assumed.

| Condition | count | frac | cover | Taught mass |
|---|---|---|---|---|
| C1 spread15 | 6.00 | 0.429 | 1.000 | 2909 ± 27 |
| C2 dilute30 | 6.00 | 0.207 | 1.000 | 2739 ± 63 |
| C3 clumped30 | 6.25 | 0.216 | 0.417 | 2082 ± 36 |

The count account is refuted: C3 carries the largest reachable pool and learns the least, significantly below both other conditions (vs C2, 1.32x, Welch p = 3.6×10⁻¹¹, d = 12.8; vs C1, 1.40x, p = 1.5×10⁻¹⁶). The coverage effect is confirmed in isolation, since C2 and C3 are matched on both count and partner share and differ only in coverage. Partner share, however, is not inert: C1 exceeds C2 by 1.06x (p = 5.0×10⁻⁵, d = 3.5) at matched count and coverage. Smaller blobs have proportionally more boundary, making C1 about 7% sparser than C2 in nearest-neighbour spacing, so we controlled for that directly by rescaling C2's blob radii to match C1's spacing at fixed count, share and coverage: learning moved 0.4% (p = 0.72), against the 6.2% C1-vs-C2 gap. The partner-share effect is therefore real rather than a density artifact.

We therefore restate the mechanism of Section 4.7 more precisely, and with one part of it now confirmed rather than merely fitted. The mean *size* of the reachable correlated pool is not the mediator — an account based on it is refuted by a placement built in advance to test it. What matters is fractional: primarily *whether* a correlated neuron has any partner the local rule can reach (a 2.4x coverage change moves learning 1.32x), and secondarily what share of its reachable neighbourhood those partners represent (a 2.1x change moves learning 1.06x). Neither quantity alone predicts magnitudes across all conditions — coverage falls 2.4x between C2 and C3 while learning falls only 1.32x, and the corresponding drop in Section 4.6 was far steeper — so we claim the ordering and the mechanism, not a calibrated functional form.

### 4.9 The energy-learnability frontier

Everything above concerns what a placement can learn. The objective a mapping tool actually optimizes is communication cost, and this paper has so far reported that side of the ledger only in passing. This section measures both together: for each placement, the summed squared wire length of the network's *realized* connections — after structural plasticity has rewired them — against what that placement learned. Eight seeds, N=900.

One structural fact makes the central comparison exact rather than statistical. In this benchmark the segregated, interleaved and random conditions place neurons at *identical coordinates* and differ only in which type label sits at each position, and the initial connectivity is seeded identically. Any energy difference between them is therefore attributable entirely to the connections plasticity built, with no geometric confound whatsoever. The two SpiNeMap conditions do differ geometrically, since clustering assigns different neurons to different cores, and are reported alongside rather than as part of that exact comparison.

**Table 9.** Communication energy (summed squared wire length over realized connections) and learning, mean over eight seeds. "Frozen" is before plasticity, "plastic" after. Sorted by learning.

| Placement | frozen E | plastic E | taught | E per unit taught |
|---|---|---|---|---|
| Segregated (ours) | 5.019×10⁶ | 2.529×10⁶ | 687 | 3680 |
| SpiNeMap — population graph | 4.983×10⁶ | 2.530×10⁶ | 861 | 2940 |
| Random | 5.019×10⁶ | 2.564×10⁶ | 2741 | 936 |
| Interleaved (ours) | 5.019×10⁶ | 2.565×10⁶ | 2935 | 874 |
| SpiNeMap — functional graph | 4.989×10⁶ | 2.493×10⁶ | 2952 | 844 |

**Inference energy is not approximately equal; it is identical.** Section 4.1 reported that frozen-phase energy "differs negligibly" between placements. It differs not at all: across all eight seeds the segregated and interleaved conditions record bit-identical frozen energy (5.01906×10⁶), which follows from their sharing coordinates and initial connectivity exactly. The claim that the learning penalty is incurred at zero inference-energy cost can be stated without hedging.

**Choosing learnability does cost communication energy, and the cost is 1.4%.** We predicted before running that interleaving would carry no post-plasticity energy penalty. That prediction is falsified: interleaved placement ends with 1.0143× the wire energy of segregated (2.5648×10⁶ vs 2.5288×10⁶, Welch *p* = 3.6×10⁻⁵). The effect is real and statistically unambiguous. It is also, in magnitude, 1.4% — bought with 4.27× the learned structure. We report the falsification rather than the framing that would have been convenient, but we do not think a 1.4% energy premium for a 4.3-fold learning gain should be described as a tradeoff in any practical sense. Per unit of learned structure the ordering inverts sharply: segregated placement costs 3680 energy units per unit taught against interleaved's 874, a factor of 4.2.

**Plasticity reduces communication energy.** In every condition, post-plasticity energy is roughly half the frozen value (≈2.5×10⁶ against ≈5.0×10⁶). This is a consequence of the rewiring rule's distance discount: connections are replaced preferentially with nearby partners, so a network that rewires ends up cheaper to communicate across than the random initial connectivity it started from, regardless of placement. Structural plasticity is not merely compatible with a communication-energy objective here; left to run, it pursues one.

**The association-aware mapper dominates outright.** SpiNeMap given the functional graph is Pareto-dominant over every other condition tested — against interleaved, 0.972× the energy at 1.006× the learning; against our segregated baseline, 0.986× the energy at 4.30× the learning; against its own population-graph counterpart, 0.985× the energy at 3.43× the learning. There is no axis on which it is worse. Combined with Section 4.8's finding that it is also the only condition immune to fabric sparsity, this is the paper's clearest practical statement: a communication-minimizing mapper that is told which populations must associate gives up nothing at all — not energy, not learnability, not robustness to floorplan — relative to any other placement we tested, including our own recommended one.

(These figures were produced under NumPy 1.26.4; see Section 3.4. All comparisons in this section are within that single environment, and the exact-equality result for frozen energy is structural rather than numerical.)

---

## 5. Discussion

Taken together, Sections 4.3-4.5 constitute three independent,
increasingly demanding tests of the same core claim, each addressing
a distinct objection a careful reader would raise: is it really about
space, or just locality (4.3); does it depend on our specific
learning rule (4.4); does it survive real data (4.5). The effect
persists through all three, with the magnitude varying by test in
ways we believe are individually explicable rather than arbitrary.
Section 4.6 addresses a fourth objection of a different kind — not
whether the effect is real but whether our segregated baseline
represents a tool anyone deploys — by running SpiNeMap's actual
algorithm through the same benchmark, and finds that it does: given
the map-time graph, the published tool lands next to our segregated
condition, and is rescued only by being told the associations in
advance. Section 4.7 then moves from *whether*
to *how*: interpolating continuously between the conditions shows the
effect is a monotonic dose-response mediated by a single placement-only
quantity — the reachable correlated fraction — onto which the K=2 and
K=4 sweeps collapse, and a controlled dissociation (varying reach while
holding class arrangement fixed by pure relabeling) confirms reach is
*causal* rather than merely correlated. This reframes the whole result
mechanistically: placement matters exactly insofar as it sets how much
of the correlated structure the local plasticity rule can physically
reach. Section 4.8 then sweeps the fabric's core pitch against the
plasticity radius, which does three things: it shows the mapping-tool
penalty saturating to the full segregated penalty beyond a pitch of
about 1.5 radii, it shows that supplying the association structure
protects at every density tested, and — by holding learning exactly
constant across a 4x change in wire distance at fixed reach — it rules
out distance itself, as opposed to reach, as the operative variable.
It also refines how reach should be measured: coverage of the
correlated population, not the mean size of the reachable pool — a
refinement then confirmed against its rivals on a placement family
built in advance to discriminate them. Section 4.9 closes the loop by
measuring the objective a mapper actually optimizes, and finds the
association-aware placement Pareto-dominant: lower communication
energy *and* higher learning than every alternative tested, including
our own recommended interleaving.

**Practical implication for hardware mapping.** A mapping tool
optimizing purely for communication energy, applied to hardware with
active structural plasticity, can be harmful to the network's ability
to learn. Section 4.9 measures both sides of that ledger rather than
asserting them. The asymmetry is stark: segregated placement buys
*exactly* zero inference-energy advantage (frozen energy is
bit-identical, not merely similar) and 1.4% lower post-plasticity
energy, in exchange for 4.3x less learned structure. Choosing
learnability is therefore not free — the 1.4% is real and
statistically unambiguous, and we had predicted it would be zero —
but the exchange rate is such that describing these as competing
objectives overstates the conflict by more than two orders of
magnitude. This
is not a hypothetical failure mode of a baseline we invented: Section
4.6 shows a published tool incurring it on the graph it actually has
at map time. What determines the outcome is the information the
optimizer is given, not the objective it minimizes — the same tool
lands at interleaved when the to-be-learned associations are declared
to it. Since a plastic substrate exists precisely to learn
associations unknown at map time, that information is normally
missing, and the risk is therefore the default case rather than the
exception. It is also checkable and fixable: the placement-only reach
statistic of Sections 4.7-4.8 is computable before deployment, so a
plasticity-aware mapper could constrain against it directly — and
Section 4.8 shows that a communication-minimizing mapper given the
association structure not only avoids the penalty at every fabric
density tested but outperforms uniform interleaving on sparse
fabrics, since it co-locates the populations that must associate
rather than mixing all types indiscriminately.

---

## 6. Limitations

*This section was written adversarially — attempting to identify the
respects in which this work's central claim should not yet be fully
trusted, rather than defending it.*

**Baseline fidelity.** Our "segregated" placement condition is our
own wire-length-minimizing heuristic, not the output of a published
mapping tool. Section 4.6 addresses this directly: we implemented
SpiNeMap's actual algorithm and found that, on the population graph
available at map time, it *does* reproduce the pathological
placement — 3.27x below interleaved, only 7.7% of the way from our
segregated baseline toward it. Our hand-rolled baseline is therefore
a reasonable stand-in for a deployed tool's output on a plastic
substrate, not a strawman, though it remains somewhat more extreme
(SpiNeMap's placement is 1.24x above it, and unlike ours leaves
roughly 70% of correlated neurons with at least one partner in
reach). The decisive variable is the graph the mapper is given: with
the to-be-learned associations declared up front, the same tool lands
at interleaved. A previous version of this section drew the opposite
conclusion from a partitioner that had silently failed on the input
the placement path uses; the error, its cause, and the fix are
documented in Section 4.6. That episode is itself a limitation worth
stating: these results rest on a single implementation, and one
load-bearing component of it was wrong for a period without any
result looking anomalous.

**Mediator form and its limits.** The coverage form of the reach
metric adopted in Section 4.8 was chosen after observing that the
pre-registered mean-count form failed on the corrected Section 4.6
result. It has since been tested confirmatorily, on three placements
built in advance so the rival accounts predict different orderings,
and it survived while the count account was refuted. Two residual
limits remain. First, coverage is not the whole story: the share of a
neuron's reachable neighbourhood that consists of partners has a
real, independent, though much smaller effect (1.06x for a 2.1x
change, density-controlled). Second, none of these forms is
quantitatively calibrated — the same relative drop in coverage
produces a 1.32x learning penalty in one setting and a 3.27x penalty
in another, so the metric predicts direction and ordering reliably
but not magnitude. A mapper could use it as a constraint; it should
not yet be used as a predictor of how much learning a given placement
will cost.

**Task and scale realism.** All synthetic results (Sections 4.1-4.4)
use hand-designed stimulus patterns at N=900-5,000. Experiment 33's
architecture proportions were drawn from a published hardware
design, but its input remained synthetic. Only Section 4.5 uses real
data, and at a substantially smaller scale (N=200) than the
synthetic benchmarks.

**Effect size interpretation.** As stated in Section 3.4, several
reported effect sizes (Cohen's d > 70 in Section 4.1) are far larger
than typical in empirical research and are a property of low-noise
synthetic simulation. The statistical significance of the *direction*
of the effect is not in question; its precise magnitude should not
be over-interpreted from simulation alone, and the real-data result
in Section 4.5 — while itself imperfect — is a better guide to
expected real-world magnitude than the synthetic benchmarks.

**Independent replication.** All results in this paper, including the
geometry-ablation and real-data generalization tests, were produced
by a single implementation and a single author. No independent
reimplementation has yet replicated any of these findings. A
collaboration is in progress to test the placement-learning
interaction on physical neuromorphic hardware (Catalyst Neuromorphic
N4), though the collaborator's current chip tapeout does not include
structural plasticity, limiting near-term hardware validation to an
inference-only comparison that cannot test the central claim
directly.

**Real-data result variance and scope.** The Section 4.5 real-data result was strengthened after initial drafting: a twelve-seed, twenty-distinct-sample-per-class replication directly tested and rejected the concern that four examples per class was too narrow a sample — five-fold more distinct data produced a statistically indistinguishable ratio. The remaining open question is not sample size but training-exposure regime: reported magnitude ranges from 3.6x to 7.3x depending on how long the network is exposed before readout, and segregated-condition outcomes remain intrinsically high-variance (60-70% relative) even at n=12. We consider the direction of this result — segregation harms real-data learning, substantially and reliably — solidly established; the precise magnitude should still be read as regime-dependent rather than as a single fixed number.

---

## 7. Conclusion

Placement determines learnability on plastic neuromorphic hardware,
not merely communication energy. This effect is robust to removing
physical geometry entirely, to substituting an externally measured
learning rule for our own, and — with important caveats disclosed
above — to replacing synthetic stimulus patterns with real human
speech. We believe this constitutes a genuine, previously
unaddressed design variable for neuromorphic hardware mapping, and we
report both the strength of the evidence for it and the specific
respects in which further validation, particularly independent
replication and physical hardware testing, remains necessary before
the finding should be considered fully established.

---

## References

1. Cramer, B., Stradmann, Y., Schemmel, J., Zenke, F. "The Heidelberg
   Spiking Data Sets for the Systematic Evaluation of Spiking Neural
   Networks." *IEEE Transactions on Neural Networks and Learning
   Systems*, 2022.

2. Kim, S.J., Zhao, R., Sud, P., Xu, Y., Zhao, J., Liao, H.-T., Zhang,
   S., Midya, R., Qiu, Q., Yang, J.J. "Self-powered analogue
   neuromorphic system for multimodal sensing, encoding and learning
   with diffusive and drift memristors." *Nature Sensors* 1, 535-544
   (2026). DOI: 10.1038/s44460-026-00067-7.

3. Tero, A., Takagaki, S., Saigusa, T., Ito, K., Bebber, D.P.,
   Fricker, M.D., Yumiki, K., Kobayashi, R., Nakagaki, T. "Rules for
   biologically inspired adaptive network design." *Science* 327,
   439-442 (2010).

4. [SpiNeMap citation — carried from prior draft; verify exact
   venue/year before submission.]

5. [NEUTRAMS citation — carried from prior draft; verify exact
   venue/year before submission.]

*Note: this reference list is incomplete relative to the full related-
work discussion in Section 2. Citations for the cognitive-science
interleaving/spacing-effect literature and for specific neuromorphic
hardware platforms referenced in Section 2 require verification
against original sources before this draft is submitted anywhere.
This is flagged explicitly rather than filled with unverified
citation details.*
