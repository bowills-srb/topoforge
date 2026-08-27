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
wire-length-optimal blocks ("segregated" placement — the output every
communication-aware mapping tool converges toward) produces a
3.8-4.0x learning penalty relative to placement that interleaves
those populations, at negligible energy cost during frozen operation.
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
Welch p = 3.2×10⁻¹⁴). We report effect sizes, statistical tests,
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
  contiguous region (the output a wire-length-minimizing heuristic
  converges toward).
- **Interleaved**: functional types are spatially mixed.
- **Random**: type assignment is independent of position (a control).

We are explicit that our "segregated" condition is our own
implementation of a wire-length-minimizing heuristic, not the
measured output of a published mapping tool such as SpiNeMap or
NEUTRAMS run on our task; we have not benchmarked against those
tools' actual output, and this is discussed further in Section 6.

### 3.4 Statistical methodology

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

---

## 5. Discussion

Taken together, Sections 4.3-4.5 constitute three independent,
increasingly demanding tests of the same core claim, each addressing
a distinct objection a careful reader would raise: is it really about
space, or just locality (4.3); does it depend on our specific
learning rule (4.4); does it survive real data (4.5). The effect
persists through all three, with the magnitude varying by test in
ways we believe are individually explicable rather than arbitrary.

**Practical implication for hardware mapping.** A mapping tool
optimizing purely for communication energy, applied to hardware with
active structural plasticity, may be actively harmful to the
network's ability to learn — the two objectives are not merely
independent but can trade off directly, with our synthetic and
real-data results suggesting the learning cost can substantially
exceed any energy benefit gained.

---

## 6. Limitations

*This section was written adversarially — attempting to identify the
respects in which this work's central claim should not yet be fully
trusted, rather than defending it.*

**Baseline fidelity.** Our "segregated" placement condition is our
own implementation of a wire-length-minimizing heuristic, not the
output of a published mapping tool. To close this gap we implemented
the actual algorithm of SpiNeMap (Balaji et al., arXiv:1909.01843) —
which has no public code release — from its published description:
greedy Kernighan-Lin clustering (SpiNeCluster) followed by
particle-swarm placement (SpiNePlacer), both written from scratch and
verified before use (the partitioner reaches the theoretical minimum
cut on the benchmark graph and matches networkx's bisection; the
placer reduces wire energy to 0.42-0.46x of random assignment). We
ran it through the same Placement-Learning Benchmark (Experiment 38,
ten seeds). The result runs against the intuition that any
wire-length-optimizing tool segregates: SpiNeMap does *not* reproduce
our pathological baseline. It produces 2,604-2,678 units of learned
structure (depending on the connectivity graph supplied) versus 686
for our segregated condition and 2,926 for interleaved — that is,
86-89% of the way from segregated to interleaved, and roughly four
times the learned structure of our own segregated baseline. Our
hand-rolled baseline is therefore *more* pathological than a real
tool, and the 4x figure should be read as the penalty a naive
type-segregation incurs, not the penalty a deployed mapper like
SpiNeMap would produce.

This sharpens rather than overturns the central claim. The penalty
attaches to spatial *segregation of functionally correlated neurons*
specifically, not to wire-length optimization in the abstract.
SpiNeMap minimizes global communication by compacting each population;
at our core pitch (equal to the plasticity radius) that compaction
leaves correlated cross-type neurons within reach, so associations
still form. A tool incurs the penalty only insofar as its output
spreads correlated populations beyond the plasticity radius — which
our block-segregated heuristic does and SpiNeMap, at this scale, does
not. We tested both the pre-plasticity case (SpiNeMap given only the
population structure realistically available before learning) and the
associations-declared case (the target connectivity supplied up
front); both land near interleaved, so the conclusion does not hinge
on that modeling choice. The result is geometry-dependent, however: on
a sparser fabric, or with inter-core distances large relative to the
plasticity radius, even a communication-minimizing placer could
separate populations enough to re-incur the penalty. Quantifying that
crossover is future work.

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
