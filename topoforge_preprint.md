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
3.83x learning penalty (95% CI [3.67x, 3.98x]) relative to placement
that interleaves
those populations, at *identical* energy cost during frozen operation
(bit-identical, since the conditions differ only in which type sits at
which position) and a 1.6% post-plasticity energy premium.
We show this effect survives five independent stress tests: removal
of physical geometry entirely (the effect persists at 2.92x under
pure graph-locality constraints, indicating the mechanism is local-
versus-global connectivity rather than spatial embedding specifically);
replication on a network sized to a published hardware architecture's
own proportions; replacement of our rewiring rule with a published
homeostatic structural-plasticity model containing no correlation,
value, or reward term, under which segregation suppresses cross-type
connectivity to less than half the level a type-blind rule reaches by
chance (0.367 against 0.813, chance ceiling 0.801; twelve seeds,
Welch p = 2.5×10⁻³⁷); most consequentially, replication on real
spike-encoded human speech (Spiking Heidelberg Digits), where an
adjacency-matched, twelve-seed comparison shows a segregated-placement
learning penalty of 2.74x, 95% CI [2.27x, 3.41x], Welch
p = 9.8×10⁻¹¹; and replication on a second speech corpus
(Spiking Speech Commands) across three independent class pairs, at
2.89x, 95% CI [2.36x, 3.50x] — an interval closely overlapping the
Heidelberg result, and therefore evidence of the same effect at the
same magnitude on an independent corpus. The real-data figures are
corrected values: earlier drafts reported 8.22x and 11.61x, inflated
roughly threefold by an initial wiring draw that was pinned across
seeds, and those figures are withdrawn (Section 4.5). The effect
survives the correction at reduced magnitude; the synthetic result is
substantively unchanged, though its published error bars were not.
Segregation's cost is also measured directly in task terms rather than
only structurally: on held-out speech samples the plastic phase never
saw, and at matched pre-learning decodability, segregated placement
costs 6.94 percentage points of classification accuracy
(p = 0.012, n = 40). The penalty is not confined to a
baseline of our own construction: a from-scratch reimplementation of
a published mapping tool (SpiNeMap), given the connectivity graph it
actually has at map time — one that cannot contain associations
plasticity has yet to build — produces a placement whose
plasticity-attributable learning is net-negative, in the same regime as
our segregated baseline rather than partway toward interleaved (Section
4.6). The same optimizer becomes harmless when
the to-be-learned associations are declared to it up front, which
locates the problem in the graph a mapper is given rather than in
wire-length optimization as such. Sweeping the fabric's core pitch
against the plasticity radius shows the penalty saturating to its full
magnitude beyond a pitch of roughly 1.5 radii, while an
association-aware mapper is unharmed at every density tested and, on
sparse fabrics, outperforms uniform interleaving by 1.43x.
An independent third-party measurement on physical hardware (Catalyst
Neuromorphic N4, Section 4.14) bounds the claim's scope from the other
side: on a backpropagation-trained recurrent network pruned to as
little as 0.1% of its weights, segregated, interleaved and random
placements are indistinguishable in routed traffic at every sparsity
level. Sparsity alone does not create the locality the effect requires;
the penalty applies to connectivity built by local structural growth,
not to any sparse network. We report effect
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

**A structural analogy, and a caution about it.** Part of the original motivation for this line of investigation was a visual resemblance between the void-and-filament structure of large-scale cosmic matter distribution and the branching structure of synaptic connectivity. We flag this motivation for transparency, and immediately caution against over-reading it: filament-and-void geometry is a generic outcome of essentially any system built from local interaction rules acting on a stochastically seeded field — gravitational clustering, reaction-diffusion, diffusion-limited aggregation, and percolation near a critical threshold all produce qualitatively similar geometry despite sharing no physical mechanism. Structural resemblance between two such systems is therefore not, by itself, evidence of a shared operative principle; it is closer to a generic signature of "local rule plus randomness" than a specific discovery about either system. Section 4.10 reports the one place in this paper where that analogy was tested on its own mechanistic terms rather than taken as inspiration, with a result that does not straightforwardly vindicate it.

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

Locality enters the rewiring rule through two separate channels, not
one, a distinction confirmed directly rather than assumed (Section
4.10): a bounded radius gates which pairs can accrue C/E/V state at
all (**candidate discovery**), and a separate distance discount, 1/(1
+ 0.05 d²), inside the score that ranks discovered candidates for a
limited number of rewiring slots each epoch (**candidate scoring**).
A placement can therefore fail to learn an association either because
the pair never becomes a candidate, or because it becomes a candidate
but is consistently outcompeted by nearer ones — the paper's mechanism
claim depends on both channels together, not on locality-of-discovery
alone.

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

**Computational reproducibility.** All results in this paper are deterministic and reproduce identically across NumPy versions. This was not true of earlier drafts, and the reason is worth stating because it bounds how precisely any magnitude here should be read.

The structural-plasticity step selects connections to replace by ranking a score array in which most entries are exactly zero — the great majority of candidate pairs have accrued no reward-attributed value. Ranking was performed with NumPy's default sort, which is not stable, so *which* of the tied-at-zero connections was replaced depended on sort implementation details rather than on the simulation. Those details changed between NumPy 1.26.4 and 2.5.1: on the benchmark of Section 4.1, seed 0, the two versions returned 2955 and 2979 for interleaved placement and 682 and 724 for segregated. Every experiment now sorts with an explicitly stable order, which makes tie-breaking index-determined and therefore identical on any platform; the change alters no random draw and no physics, and was verified against a bit-for-bit equivalence test of the real-data implementation before any number was regenerated.

The diagnostic value of that episode outlives the fix. Re-running every experiment under the corrected sort showed that **tie-sensitivity is not uniform across conditions — it is concentrated almost entirely in conditions that learn very little.** Conditions with substantial learned structure moved by under 1% (the three placements of Section 4.8's confirmatory test moved 0.6%, 0.7% and 0.7%; interleaved placement in Section 4.1 moved 0.1%). Conditions in which almost nothing is learned moved far more, because in those the measured quantity consists largely of connections that tie-breaking happened to assign. Since a segregated condition is by construction the one that learns least, and since it sits in the denominator of every ratio we report, ratios are systematically more uncertain than the seed-to-seed variance of either condition alone would suggest.

We therefore ask readers to treat the *direction* and *ordering* of every result here as robust — they survived a change that moved some underlying magnitudes substantially — and to treat specific multipliers as accurate to roughly the leading digit rather than to three significant figures. Where a magnitude proved especially sensitive we say so at the point it is reported, rather than only here.

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
mean learning-quality score of 703.3 +/- 1.9; interleaved placement
achieves 2950.1 +/- 37.7 — a 4.20x ratio. Welch's t-test: t(9.0) =
188.0, p = 1.5x10^-17, Cohen's d = 84.1. At N=5,000: segregated
4005.5 +/- 0.8, interleaved 15399.9 +/- 203.4 — a 3.85x ratio; t(9.0) =
177.1, p = 3.0x10^-17, d = 79.2. The effect is consistent across a
5.5x increase in scale. (Degrees of freedom are Welch's, matching the
test actually used; earlier drafts reported Student's pooled df of 18
alongside a Welch p-value.) As noted in Section 3.4, the very large
Cohen's d values reflect the low noise of a fully controlled
simulation and should not be read as a prediction of effect
magnitude on physical hardware.

**Precision correction (2026-09-04, wiring-draw audit).** The synthetic
*ratio* above survives the pinned-draw defect described in Section 4.5;
its *error bars* do not. This harness contains the same fixed
`default_rng(7)` initial wiring, so the ten seeds reported above reuse
one wiring draw and their standard deviations contain no wiring
variance — the structural audit shows initial taught mass of
1430.0 ± 0.0 against 1488.0 ± 0.0, and excitatory counts of 1136.0 ± 0.0
against 1223.0 ± 0.0, a 7.66% pre-learning advantage to interleaved with
zero variance. Re-running with wiring resampled per seed over logical
identities (twenty seeds, equivalence to the original verified
bit-identically in legacy mode) gives **3.83x, 95% CI [3.67x, 3.98x],
p = 3.4×10⁻¹⁹, d = 11.43** — only 9.3% below the pinned 4.20x, and the
growth sign flip that is this section's corrected primary result is
fully preserved (segregated −722.1 ± 21.3, interleaved +1337.5 ± 253.3).
What must be revised is the reported precision: the pinned interval is
0.05x wide against a true 0.32x, and the interleaved standard deviation
was understated roughly sixfold. **The number stands; the error bars
were fiction.**

Why this section survives where Section 4.5 did not is mechanistic and
was predicted before the audit ran: here the segregated condition has
*zero* adjacency on the measured association pairs, a fact of geometry
that no wiring draw can rescue, so its growth barely moves
(−727.1 → −722.1). On the real-data harness the segregated condition had
adjacency to lose, and an unlucky draw suppressed it roughly threefold.
Every experiment importing this harness — Sections 4.7 through 4.11 and
the topology work of Section 4.10 — inherits understated variance rather
than shifted point estimates, and their intervals should be read as
correspondingly optimistic pending re-derivation.

**Metric correction (2026-08-29, adversarial audit).** The score above
is a raw count of edges landing on the associated cluster-type pair.
The benchmark's *initial* edge set, however, is drawn uniformly at
random over all N neurons with no locality constraint at all —
identical in expectation regardless of placement — so both conditions'
raw scores are inflated by the same non-mechanistic baseline (~1430 at
N=900). For segregated placement specifically, the raw local
adjacency between the measured type pair is exactly zero (verified
directly against the deployed `SpatialGrid` radius), so the mechanism
this paper argues for predicts a floor of *zero* learned structure for
that condition, not merely "4.2x less." Isolating the
plasticity-attributable signal directly — growth = (post-plasticity
score) − (pre-rewire score), using the checkpoint the benchmark already
records — gives a materially different and, we think, more accurate
picture: at N=900, ten seeds, interleaved placement *gains* 1462.3 +/-
36.4 units of cross-type structure during the plastic phase; segregated
placement *loses* 726.6 +/- 2.3 (Welch t = 183.2, p = 1.9x10^-17). This
is a sign flip, not a magnitude difference: segregated placement does
not learn the association weakly, it actively erodes whatever
cross-type structure existed by chance from the random initializer,
because it has zero local candidates to grow toward instead. We regard
this corrected, delta-based quantity as the primary result and the raw
4.20x/3.85x ratios above as a diluted restatement of it, retained for
continuity with the originally published numbers. The sign flip
replicates at N=5,000: interleaved placement gains 7362 +/- 193 units;
segregated placement loses 4014 +/- 1 (Welch t = 176.8, p = 3.0x10^-17).
The effect is not a small-N artifact.

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
(a 4.43x ratio in one benchmark), suggesting the substrate's own
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

Segregated: 701 +/- 0. Interleaved: 2045 +/- 10. Ratio: 2.92x — smaller
than the 4.20x observed with spatial embedding at the same scale, but
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

This test substitutes an externally measured timing curve while retaining our own candidate-selection and scoring machinery. Section 4.12 goes further, replacing the structural-plasticity rule in its entirety with a published model that contains no correlation, value, or reward term at all.

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

Correcting both issues — matching the decay rate to the life length, and placing both conditions in matched geometry (same disc, same neuron density, differing only in whether functional types occupy contiguous wedges or are shuffled) — yields an adjacency-matched comparison at the validated life-length/decay regime (approximately 8,600 steps, V retention 0.651).

**Correction (2026-09-04): this section's headline was inflated roughly threefold by a pinned random draw, and the corrected value is 2.74x.** An earlier version of this paper reported 8.22x, 95% CI [7.03x, 10.05x], across twelve seeds. That number is withdrawn. The cause is stated in full because it is instructive, and because it defeated every check we had.

The simulation harness drew its initial connectivity *and* its excitatory/inhibitory assignment from a fixed generator, `default_rng(7)`, independent of the per-run seed. Two things followed. First, every "seed" in every real-data experiment reused **the same initial wiring**, so the across-seed standard deviations and bootstrap intervals we reported contained no wiring-draw variance whatsoever — the structural audit shows excitatory input→output counts of 313.0 ± 0.0 against 341.0 ± 0.0, an 8.95% pre-learning advantage to interleaved with *literally zero* variance. Second, the wiring indices referred to global neuron identity while the placement determines which identity holds which type, so the two conditions did not receive the same logical network.

Re-running with the wiring resampled per seed, and drawn over logical identities so both conditions receive an identical network, gives **2.74x, bootstrap 95% CI [2.27x, 3.41x], Welch p = 9.8×10⁻¹¹, Cohen's d = 2.85** across twenty seeds. Segregated bridge mass is 2941.5 ± 1548.1; interleaved is 8045.6 ± 2003.4. Decomposing the change, un-pinning the draw accounts for essentially all of it and the logical-connectivity asymmetry for almost none: resampling raises the *segregated* condition roughly threefold (1011 → 2942) while interleaved barely moves (−11%). `default_rng(7)` was simply an unusually unfavourable draw for the segregated condition specifically. The withdrawn and corrected intervals do not overlap. An equivalence check confirms the corrected harness reproduces the original bit-identically when run in legacy mode, so the physics is unchanged and only the draw differs.

Two features of this failure are worth recording. The corrected interval is **narrower** than the withdrawn one, not wider — so the published interval was not merely overconfident, it was centred on the wrong value while looking well determined, which is why seed sweeps never flagged it. And the defect's severity depends on whether geometry leaves room for it: the synthetic benchmark of Section 4.1, audited identically, barely moved (Section 4.1's correction note), because there the segregated condition has *zero* adjacency on the measured pairs and no wiring draw can rescue what geometry forbids.

**A regime comparison we now withdraw.** An earlier version reported that extending exposure to two epochs reduced the ratio to 5.09x, and presented "approximately 5.1x to 8.2x depending on training exposure" as the honest range for real speech data. Both figures were measured under the pinned draw. The longer-exposure regime has **not** been re-run under corrected wiring, so we withdraw the range rather than restate it, and report only the corrected validated-regime figure of 2.74x. Whether the effect still varies with training exposure is now an open question.

The corrected value no longer sits above the synthetic result. At 2.74x [2.27, 3.41] against the synthetic 3.83x [3.67, 3.98] (Section 4.1, corrected), real speech data now shows a *somewhat smaller* penalty than hand-designed patterns — the opposite direction to what the withdrawn numbers suggested. We decline to construct a post-hoc account of why, noting only that the two are measured on different tasks at different scales (N=200 versus N=900), and that the earlier puzzle we reported here — why real speech should exceed synthetic — dissolves along with the numbers that produced it.

**A comparison we previously drew, and now withdraw.** An earlier version of this section argued that sample diversity is not the driver of the effect, on the grounds that a four-samples-per-class configuration and a twenty-samples-per-class configuration produced statistically indistinguishable ratios (7.22x versus 7.25x). Those two figures no longer agree: regenerated under a deterministic tie-breaking rule (Section 3.4), the four-sample configuration gives 16.24x, 95% CI [11.70x, 26.26x], against the twenty-sample configuration's 8.22x [7.03x, 10.05x] — non-overlapping intervals. We do not read this as evidence that sample diversity *does* drive the effect. The four-sample configuration's segregated condition accumulates only 312 units of bridge mass, a quantity small enough that which connections it contains is substantially determined by tie-breaking rather than by learning, so its ratio does not reliably estimate anything. The earlier agreement at 7.22x/7.25x is better understood as two poorly-determined quantities happening to coincide than as a demonstration. The sample-diversity question is therefore open, and the twenty-sample configuration — whose segregated condition is correspondingly better determined — is the one we report as the headline. *(All figures in this paragraph, including the 8.22x and the 16.24x, were measured under the pinned wiring draw corrected above and are retained only to document the withdrawn comparison. Neither has been re-derived under corrected wiring, and neither should be cited as a current result.)*

We report one further caveat, revised from earlier drafts. Segregated-condition outcomes remain substantially more variable than interleaved ones in every regime (relative standard deviation 32.8% at the validated regime and 60.2% at the longer exposure, against 9.9% for interleaved in both). An earlier version additionally reported that two of twelve seeds in the longer-exposure regime produced bridge mass of essentially zero; on regeneration no seed does, the minimum being 274. The reliability difference between placements is real and we continue to report it — segregation makes learning outcomes less consistent, not merely smaller on average — but the specific claim of near-total failure in individual runs does not survive and is withdrawn.

This section rests on a single corpus. Section 4.13 replicates it on a second, larger speech dataset across three independent class pairs, which is where the question of corpus-specificity is addressed.

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

Learning quality follows that structure, and the two graph conditions fall on opposite sides of the benchmark (Table 4). On the population graph — the honest map-time case — SpiNeMap lands 7.7% of the way from our segregated baseline toward interleaved on raw taught mass: it learns 3.37x less than interleaved placement (876.3 ± 49.8 vs 2950.1 ± 37.7; Welch p = 4.6×10⁻²⁵, d = 46.9), statistically distinguishable from our hand-rolled segregated condition but only 1.25x above it. On the functional graph it lands at 99.8% — indistinguishable from interleaved (2946.5 ± 26.1, a 0.1% difference, p = 0.81).

**Metric correction (2026-08-29, adversarial audit).** As in Section 4.1, raw taught mass is inflated by a shared non-local random-initialization baseline common to every placement, so the "7.7% of the way" figure understates how badly the population graph fares. Isolating plasticity-attributable growth directly, ten seeds: SpiNeMap-population *loses* 554.3 ± 46.9 units of cross-type structure during the plastic phase — the same sign as our segregated baseline's loss of 726.6 ± 2.3, not a partial version of interleaved's gain (Welch t = 10.97 vs. segregated growth, p = 1.6×10⁻⁶). SpiNeMap-functional gains 1516.0 ± 25.3, statistically indistinguishable in kind from interleaved's gain of 1462.3 ± 36.4 (t = 271.3 vs. segregated growth, p = 4.4×10⁻¹⁹). The corrected reading strengthens rather than weakens the section's conclusion: SpiNeMap given the map-time graph does not merely learn less than interleaved, it is mechanistically in the same regime as our segregated baseline — actively eroding non-mechanistic pre-wiring rather than building any cross-type structure at all.

**Table 4.** Learning quality, mean ± s.d. over ten seeds, N=900. "Taught mass" (raw, retained for continuity) is inflated by a shared non-mechanistic baseline (Section 4.1); "Growth" (plastic − pre-rewire, the corrected primary quantity) isolates what structural plasticity itself did. Position is on the segregated → interleaved axis using growth.

| Placement | Taught mass | Growth | Position (growth) |
|---|---|---|---|
| Segregated (ours) | 703.3 ± 1.9 | −726.6 ± 2.3 | 0% |
| SpiNeMap — population graph | 876.3 ± 49.8 | −554.3 ± 46.9 | 7.9% |
| Random | 2733.7 ± 23.3 | +1327.0 ± 23.3 | 93.9% |
| Interleaved (ours) | 2950.1 ± 37.7 | +1462.3 ± 36.4 | 100% |
| SpiNeMap — functional graph | 2946.5 ± 26.1 | +1516.0 ± 25.3 | 102.5% |

The conclusion is the opposite of what we previously reported, and it strengthens rather than weakens the paper's central claim. A published, communication-minimizing mapping tool, given the connectivity graph it actually has at map time, **does** produce the pathological placement: our hand-rolled segregated baseline is not a strawman but a close stand-in for what such a tool emits on a plastic substrate. What rescues SpiNeMap is not its objective but its input — declaring the to-be-learned associations up front moves it the entire way to interleaved. This is a sharper and more actionable statement of the problem than "wire-length optimization is harmful": the harm comes from optimizing a communication objective over a graph that omits the associations plasticity has yet to build, and the same optimizer becomes harmless the moment those associations are in the graph it is given. Since a plastic system's whole purpose is to learn associations that were not known at map time, the population-graph case is the one that matters in practice.

(One internal inconsistency is also corrected here: an earlier Table 4 reported 686 ± 5 and 2926 ± 26 for the segregated and interleaved conditions, which disagreed with the same conditions in Section 4.1. The values above were freshly reproduced and match Section 4.1 exactly.)

### 4.7 Dose-response and mediation: the reachable correlated fraction

Every result to this point is a two-point contrast (segregated versus interleaved). To test whether placement acts *causally* through a single measurable quantity rather than through some property peculiar to one geometry, we interpolated continuously between the two conditions. A knob α scatters each neuron from its segregated position to a random disc position with probability α, so α=0 reproduces the segregated placement and α=1 the interleaved one. Before any learning we measure a placement-only mediator — the mean number of input neurons within the plasticity radius of each output neuron, the structural "reach" that permits input→output bridges to form at all. We swept α on the real-data task at both K=2 and K=4 (eight seeds each, the same 8,600-step regime).

Two predictions were registered before running. First, a **dose-response**: total learned bridge mass should increase monotonically with α. It does — Spearman ρ = 1.000 at both K=4 and K=2, with learning rising 3.0× (K=4) and 3.9× (K=2) from α=0 to α=1. (An earlier draft reported ρ = 0.89 at K=2 and apologised for "a mild non-monotonic wobble" in the low-α region; that wobble was an artifact of non-deterministic tie-breaking rather than seed noise, and disappears under the correction of Section 3.4.) Second, **mediation**: if reach is the variable placement acts through, normalized learning should be a single function of reach, with the K=2 and K=4 sweeps falling on one curve rather than two. This holds — pooling both K, reach predicts normalized learning with Pearson R² = 0.92 (Spearman ρ = 0.97), and the two sweeps *interleave* along the reach axis (Table 5) rather than separating.

**Table 5.** Dose-response pooled across K and α, sorted by the pre-learning reach metric. Learning is normalized to each K's interleaved (α=1) ceiling; K=2 and K=4 rows interleave rather than forming two curves.

| K | α | mean reach | normalized learning |
|---|---|---|---|
| 2 | 0.00 | 24.05 | 0.256 |
| 2 | 0.15 | 25.82 | 0.444 |
| 4 | 0.00 | 26.91 | 0.331 |
| 2 | 0.30 | 28.43 | 0.470 |
| 2 | 0.45 | 31.13 | 0.547 |
| 2 | 0.60 | 33.75 | 0.715 |
| 4 | 0.33 | 34.02 | 0.643 |
| 2 | 0.80 | 34.62 | 0.888 |
| 4 | 0.67 | 36.96 | 0.812 |
| 2 | 1.00 | 37.28 | 1.000 |
| 4 | 1.00 | 38.68 | 1.000 |

We read this as evidence that placement affects learning through a single mediating quantity — the fraction of correlated pairs the local plasticity rule can physically reach — rather than through anything specific to a given geometry, task, or class count. (Section 4.8 revisits how that quantity should be *measured*: across a wider set of placement families, the mean-count form used here fails, and a coverage form succeeds. The mediation claim survives; the metric is refined.) On one curve it accounts for why removing geometry still leaves an effect (Section 4.3), why a communication-minimizing mapper lands next to our segregated baseline when given the map-time graph but at interleaved when given the associations (Section 4.6), and why the penalty's magnitude shifts with regime (Section 4.5): each changes reach. One honest limit remains at this point: in the sweep, reach is *measured*, not manipulated while holding all other geometric properties fixed, so the evidence so far is correlational. We resolve that next with a controlled dissociation.

To turn the mediation from correlational to causal, we manipulated reach and the most obvious confound — whether output classes are spatially clustered or mixed — as two independent factors. Reach was set by rigidly translating the output population a distance from the input population, which changes input–output distances while preserving input–input and output–output structure exactly. Class arrangement was set by assigning class labels to a *fixed* set of output positions either in contiguous spatial blocks or shuffled: a pure relabeling that leaves the geometry, and therefore reach, byte-identical between the two (confirmed to machine precision at every separation — the reach column of Table 6 is the same number twice over). Across eight seeds and four separations (mean reach 71.0 → 57.0 → 22.5 → 2.6 input neighbours per output neuron), learning tracked reach monotonically and very steeply: a 293-fold change from highest to lowest reach under clustered arrangement and 346-fold under mixed, Spearman ρ = 1.00 in both, pooled R² = 0.90.

**Table 6.** Controlled dissociation. At each separation, class arrangement (clustered vs mixed) is a pure relabeling of identical output positions, so reach is byte-identical across the two columns; only the input–output separation changes reach. Total bridge mass, mean ± s.d. over eight seeds.

| separation | mean reach | clustered | mixed | arrangement *p* (paired) |
|---|---|---|---|---|
| 0 | 71.03 | 8796 ± 6516 | 8223 ± 4536 | 0.685 |
| 5 | 56.97 | 4101 ± 716 | 4419 ± 599 | **0.005** |
| 10 | 22.52 | 669 ± 310 | 975 ± 541 | **0.048** |
| 15 | 2.57 | 30 ± 17 | 24 ± 17 | 0.466 |

Reach is therefore the variable placement acts *through*, and the dissociation establishes it causally rather than by correlation. Class arrangement, however, is **not** inert, and here we correct a claim made in an earlier version of this paper. That version reported arrangement as non-significant at every separation (p = 0.09–0.28) and described the spatial clustering of classes as, with reach held fixed, having no effect on its own. Under deterministic tie-breaking (Section 3.4) arrangement is significant at two of the four separations — p = 0.005 at separation 5 and p = 0.048 at separation 10 — and in both cases mixed arrangement outperforms clustered. The earlier version had already declined to claim the effect was exactly zero, noting "a weak, non-significant trend favouring mixed"; that trend is now significant, and the correct statement is that arrangement has a small but real effect in the same direction the paper's central thesis predicts.

The two effects differ in magnitude by more than two orders of magnitude: arrangement moves learning by 1.08x (separation 5) and 1.46x (separation 10), against reach's 293–346x. Reach remains overwhelmingly the dominant channel, and the paper's mechanism is unchanged. But "interleaving helps only by increasing reach" is too strong: at matched reach, mixing classes still helps a little. This converges with an independent result in Section 4.8, where the partner *share* of a neuron's reachable neighbourhood was also found to have a small, real effect at matched reach and coverage — mixed arrangement is precisely what raises that share. Two experiments built for different purposes therefore agree that a second, minor channel exists alongside reach, and we now report it as such rather than as noise.

---

### 4.8 Fabric sparsity: when the penalty saturates, and what protects against it

Section 4.6 shows a real mapping tool incurring the penalty at this benchmark's geometry, where the core pitch happens to equal the plasticity radius. Since Section 4.7 identifies reach as the operative variable, the natural question is how the result moves when that coincidence is broken. We swept the fabric's core pitch from 0.5x to 3x the plasticity radius — the radius being a property of the substrate's plasticity mechanism and the pitch a property of the floorplan — holding everything else fixed. Placements are not regenerated per pitch: each strategy's placement is produced once by the same functions used in Sections 4.1 and 4.6, and each core's disc is then rigidly translated to the new pitch, which preserves within-core geometry exactly and is bit-identical to the original at ρ = 1 (both verified). Eight seeds per cell.

**Table 7.** Learning (taught mass) versus fabric sparsity ρ = core pitch / plasticity radius. Mean ± s.d. over eight seeds, N=900. The segregated baseline is included as the floor.

| ρ | interleaved | SpiNeMap — population | SpiNeMap — functional | segregated |
|---|---|---|---|---|
| 0.5 | 3064 ± 20 | 1546 ± 70 | 2979 ± 39 | 717 ± 3 |
| 1.0 | 2959 ± 37 | 875 ± 51 | 2951 ± 26 | 704 ± 2 |
| 1.25 | 2278 ± 26 | 718 ± 14 | 2928 ± 16 | 701 ± 0 |
| 1.5 | 2050 ± 16 | 701 ± 1 | 2931 ± 15 | 701 ± 0 |
| 2.0 | 2051 ± 19 | 701 ± 1 | 2931 ± 15 | 701 ± 0 |
| 3.0 | 2051 ± 19 | 701 ± 1 | 2931 ± 15 | 701 ± 0 |

Three results, one of which contradicts a prediction we registered before running.

**The penalty saturates rather than switching on.** Population-graph SpiNeMap already carries a penalty at the densest fabric (1.98x below interleaved at ρ = 0.5) and degrades monotonically until, at ρ ≥ 1.5, it is numerically identical to the segregated baseline itself (701 ± 1 vs 701 ± 0). So sparsity does not create the penalty — the map-time graph does — but sparsity determines how complete it becomes. Beyond ρ ≈ 1.5 a communication-minimizing placement built on the pre-plasticity graph has lost the ability to form cross-type associations entirely, and no amount of further spreading changes that.

**Declaring the associations protects at every fabric density.** Functional-graph SpiNeMap is flat across the whole sweep (2979 → 2931, a 1.6% change over a 6x change in pitch). More striking, beyond ρ = 1.25 it *exceeds* interleaved placement — 2931 vs 2051, a 1.43x advantage — because it deliberately co-locates the specific populations that must associate, whereas interleaving mixes all types uniformly and therefore spends core capacity on neurons that have no reason to be adjacent. On a sparse fabric, uniform interleaving is a blunt instrument; targeted co-location of correlated populations is strictly better. This is the paper's most directly actionable result for tool builders: the fix is not to abandon communication-aware mapping but to give it the association structure.

**The interleaved control, and a prediction that failed.** We predicted interleaved placement would be flat in pitch, on the reasoning that its cross-type partners sit inside the same core disc (radius 1.5, well under the plasticity radius) and respacing cores cannot remove them. That is wrong: interleaved also draws partners from *neighbouring* cores at low ρ, and loses them, falling 1.49x from ρ = 0.5 to ρ = 1.5. What the sweep does deliver is a stronger control than the predicted flatness would have been. Interleaved learning is identical at ρ = 1.5, 2.0 and 3.0 (2050, 2051, 2051) across a 4x increase in mean squared wire distance, and its reach is likewise pinned at 3.00 across those points. The plasticity rule's rewiring score contains a distance discount, 1/(1 + 0.05d²), so a plausible alternative explanation for every result in this paper is that distance *per se* penalizes candidate pairs. That explanation predicts continued decline from ρ = 1.5 to 3.0. It does not occur, in any of the four placement families. Distance matters only through whether it puts a correlated partner inside the radius.

**Refining the mediator (exploratory).** Section 4.7 measured reach as a mean *count* of correlated partners within the radius, on a placement family where every reasonable form of that measure co-varies. This sweep breaks the tie, and the count form does not survive it: pooled across all four placement families and six pitches, count explains R² = 0.263 of the variance in learning. The failure is systematic, not noisy — population-graph SpiNeMap at ρ = 0.5 has a *higher* mean count than interleaved (36.8 vs 29.6) while learning half as much, because type-pure cores concentrate the reachable pool onto boundary neurons and leave core interiors with nothing in reach. Substituting *coverage* — the fraction of correlated neurons with at least one partner within the radius — raises this to R² = 0.837 (Spearman ρ = 0.920), and five pairs of conditions with matched count (6.3–8.1) but different coverage all separate as coverage predicts: coverage 1.00 gives ≈2980 taught, coverage 0.61 gives 904. Adding count back on top of coverage buys +0.003 R², while coverage adds +0.178 over count alone. Pool size is not irrelevant, but its effect is strongly saturating: within interleaved placement, where coverage is pinned at 1.000 at every pitch, a 9.9x reduction in mean count costs only 1.49x in learning, against roughly 4x for losing coverage altogether.

**Confirming it on a family built to discriminate.** Because coverage was formulated after seeing the data above, we ran a separate confirmatory test on three placements constructed in advance to make the rival accounts predict different orderings, with the predictions and decision rules registered before the run. All 900 neurons are placed in blobs of constant density on a lattice spaced so that every intra-blob pair lies inside the plasticity radius and every inter-blob pair outside it, so a blob's composition fixes its members' reach by construction rather than by measurement (verified: zero inter-blob neighbours, maximum intra-blob distance 2.95-4.14 against a radius of 5.0). C1 uses 60 blobs of 15; C2 uses 30 blobs of 30 with all five types mixed; C3 uses 30 blobs of 30 with the partner types concentrated into a few blobs and the remaining neurons parked in blobs whose types are not associated. C1→C2 changes partner share 2.1x at fixed count and coverage; C2→C3 changes coverage 2.4x at matched count and partner share. The design is adversarial to the coverage account: C3 has the *highest* mean count (6.25 against 6.00), so the count account predicts C3 wins outright rather than merely tying.

**Table 8.** Confirmatory test. Mean ± s.d. over eight seeds, N=900. Metrics are measured on the built placements, not assumed.

| Condition | count | frac | cover | Taught mass |
|---|---|---|---|---|
| C1 spread15 | 6.00 | 0.429 | 1.000 | 2925 ± 35 |
| C2 dilute30 | 6.00 | 0.207 | 1.000 | 2719 ± 53 |
| C3 clumped30 | 6.25 | 0.216 | 0.417 | 2096 ± 44 |

The count account is refuted: C3 carries the largest reachable pool and learns the least, significantly below both other conditions (vs C2, 1.30x, Welch p = 7.3×10⁻¹³, d = 12.8; vs C1, 1.40x, p = 1.7×10⁻¹⁵). The coverage effect is confirmed in isolation, since C2 and C3 are matched on both count and partner share and differ only in coverage. Partner share, however, is not inert: C1 exceeds C2 by 1.08x (p = 8.4×10⁻⁷, d = 4.6) at matched count and coverage. Smaller blobs have proportionally more boundary, making C1 about 7% sparser than C2 in nearest-neighbour spacing, so we controlled for that directly by rescaling C2's blob radii to match C1's spacing at fixed count, share and coverage: learning moved 0.0% (p = 0.997), against the 7.6% C1-vs-C2 gap. The partner-share effect is therefore real rather than a density artifact.

We therefore restate the mechanism of Section 4.7 more precisely, and with one part of it now confirmed rather than merely fitted. The mean *size* of the reachable correlated pool is not the mediator — an account based on it is refuted by a placement built in advance to test it. What matters is fractional: primarily *whether* a correlated neuron has any partner the local rule can reach (a 2.4x coverage change moves learning 1.30x), and secondarily what share of its reachable neighbourhood those partners represent (a 2.1x change moves learning 1.08x). Neither quantity alone predicts magnitudes across all conditions — coverage falls 2.4x between C2 and C3 while learning falls only 1.30x, and the corresponding drop in Section 4.6 was far steeper — so we claim the ordering and the mechanism, not a calibrated functional form.

### 4.9 The energy-learnability frontier

Everything above concerns what a placement can learn. The objective a mapping tool actually optimizes is communication cost, and this paper has so far reported that side of the ledger only in passing. This section measures both together: for each placement, the summed squared wire length of the network's *realized* connections — after structural plasticity has rewired them — against what that placement learned. Eight seeds, N=900.

One structural fact makes the central comparison exact rather than statistical. In this benchmark the segregated, interleaved and random conditions place neurons at *identical coordinates* and differ only in which type label sits at each position, and the initial connectivity is seeded identically. Any energy difference between them is therefore attributable entirely to the connections plasticity built, with no geometric confound whatsoever. The two SpiNeMap conditions do differ geometrically, since clustering assigns different neurons to different cores, and are reported alongside rather than as part of that exact comparison.

**Table 9.** Communication energy (summed squared wire length over realized connections) and learning, mean over eight seeds. "Frozen" is before plasticity, "plastic" after. Sorted by learning.

| Placement | frozen E | plastic E | taught | E per unit taught |
|---|---|---|---|---|
| Segregated (ours) | 5.019×10⁶ | 2.525×10⁶ | 704 | 3589 |
| SpiNeMap — population graph | 4.983×10⁶ | 2.497×10⁶ | 875 | 2854 |
| Random | 5.019×10⁶ | 2.556×10⁶ | 2730 | 936 |
| Interleaved (ours) | 5.019×10⁶ | 2.566×10⁶ | 2959 | 867 |
| SpiNeMap — functional graph | 4.989×10⁶ | 2.501×10⁶ | 2951 | 847 |

**Inference energy is not approximately equal; it is identical.** Section 4.1 reported that frozen-phase energy "differs negligibly" between placements. It differs not at all: across all eight seeds the segregated and interleaved conditions record bit-identical frozen energy (5.01906×10⁶), which follows from their sharing coordinates and initial connectivity exactly. The claim that the learning penalty is incurred at zero inference-energy cost can be stated without hedging.

**Choosing learnability does cost communication energy, and the cost is 1.6%.** We predicted before running that interleaving would carry no post-plasticity energy penalty. That prediction is falsified: interleaved placement ends with 1.0162× the wire energy of segregated (2.5656×10⁶ vs 2.5247×10⁶, Welch *p* = 1.9×10⁻⁹). The effect is real and statistically unambiguous. It is also, in magnitude, 1.6%. **Metric correction (2026-08-29):** "bought with 4.20× the learned structure" understates what the 1.6% premium actually buys. On the corrected, delta-based learning metric (Section 4.1), segregated placement's growth is *negative* (−726) while interleaved's is positive (+1471) — the 1.6% energy premium is not purchasing a 4-fold quantity increase, it is purchasing the difference between a placement that erodes structure and one that builds it, a distinction a same-sign ratio cannot express. Per unit of learned structure the old ordering inverts sharply (segregated: 3589 energy units per unit taught vs. interleaved's 867); the corrected version does not have a finite ratio at all, since segregated's plasticity-attributable denominator is negative.

**Plasticity reduces communication energy.** In every condition, post-plasticity energy is roughly half the frozen value (≈2.5×10⁶ against ≈5.0×10⁶). This is a consequence of the rewiring rule's distance discount: connections are replaced preferentially with nearby partners, so a network that rewires ends up cheaper to communicate across than the random initial connectivity it started from, regardless of placement. Structural plasticity is not merely compatible with a communication-energy objective here; left to run, it pursues one.

**The association-aware mapper is the best available option, but it does not dominate outright.** SpiNeMap given the functional graph achieves the lowest energy-per-unit-learned of any condition tested and beats our segregated baseline on both axes at once — 0.990× the energy at 4.20× the raw-taught learning. Against the other high-learning conditions it trades rather than dominates on raw taught mass: relative to interleaved placement it spends 0.975× the energy for 0.997× the learning, and relative to its own population-graph counterpart it spends 1.001× the energy for 3.37× the learning. An earlier version of this paper described it as Pareto-dominant over every condition tested, with no axis on which it was worse; that was measured before the tie-breaking correction of Section 3.4 and is not correct. **Metric correction (2026-08-29):** on the corrected, delta-based learning metric the comparisons against segregated and the population-graph mapper are sign flips, not ratios — functional-graph SpiNeMap's growth is +1521 against segregated's −726 and population-graph's −555 — while the comparison against interleaved remains an even trade (growth ratio 1.03×, energy 0.975×). The differences that remain on the losing axis (interleaved) are small — a few percent either way — and we would not want a reader to draw a practical distinction from them. The defensible statement is that supplying the association structure costs essentially nothing on the energy axis while being the only tested condition, other than interleaving itself, that reliably builds rather than erodes cross-type structure, and that this holds at every fabric density tested (Section 4.8).

### 4.10 Mechanism validation, a sparse-bridge counterexample, and connectivity topology

Three further checks, run 2026-08-29 as a deliberate adversarial pass against the paper's own central claim rather than an extension of it. N=900, the PLB substrate, eight seeds unless noted.

**The mechanism holds when both its locality channels are removed together, and not before.** The claim throughout this paper is that a local plasticity rule cannot discover associations between neurons it cannot reach. We tested this directly by forcing the rule to be non-local and checking that the placement effect disappears, as the mechanism requires. A first attempt — expanding the candidate-discovery radius to exceed the substrate's extent — only partially closed the gap (segregated growth improved from −727 to −523, but a large, significant gap remained, Welch *p* = 2.3×10⁻¹⁷). The reason is that locality enters twice (Section 3.2): expanding the discovery radius fixes which pairs can become candidates, but the rewiring rule's score, (V + 0.01·C) / (1 + 0.05·d²), still weights discovered candidates by true physical distance. Removing *both* channels — global candidate discovery and an unweighted score — collapses the gap completely: segregated and interleaved converge to statistically indistinguishable, similarly-sized positive growth (+1556 vs. +1547, Welch *p* = 0.72), not merely a non-significant difference but the same outcome by two independent routes. This is confirmatory evidence for the mechanism as now stated precisely in Section 3.2, and it corrects an imprecision in earlier statements of it that named only candidate discovery.

**A sparse long-range bridge does not cheaply substitute for interleaving.** Real cortex is locally segregated at the columnar scale, with cross-domain association carried by sparse long-range white-matter projections rather than local mixing — a standing counterexample worth testing directly rather than dismissing. We relocated a small fraction of one associated population's neurons from their segregated block into the partner population's territory (physically embedded neurons, not synthetic edges — a fair analogue of a long-range projection terminating where local circuitry can actually use it) and swept that fraction from 0% to 40%. Recovery of the segregated-to-interleaved gap tracks the bridge fraction *roughly proportionally* rather than the strongly front-loaded curve a "cheap sparse fix" would need: a 10% embedded-neuron budget recovers only 21% of the gap, and 40% is needed to recover 74%. Every step is individually significant (*p* < 10⁻⁷ throughout) and monotonic, as predicted, but the shape refutes the practical hope that a small bridge budget approximates interleaving on this substrate. We register the caveat that only individually-scattered embedded neurons were tested; a structured bridging scheme (e.g. bridge neurons clustered together, or closer to how a real cortical projection pattern terminates) might behave differently, and this result should be read as refuting the naive version of the counterexample, not every possible one.

**Connectivity topology has a real effect beyond first-order reach statistics, confirmed at matched count and coverage — smaller than initially measured, and in the opposite direction from what a naive reading of the cosmic-web analogy motivating this project would predict.** Sections 4.7-4.8 established that coverage (whether a neuron has any reachable partner) is the dominant mediator of the placement effect. We asked the sharper question directly: at *matched* coverage and count, does the large-scale shape of the connectivity — a connected, sparse, filament-like network versus isolated, locally-redundant blobs — matter on its own? An exploratory sweep (filament geometry generated as a smoothed density field over a Voronoi ridge skeleton, chosen for genuine curved, void-containing structure after an earlier polygonal version was correctly flagged as not resembling real filament/void geometry) found a striking, if imprecisely matched, result: isolated blobs stayed robustly positive down to coverage ≈0.42, while the connected filament network collapsed to net-negative growth by coverage ≈0.5. A confirmatory follow-up — a 2×2 design (topology × coverage level) with placements fixed in advance, predictions and a decision rule registered before running, and count matched within 9% and coverage within 6% (substantially tighter than the exploratory pass) — produced a partial confirmation: the high-coverage manipulation check replicated a third time (indistinguishable, *p* = 0.12), and the low-coverage contrast was significant (filament learns 1.81× less than blob at matched coverage, *p* = 6.8×10⁻¹⁰), but filament growth remained net-*positive* (+381) rather than crossing to net-negative as the exploratory pass suggested. The calibrated reading is that the exploratory magnitude was partly an artifact of imperfectly matched count (its low-coverage filament configurations had count as low as half the blob comparison point, stacking an additional count deficit on top of the real topology effect); once controlled, a real, confirmatory-grade, but smaller effect remains. We read this mechanistically as a redundancy effect rather than a coverage effect: isolated blobs are locally-redundant cliques that tolerate partial reach loss, while a connected, sparse, chain-like network has no redundant path to fall back on, so the same coverage loss costs more. This is, to our knowledge, the first result in this project's history that depends on connectivity topology beyond simple local-reach statistics, but its direction is the opposite of what the cosmic-web analogy that motivated the investigation would suggest: connected, web-like structure is *more* fragile under reduced local opportunity than fragmented structure, not more robust. We regard the resemblance between cosmic-web and cortical connectivity patterns that motivated this line of inquiry as visually real but mechanistically unproven; local-interaction systems generically produce filament/void geometry regardless of the underlying physics (Section 2), so structural resemblance alone is not evidence of a shared operative principle, and this section's result is offered as the first test of that principle on its own mechanistic terms, not a vindication of the analogy that suggested testing it.

**Post hoc, this finding survived a genuine search, not just the one confirmatory comparison above.** A separate, tinkering-grade exploration (reduced seed counts, not pre-registered, reported here only as corroborating context rather than a numbered result) tried six successive ways to close the blob/filament gap at matched coverage: tube thickness, fragmentation into disconnected components, fragmentation combined with blob's own discrete composition trick (the best partial fix found, reaching 1.81-2.50x below blob depending on exact matching, consistent with the confirmatory result above), and three physical-migration mechanisms letting neurons dynamically relocate toward learned or sensed value. None closed the gap. A final systematic search — 18 shapes spanning component count, local density, and thickness, calibrated to blob's coverage and screened then refined with additional seeds — found nothing that beat blob; the best candidate found lost by 2.4x despite having 3.6x blob's raw local count, which rules out insufficient search as the explanation. We report this as strengthening, not establishing, the confirmatory result above: isolated, locally-redundant topology appears to be a genuine local optimum for this substrate's plasticity rule, not merely the one alternative we happened to test.

### 4.11 Practical actionability: how much foreknowledge does a mapper actually need?

Sections 4.1-4.10 establish that placement matters and why. None of them establish whether a real mapper could act on that fact under realistic conditions. Section 4.6 tested only two extremes — SpiNeMap given zero foreknowledge of future associations (the honest map-time case) and complete foreknowledge (an unrealistic upper bound) — and no real deployment has either. This section asks the practically decisive question: given only *partial* foreknowledge, how much of the functional graph's benefit does a mapper still capture?

**On the synthetic benchmark, the answer is a threshold, not a slope.** We generalized SpiNeCluster's input graph to accept a continuous knowledge_frac in place of the hard population/functional distinction — the cross-type affinity SpiNeCluster is given for the two association pairs, scaled continuously rather than switched on or off, everything else about the pipeline unchanged (verified to reproduce population and functional exactly at knowledge_frac=0 and 1). Growth stays flat and negative from 0% to 75% foreknowledge (−555, −521, −494, −508, −414) and jumps only at 100% (+1521; population-to-functional span 2076, eight seeds per point). Fifty-percent recovery is not reached until 100% knowledge. The mechanistic reading is that SpiNeCluster is a hard combinatorial partitioner — a neuron is assigned to one cluster or another, with no partial assignment — so a partial-confidence cross-type signal has to actually *outweigh* the always-full-strength same-type affinity before it changes which cluster a neuron lands in; below that threshold it is real information that the optimizer's discrete choices cannot use.

**On real data, the same conclusion holds, after the audit caught and corrected an initial false positive.** We built a genuinely data-derived association graph from real SHD samples — for each of 140 real input channels, its real spike activity during target-class versus distractor-class presentations, in place of the synthetic uniform-weight matrix — and re-ran the sweep on it. A first pass (one placement/clustering seed per knowledge level) appeared to refute the threshold finding outright: 25% knowledge recovered 135% of the population-to-functional gap, beating full knowledge. This did not survive a robustness check. SpiNeCluster and SpiNePlacer are randomized heuristics (multi-restart Kernighan-Lin and particle-swarm optimization), and a second pass with three independent placement seeds per knowledge level showed the 25% result was carried by a single seed (+3512) against two unremarkable ones (+633, +465) — placement-search variance, not a knowledge effect, and the finding was withdrawn as inconclusive rather than reported. A third, properly-powered pass (ten independent placement seeds per knowledge level, 240 runs total, with a resolvability rule — adjacent points differing by less than 2 standard errors of the placement-seed mean are reported as statistically flat — registered before running) resolved it: three of five steps are real, two are noise. There is a small genuine gain from 0% to 10% knowledge, then a real plateau from 10% through 50% (three statistically indistinguishable points), then the real payoff arrives at 50-75% knowledge and continues to 100%. Fifty-percent recovery is reached only at 75% knowledge; ninety-percent recovery only at 100%. This is not an identical curve to the synthetic result — real data shows a small early gain the uniform synthetic graph did not — but the practical conclusion is the same: **moderate, uncertain confidence in a future association is close to useless to this mapper; the benefit is concentrated in the high-confidence regime.**

**The practical upshot for tool builders**: a plasticity-aware mapper is not a cheap add-on that rewards partial profiling or soft priors over likely associations. It pays off when an engineer can declare associations with high confidence, and does close to nothing otherwise. This tempers Section 4.9's recommendation ("give the mapper the association structure") with a concrete requirement: the structure has to be known with confidence, not guessed at.

### 4.12 Generalization test 5: a published structural-plasticity rule

Section 4.4 substituted an externally measured *timing curve* for our own, but left the surrounding machinery intact: which candidate pairs were considered, and how they were scored, remained ours. Every number in this paper therefore still depended on one rewiring rule of our design — correlation, eligibility, and reward-attributed value accumulated on candidate pairs. This leaves a specific objection unanswered, and it is the growth-rule counterpart of the one Section 4.6 answered on the mapper side: is the penalty a fact about placement, or an artifact of how *our* rule chooses what to wire?

To test this we replaced the rewiring rule entirely with the homeostatic structural-plasticity model of Butz and van Ooyen (2013), a published model of activity-dependent cortical rewiring. Each neuron maintains axonal and dendritic synaptic elements whose number is driven by a Gaussian growth curve of that neuron's *own* firing rate relative to a homeostatic target; synapses form by randomly pairing available elements, with pairing probability falling off as a Gaussian in inter-neuron distance. The rule contains no correlation term, no reward or value term, and no representation of neuron type. It cannot, even in principle, prefer a cross-type partner. The network starts from zero edges and wires itself over 780 steps.

The comparison is structurally airtight in a way the earlier sections had to argue for rather than assert. Both conditions are generated from the same seeded construction, so they have *bit-identical coordinates* and identical per-type population counts (180 in each of five types); only which type occupies which fixed position differs. Every spatial quantity the rule can observe — inter-neuron distances, the pairing kernel, neighbourhood density — is therefore exactly equal across conditions, and the adjacency-mismatch failure mode documented in Section 4.5 is arithmetically impossible here rather than merely controlled for.

**Table 10.** Connectivity grown from zero under the Butz–van Ooyen rule, mean ± s.d. over twelve seeds, N=900. The chance ceiling is the cross-type fraction a type-blind rule produces by construction: with 180 of each of five types, a random partner is cross-type with probability (900−180)/(900−1) = 0.801.

| Condition | Final edges | Cross-type fraction | Taught pairs (0,3)+(1,4) |
|---|---|---|---|
| Segregated | 6581.3 | 0.367 ± 0.005 | 6.1 ± 5.0 |
| Interleaved | 5134.2 | 0.813 ± 0.006 | 926.6 ± 52.9 |
| *Type-blind chance ceiling* | — | *0.801* | — |

On the pre-registered primary endpoint — cross-type fraction of grown edges — the conditions separate at Welch p = 2.5×10⁻³⁷ (d = 82.0; difference +0.446, bootstrap 95% CI [+0.442, +0.450]). The effect survives a rule that knows nothing about types, values, or rewards.

**How this result should and should not be stated.** Interleaved's 0.813 sits *at* the type-blind chance ceiling of 0.801, not above it — and it cannot go above it, because a rule with no type-preference has no mechanism for exceeding chance. The ratio between conditions is therefore bounded near 2.2x by construction, however strong the underlying effect. The correct claim is the mirror image of the one a reader might expect: interleaving does not *enhance* cross-type wiring above chance; **segregation suppresses it to less than half of what a type-blind rule would produce by chance alone**, because same-type blocks dominate each neuron's spatial neighbourhood and the pairing kernel is local. We state this explicitly because the ratio invites the stronger and wrong reading.

Two further properties of this result are worth reporting. First, on the secondary endpoint — mass on the specific association pairs the benchmark rewards elsewhere — the gap is 152.3x (p = 2.1×10⁻¹⁵); we report it as secondary rather than primary because scoring only those pairs would import our own pattern schedule into a test whose purpose is independence from our design choices. Second, an asymmetry we do not have an explanation for: the segregated condition consistently grows *more* total edges (6581 vs 5134, a factor of 1.28). This is normalized out of the fraction-based primary endpoint, and it runs *against* the raw-count secondary endpoint, making that figure conservative rather than inflated — but it is unexplained, and we flag it rather than pass over it.

This test was pre-registered — endpoint, thresholds, and decision rule fixed before execution — and run on seeds disjoint from the exploratory pass that motivated it, which it reproduced closely (0.364→0.367 and 0.810→0.813). As with Section 4.7's confirmatory test, the predictions were derived from that exploratory pass and are not blind; the guarantee is that the endpoint and thresholds were committed before these particular seeds were run.

### 4.13 Generalization test 6: a second real speech corpus

Section 4.5 is the paper's most consequential result and rests on a single dataset. That leaves an obvious objection: the measured penalty may be a property of the Spiking Heidelberg Digits corpus specifically — its twenty digit classes, its speakers, its spike statistics — rather than of placement. We replicated the comparison on Spiking Speech Commands (SSC), the sibling corpus from the same release: thirty-five word classes rather than twenty digits, different speakers, the same underlying event-based encoding.

The simulation physics is untouched — placement construction and the simulation loop are imported verbatim from the Section 4.5 experiment, at the same validated life-length and decay regime (V retention 0.651) — so any difference is attributable to the data. Two upgrades over a single-corpus check were registered in advance. First, sixteen seeds per cell, on a seed range disjoint from the exploratory pass. Second, and more importantly, **three disjoint class pairs** rather than one, which answers the immediate follow-on objection that a single corpus result might rest on one unusually separable pair of words. The registered criterion required *every* pair to clear both statistical significance and a substantive floor of 3.0x.

**These numbers were re-derived after the wiring-draw correction of Section 4.5**, which applies here identically: this experiment imports the same simulation loop, and its structural audit shows the same pinned 313.0 ± 0.0 against 341.0 ± 0.0 excitatory imbalance. The values below are the corrected ones, from wiring resampled per seed in logical space. The withdrawn pre-correction figures were 9.89x, 10.47x and 16.07x, aggregate 11.61x [9.33, 14.99].

**Table 11.** Bridge mass by class pair on SSC, mean over sixteen seeds, adjacency-matched conditions, corrected wiring.

| Class pair | Segregated | Interleaved | Ratio | 95% CI | Welch p | d |
|---|---|---|---|---|---|---|
| 13 v 27 | 2235.0 | 6872.3 | 3.07x | [2.31, 4.26] | 7.1×10⁻⁹ | 2.89 |
| 10 v 25 | 2613.2 | 7326.8 | 2.80x | [1.82, 4.18] | 1.3×10⁻⁵ | 1.86 |
| 12 v 5 | 2209.6 | 6166.2 | 2.79x | [2.28, 3.54] | 2.2×10⁻⁸ | 2.80 |
| **Aggregate** | 2352.6 | 6788.4 | **2.89x** | **[2.36, 3.50]** | 6.4×10⁻¹⁹ | 2.32 |

**Only one of three pairs now clears the pre-registered 3.0x floor**, and we report that plainly rather than re-baselining it. The 13 v 27 pair passes at 3.07x; 10 v 25 (2.80x) and 12 v 5 (2.79x) fall below. By the decision rule registered before this experiment ran, one pass in three scores as *refuted as a general SSC result*, and on the letter of that rule the "not a lucky word pair" conclusion is not established.

Two considerations argue against reading that as a substantive refutation, and we state them without pretending they nullify the registered rule. All three pairs remain highly significant (p ≤ 1.3×10⁻⁵) with large effect sizes, so the *effect* is present in every pair; what fails is a magnitude threshold, not a direction. And the 3.0x floor was itself derived from the pre-correction SHD range — a number the same defect inflated — so the bar was set with contaminated values. Measured against the corrected SHD benchmark of 2.74x, all three pairs sit at or above the first corpus. A reader who holds us to the registered rule as written should conclude the pair-generality claim is unproven; a reader who accepts that the threshold was miscalibrated by a defect discovered later should conclude the effect replicates consistently across pairs at a lower magnitude. We think the second is correct and the first is defensible, and we decline to hide the tension.

**Cross-corpus agreement is now stronger, not weaker.** The corrected SSC aggregate [2.36, 3.50] overlaps the corrected SHD interval [2.27, 3.41] closely — 2.89x against 2.74x. Before correction the two corpora differed substantially in point estimate (11.61x against 8.22x) and the overlap claim rested on wide intervals; after correction they nearly coincide. The conclusion the section was written to support — the same effect at the same order of magnitude on a second corpus — survives the correction and is better supported by the corrected numbers than by the withdrawn ones.

Two caveats remain from the original design. We used SSC's `valid` split rather than its canonical train/test split — a download-size decision, and a real deviation from the corpus's benchmark protocol, though one that does not threaten the internal validity of a comparison in which both conditions see identical stimuli. And three class pairs out of thirty-five available classes is a small sample of the corpus. *(The zero-valued cell and the rounded-cache caveat reported in an earlier version applied to the withdrawn pinned-wiring run; the corrected run was cached at full float precision and contains no zero-valued cells.)*

### 4.14 A third-party boundary condition: where the penalty does *not* appear

Every result above is ours. This section is not: it was measured independently, on physical hardware we have no access to, by Henry Shulayev Barnes at Catalyst Neuromorphic, and is reported here with his permission and at his attribution. It is the only measurement in this paper produced by someone other than the author, and it is a negative result. We include it because a boundary condition — a case where the effect is absent, and a mechanistic account of why — constrains the claim more usefully than another confirmation would.

Catalyst's N4 architecture cannot express the central claim directly: structural plasticity was cut from its tapeout, so the comparison available was inference-only. It is also a substrate on which placement cannot affect accuracy at all, because spike delivery is exact and a neuron performs identical arithmetic wherever it sits. The only quantity placement moves is routed traffic, measured as **multicast fan-out** — the number of distinct cores a spiking neuron's events must be delivered to. The measurement was taken from the compiled delivery tables, so it is the mapping the hardware would actually execute rather than a simulation of one.

The model is Catalyst's deployed 48-core SHD build: 1,024 recurrent adLIF hidden units plus 20 readout, 1,486,656 quantised connections across 1,724 distinct sources and 47 occupied cores — 84.2% of the dense matrix. Four placements were compared on the identical connection set: the compiler's own first-fit-by-fan-in placement, a round-robin interleave, contiguous blocks, and uniform-random assignment.

At full density there is nothing for placement to do: every one of the 1,724 sources reaches all 47 occupied cores, with minimum, mean and maximum fan-out all exactly 47. The informative part is what happens under pruning.

**Table 12.** Mean cores reached per source, N4 deployed model under magnitude pruning. Measured by Catalyst Neuromorphic, September 2026, and reproduced here with permission.

| Sparsity | Edges | Deployed | Interleaved | Blocked | Random |
|---|---|---|---|---|---|
| 0% | 1,486,656 | 47.0 | 48.0 | 48.0 | 48.0 |
| 50% | 730,934 | 45.8 | 47.6 | 47.5 | 47.4 |
| 80% | 293,061 | 34.6 | 38.4 | 38.3 | 38.2 |
| 90% | 148,512 | 23.9 | 26.6 | 26.6 | 26.5 |
| 95% | 73,433 | 15.3 | 16.8 | 16.8 | 16.7 |
| 98% | 29,627 | 8.5 | 9.2 | 9.1 | 9.2 |
| 99% | 14,679 | 5.1 | 5.5 | 5.5 | 5.6 |
| 99.5% | 7,363 | 3.0 | 3.3 | 3.1 | 3.3 |
| 99.9% | 1,462 | 0.7 | 0.8 | 0.8 | 0.8 |

Blocked, interleaved and random placements sit within 0.2 cores of each other at *every* sparsity level from 0% to 99.9%. A deliberately segregated arrangement — the one that costs 3.8x–4.2x on our benchmark — costs nothing measurable here. The deployed placement is modestly better than all three, but not because it exploits community structure: it is first-fit by fan-in, which correlates with which neurons happen to retain edges under magnitude pruning.

**What this does and does not say.** It would be a serious misreading to take this as "placement does not matter on real hardware." The correct reading is the opposite in emphasis, and it is the collaborator's own: a backpropagation-trained recurrent network, even stripped to a thousandth of its weights, carries **no community structure for a placement to exploit**. Sparsity alone does not create locality. The placement penalty requires connectivity in which locality is already present — which a local structural-growth rule produces as a matter of course, and which magnitude pruning of a densely trained network does not produce at any sparsity level tested.

This sharpens the scope of the paper's claim in a way we could not have established from our own simulations, and it answers a reviewer's most natural objection — that the effect is simply what any sparse network exhibits — with third-party measurement on hardware rather than with argument. It also bears directly on the baseline-fidelity question of Section 4.6: our segregated condition is a fair baseline on a plastic substrate and would be a strawman on a trained-and-pruned one. The two claims are consistent, and the distinguishing variable is the same one Section 4.6 identified — the structure of the connectivity graph, not the optimizer or the hardware.

### 4.15 The endpoint this paper is named for: classification accuracy

Every result above is structural. Taught mass, bridge mass and cross-type edge fraction all measure what the network *wired*, not what it can *do*. This paper is titled for learnability, and the step from "less associative structure forms" to "the network learns less" was, until this section, an inference rather than a measurement. A reader is entitled to ask for the measurement, and we had not made it.

**Design, registered in advance.** The readout is the crux, because it can be rigged in either direction: a decoder allowed to read input-type neurons classifies the stimulus directly and would return a null regardless of placement, while a decoder reading only the specific association pairs the plasticity rule targets would assume the conclusion. We registered the readout as the per-neuron spike counts of the sixty **output-type** neurons only, decoded by a closed-form ridge classifier with a fixed regularisation constant, no per-condition tuning, z-scored on training-fold statistics alone, evaluated by stratified cross-validation on **held-out samples the plastic phase never saw**. Each placement is run twice, with plasticity active and frozen, so the primary quantity is the accuracy that plasticity *adds* rather than what was already decodable from the stimulus.

**A confound found by the manipulation check, and the harness defect it exposed.** The first execution failed its own pre-registered P1: frozen accuracies differed by 5.97 percentage points against a 5.0 point tolerance, meaning the conditions were unequal *before any learning*. Input→output edge counts matched (419 against 418) but excitatory counts did not (313 against 341). Two candidate fixes were tried and the check rejected both. The third succeeded, and identified the deeper problem: drawing wiring and inhibition over **logical** identities — (type, within-type rank) — and mapping through each placement, rather than over global neuron indices. That fix is what later exposed the pinned-draw defect corrected in Sections 4.1 and 4.5, so this experiment's manipulation check is the reason the rest of this paper's real-data numbers are now right.

**Table 13.** Held-out classification accuracy, forty seeds per condition, corrected wiring. Chance is 50%.

| Placement | Plastic | Frozen | Plasticity-attributable gain |
|---|---|---|---|
| Segregated | 81.25 ± 12.16 | 56.96 ± 5.81 | **+24.29 ± 14.45** |
| Interleaved | 87.69 ± 6.93 | 56.46 ± 5.06 | **+31.23 ± 9.07** |

Pre-learning decodability is now matched to 0.50 points. The plasticity-attributable gain differs by **6.94 percentage points, Welch p = 0.0124**, at a sample size chosen by a power calculation from the failed run's observed variance. Plasticity buys a large accuracy gain in *both* conditions, confirming the harness genuinely learns and the readout is sensitive; segregation reduces that gain.

**What this establishes, stated at its true size.** Segregated placement does cost held-out classification accuracy, on data never seen during learning, at matched pre-learning decodability. The paper's central inference is now measured rather than assumed. But it should be stated quantitatively rather than leaned on rhetorically. Cohen's d is 0.57 — moderate, against effect sizes above 8 elsewhere in this paper. Observed variance exceeded the planning assumption, so achieved power fell below target and the magnitude estimate is imprecise. Most importantly, on these same runs the *structural* ratio is 3.59x while the *accuracy* ratio is 1.079x. The structural penalty is large; the accuracy penalty is real, directionally consistent, and much smaller in relative terms. "Placement determines learnability" is earned as a directional claim about a measurable cost. It is not earned as a claim that the large structural gap translates into a comparably large accuracy gap, and we do not make that claim.

---

## 5. Discussion

Taken together, Sections 4.3-4.5 constitute three independent,
increasingly demanding tests of the same core claim, each addressing
a distinct objection a careful reader would raise: is it really about
space, or just locality (4.3); does it depend on our specific
learning rule (4.4); does it survive real data (4.5). The effect
persists through all three, with the magnitude varying by test in
ways we believe are individually explicable rather than arbitrary.
Sections 4.12 and 4.13 extend the second and third of those tests to
their stronger forms — replacing the structural-plasticity rule
entirely with a published model rather than substituting one timing
curve, and replicating on a second speech corpus across three
independent class pairs rather than one — and the effect persists
through both.
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
measuring the objective a mapper actually optimizes. The
association-aware placement achieves the lowest energy per unit of
learned structure of any condition tested and beats our segregated
baseline on both axes at once, but it does not dominate outright: against
interleaving it spends 2.5% less energy for 0.3% less learning. An
earlier version of this paper claimed unqualified Pareto dominance; that
claim did not survive the correction described in Section 3.4.

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

### 5.1 Practical recommendations for mapping tool design

The results in Sections 4.1-4.11 support a concrete, actionable set of design rules for anyone building or configuring a mapper for plastic neuromorphic hardware, stated here as a synthesis rather than scattered across individual findings.

**Optimize coverage first.** Whether every population has *at least one* reachable partner of its true associated type is the dominant lever — it accounts for the large majority of variance in learning outcome across every placement family we tested (Sections 4.7-4.8). A mapper with a limited constraint budget should spend it here before anywhere else.

**Within a fixed coverage budget, prefer compact, isolated clusters over sparse connected structure.** Confirmed twice: once by a matched-count/coverage confirmatory test (Section 4.10, 1.81×) and independently by a systematic search over eighteen alternative shapes calibrated to the same coverage level, none of which closed the gap — including one candidate with 3.6× more raw local connectivity that still lost by 2.4×. Grouping correlated populations into self-contained clusters outperforms spreading them along a thinly-connected network, even at identical local-opportunity statistics.

**Only use association foreknowledge when it is high-confidence.** Section 4.11 shows partial or uncertain knowledge of future associations is close to worthless to an association-aware mapper, replicated on both a synthetic and a real-data association graph. Below high confidence, default to coverage-maximizing interleaving rather than encoding an unreliable prior — the engineering cost of a soft-confidence mechanism is not repaid until confidence is close to certain.

**Sparse long-range bridging is not a cheap substitute for real redundancy.** A small embedded-neuron budget does not deliver an outsized return; recovery of the placement penalty tracks the size of the budget roughly proportionally rather than paying off early (Section 4.10). A token long-range-connectivity gesture should not be expected to approximate the benefit of proper co-location.

**Treat placement as a one-time, front-loaded decision.** We tested three physical self-organization mechanisms by which a substrate might, in principle, correct a poor initial placement on its own — value-driven migration, a wide independent sensing signal, and the same with simulated-annealing-style exploration — and none rescued a segregated starting configuration (Section 4.10 references this thread; full detail in the project's supplementary record). Structural (synaptic) plasticity is real and already does substantial work in this substrate, but physical placement is not something it repairs for you. A mapper's placement decision should be treated as final at map time, not deferred to runtime adaptation.

**There is no material energy tradeoff to weigh against any of this.** The energy cost of the best-learning placements is at most 1.6%, and the association-aware condition is energy-neutral-to-better than every alternative tested (Section 4.9). An objection along the lines of "correct placement costs too much power" is not supported by anything measured in this paper.

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
placement — its plasticity-attributable learning is net-negative, the
same regime as our segregated baseline, not a diluted version of
interleaved's gain. Our hand-rolled baseline is therefore a
reasonable stand-in for a deployed tool's output on a plastic
substrate, not a strawman, though it remains somewhat less severe on
raw taught mass (SpiNeMap's placement is 1.25x above ours there, and
unlike ours leaves roughly 70% of correlated neurons with at least one
partner in reach) despite landing in the same net-loss regime on the
corrected growth metric. The decisive variable is the graph the mapper
is given: with the to-be-learned associations declared up front, the
same tool lands at interleaved. A previous version of this section drew
the opposite conclusion from a partitioner that had silently failed on
the input the placement path uses; the error, its cause, and the fix
are documented in Section 4.6. That episode is itself a limitation
worth stating: these results rest on a single implementation, and one
load-bearing component of it was wrong for a period without any result
looking anomalous. A second, unrelated implementation issue in the same
family was found and corrected on 2026-08-29 (Section 4.1, Section 4.6):
the shared benchmark harness's raw "taught mass" metric was diluted by
a non-mechanistic random-initialization baseline common to every
placement; the corrected, delta-based metric is now primary throughout.
The fabric-sparsity sweep (Section 4.8), the coverage-mediator work
(Section 4.7 and its confirmatory test), the energy-learnability
frontier (Section 4.9), and the N=5,000 replication all depend on the
same harness and metric and have now been reanalyzed under the
correction as well (same day). Every one of them held or strengthened:
the coverage-vs-count mediator comparison is essentially unchanged
(R^2 0.837 raw vs. 0.829 corrected, both far above count's ~0.27); the
fabric-sparsity crossover is intact, and a marginal penalty the
association-aware mapper appeared to show at the tightest tested pitch
on raw taught mass (p=0.0002) is NOT significant on the corrected metric
(p=0.12) — the mapper is more robust to fabric density than the original
raw-metric report suggested, not less; the energy frontier's dominance
results sharpen into sign flips against segregated and the population-
graph mapper. No result in this family reversed under the correction; in
every case the corrected number was as large as or larger than the raw
one, consistent with the raw metric diluting rather than manufacturing
the effects reported here.

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
produces different-magnitude penalties in different geometries (a
1.32x raw-mass ratio in the confirmatory test of Section 4.7 against
Section 4.6's SpiNeMap result, though the latter's own metric is under
revision — see the correction noted above), so the metric predicts
direction and ordering reliably but not calibrated magnitude. A mapper
could use it as a constraint; it should
not yet be used as a predictor of how much learning a given placement
will cost.

**Task and scale realism.** All synthetic results (Sections 4.1-4.4
and 4.12) use hand-designed stimulus patterns at N=900-5,000.
Experiment 33's architecture proportions were drawn from a published
hardware design, but its input remained synthetic. Only Sections 4.5
and 4.13 use real data, and at a substantially smaller scale (N=200)
than the synthetic benchmarks.

**Effect size interpretation.** As stated in Section 3.4, several
reported effect sizes (Cohen's d > 70 in Section 4.1) are far larger
than typical in empirical research and are a property of low-noise
synthetic simulation. The statistical significance of the *direction*
of the effect is not in question; its precise magnitude should not
be over-interpreted from simulation alone, and the real-data result
in Section 4.5 — while itself imperfect — is a better guide to
expected real-world magnitude than the synthetic benchmarks.

**Independent replication.** All results in this paper except
Section 4.14 were produced by a single implementation and a single
author. No independent reimplementation has replicated the central
positive finding, and that remains the single largest gap in this
work. The one third-party measurement we have — Catalyst
Neuromorphic's N4 fan-out study, Section 4.14 — is a negative result
that bounds the claim's scope rather than confirming it, and it could
not test the central claim in any case: that chip's tapeout does not
include structural plasticity, so the comparison available was
inference-only. It should be counted as an independent *boundary
condition*, not as independent corroboration.

No result in this paper has been produced on hardware performing
runtime structural plasticity, which is the substrate the entire
claim concerns. We surveyed the available routes and report the
outcome plainly, since it explains why: Intel's Loihi cannot express
structural plasticity at all — its learning API operates on weight,
delay and tag, with no synapse creation or destruction — and stock
NEST's structural plasticity selects partners without any spatial or
topological term, so neither can host this experiment regardless of
access. SpiNNaker can, and has had distance-dependent structural
plasticity in its maintained software stack since Bogdan et al.
(2018). We note in advance, and before running anything, that
SpiNNaker's candidate *selection* is uniform-random while only its
formation *acceptance* is distance-dependent — one of the two
mechanism channels identified in Section 4.10 — so a smaller effect
than our simulations report is the correct prediction there, and we
register it as such rather than retrofit an explanation afterwards.

**The pinned-wiring defect, and what it says about the rest of this paper.** The correction in Sections 4.1 and 4.5 is the most serious error found in this work, and its character matters more than its magnitude. A single generator, `default_rng(7)`, supplied initial connectivity and excitatory/inhibitory assignment for every run of every real-data experiment, independent of the per-run seed. Multi-seed sweeps therefore resampled the *simulation* while holding the *network* fixed, and reported intervals that quantified only the former. The defect was invisible to every check we had: results were reproducible, seed-stable, survived a numpy-version audit, and passed the deployed-code audit — because all of those verify consistency, and the numbers were perfectly consistent. They were consistently wrong. The corrected interval is narrower than the withdrawn one, so nothing about the published statistics looked suspicious; the estimate was simply centred on the wrong value. We record this because the lesson generalises beyond this paper: reproducibility checks cannot detect a variance source that is held constant, and an interval that never varies a factor is not an interval over that factor. Readers should treat every interval in this paper as conditional on the factors actually resampled, and the experiments importing the synthetic harness (Sections 4.7-4.11) as carrying optimistic error bars pending re-derivation, though their point estimates are unaffected.

**Real-data result variance and scope.** Segregated-condition outcomes remain intrinsically high-variance (relative standard deviation above 50%) even at twenty seeds, and that variance is now larger, not smaller, than earlier drafts reported, because resampling the wiring restored a source of variation the pinned draw had suppressed. The training-exposure question raised in earlier drafts is now open rather than answered: the longer-exposure regime was measured only under the pinned draw and has been withdrawn rather than re-derived, so we do not currently know whether the effect's magnitude varies with exposure. We consider the *direction* of the real-data result — segregation harms learning on real speech, substantially and reliably, across two corpora and now in classification accuracy as well as structure — solidly established. The magnitude should be read as approximately 2.7x-2.9x at the one regime that has been properly measured, not as a range.

**The two newest generalization tests, and what they do not establish.** Section 4.12 replaces our rewiring rule with a published one, closing the growth-rule counterpart of the baseline-fidelity objection above — but three limits apply. Our reimplementation simplifies the published model: synaptic element budgets are recomputed per epoch rather than bound and unbound incrementally per element as in the original, and a single homeostatic target rate is calibrated once and shared across both conditions. The total-edge asymmetry (segregated grows 1.28x more edges) is normalized out of the primary endpoint and runs against the secondary, but it is uncharacterized and we do not know its cause. Most importantly, that test establishes that the *direction* of the effect is rule-independent; because the interleaved condition is pinned at the type-blind chance ceiling by construction, it cannot speak to whether the *magnitude* measured under our own rule is correct. Section 4.13's second-corpus replication used SSC's non-canonical `valid` split, tested three class pairs out of thirty-five available classes, and resumed an interrupted run from a partially rounded cache (quantified in that section). Both tests were pre-registered replications of exploratory passes rather than blind predictions: endpoints and thresholds were fixed before the reported seeds were run, but the expected direction was already known from the exploratory work, and both inherit the single-implementation caveat that applies to everything in this paper.

**Section 4.10's three checks, individually.** The sparse-bridge result tested only individually-scattered embedded neurons; a structured bridging scheme was not tried, and we do not claim the naive version tested exhausts what "sparse long-range connectivity" could mean on this substrate. The connectivity-topology result's exploratory pass used approximately-matched count and coverage between filament and blob families and reported a magnitude (net-negative crossover) that the confirmatory pass, with tighter matching, did not reproduce; we report the confirmatory number (1.81x at matched coverage/count) as the trustworthy one and flag the larger exploratory number as superseded, not as an additional independent finding. Both the mechanism-validation and topology results are single-implementation findings at N=900 only, subject to the same independent-replication caveat as the rest of this paper.

**Section 4.11's partial-knowledge model and its own near-miss.** Partial knowledge was operationalized as a continuous *weight* on the true cross-type affinity SpiNeCluster is given, not as certain knowledge of a *subset* of the true association edges — a mapper that is fully confident about a fraction of the true associations and blind to the rest was not tested, and could plausibly behave differently (closer to the synthetic "population" endpoint applied to a smaller effective graph than to a uniformly-diluted one). The real-data version of this section also stands as a within-paper demonstration of the risk it warns about elsewhere: an initial single-placement-seed run appeared to overturn the synthetic threshold finding outright, and only a second, then third, more heavily-seeded rerun showed that result was placement-search noise, not a real effect. We report the corrected result and the process by which it was reached rather than only the final number, since the intermediate false positive is itself informative about how much placement-seed averaging this class of experiment needs before a result should be trusted.

---

## 7. Conclusion

Placement determines learnability on plastic neuromorphic hardware,
not merely communication energy. This effect is robust to removing
physical geometry entirely, to substituting an externally measured
learning rule for our own, to replacing our structural-plasticity rule
outright with a published model containing no correlation, value, or
reward term, and — with important caveats disclosed above — to
replacing synthetic stimulus patterns with real human speech, on two
independent speech corpora. It is measurable not only in connectivity
but in task performance: segregated placement costs held-out
classification accuracy at matched pre-learning decodability
(Section 4.15). Its magnitude is smaller than earlier drafts of this
paper claimed — a pinned random draw inflated the real-data figures
roughly threefold, and the corrected values are reported throughout —
and a third-party measurement on physical hardware (Section 4.14)
bounds where it applies at all: the penalty requires connectivity that
already carries locality, and is absent in a trained-and-pruned network
that has none. What survives correction is a directional effect that is
large, consistent across rules and corpora, mechanistically explained,
and reproduced by a real deployed mapping tool.
We believe this constitutes a genuine, previously
unaddressed design variable for neuromorphic hardware mapping, and we
report both the strength of the evidence for it and the specific
respects in which further validation, particularly independent
replication and physical hardware testing, remains necessary before
the finding should be considered fully established.

---

## Acknowledgements

The fan-out measurements in Section 4.14 were designed, run and
supplied by **Henry Shulayev Barnes (Catalyst Neuromorphic)**, on
Catalyst's N4 hardware, and are reproduced here with his permission.
He also identified the framing point that section now leads with — that
the result constrains *when* the placement penalty applies rather than
whether it is real — and independently raised the chance-ceiling caveat
recorded in Section 4.12 before we stated it. The interpretation, and
any error in it, remains ours.

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

6. Butz, M., van Ooyen, A. "A simple rule for dendritic spine and
   axonal bouton formation can account for cortical reorganization
   after focal retinal lesions." *PLoS Computational Biology* 9(10):
   e1003259 (2013). [Model reimplemented in Section 4.12 — verify
   exact volume/article number against the original before
   submission.]

7. Cramer et al. (2022), reference 1 above, also releases the Spiking
   Speech Commands (SSC) corpus used in Section 4.13.

8. Shulayev Barnes, H. "Multicast fan-out against placement, N4
   48-core deployed model." Catalyst Neuromorphic, 1 September 2026.
   Private communication, cited with permission (Section 4.14).

9. Bogdan, P. A., Rowley, A. G. D., Rhodes, O., Furber, S. B.
   "Structural Plasticity on the SpiNNaker Many-Core Neuromorphic
   System." *Frontiers in Neuroscience* 12, 434 (2018).
   doi:10.3389/fnins.2018.00434. [Cited in Limitations regarding
   available hardware routes; note this is also prior art for
   distance-constrained runtime synapse formation.]

*Note: this reference list is incomplete relative to the full related-
work discussion in Section 2. Citations for the cognitive-science
interleaving/spacing-effect literature and for specific neuromorphic
hardware platforms referenced in Section 2 require verification
against original sources before this draft is submitted anywhere.
This is flagged explicitly rather than filled with unverified
citation details.*
