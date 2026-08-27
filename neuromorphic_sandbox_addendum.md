# Addendum: From Static Placement to Living Substrates

## Experiments 12–19d, the Statistical Audit, and a Revised Synthesis

**Author:** Bo Wills (with Claude as research collaborator)
**Date:** August 9, 2026
**Extends:** *Spatial Topology as a First-Class Design Variable* (Experiments 1–11, August 8, 2026)
**Environment:** Same pure-NumPy testbed; lives of 400–1,400 steps; five-seed replication of all load-bearing claims (§7)

---

## Abstract

The original report established, on a 1,000-neuron testbed, that physical placement is a first-class design variable for a *fixed* network. This addendum reports two further experimental arcs on the same testbed, in which the network is no longer fixed. Arc 2 (Experiments 12–17) makes the *structure* adaptive: connections rewire under activity-dependent rules, and in later experiments the components themselves migrate. Arc 3 (Experiments 18–19d) makes *retention* adaptive: structure persists according to reward rather than raw co-activity, implemented as an eligibility-trace × broadcast-scalar (three-factor) rule and subsequently as reward-prediction error (RPE). Principal findings: (i) anatomy decomposes into a world-term and a body-term that are experimentally separable — activity-dependent wiring encodes the statistics of experience *and* the geometric priors of the substrate, and the two can be dissociated by re-running an identical life on a reshuffled body; (ii) substrate geometry governs the *retention* of learned structure, not its acquisition; (iii) interleaving functional identities across physical space ("salt-and-pepper") and clustering matter are orthogonal levers — the former controls learnability (+45% mean across seeds), the latter controls wire energy (−43%) — and their combination outperforms every hand-designed layout tested; (iv) organizing a plastic substrate with structured spontaneous activity *before* experience arrives ("dreams") recovers 97% of the performance of an optimally designed body from an adversarially designed one; (v) value-gated retention exhibits, as bookkeeping consequences rather than added assumptions: near-total selectivity between equally-frequent associations (19.2x ± 10%), sunk-cost persistence of obsolete structure (5/5 seeds), superstitious credit assignment funded by volatility, an asymmetry in which structural destruction outpaces structural recovery 3–5x, and the requirement that recovery be driven by computed relief rather than mere absence of harm. All five load-bearing claims across both reports survived pre-registered five-seed replication (§7). We close with a revised synthesis and three candidate design principles for adaptive hardware that a preliminary literature search did not locate in the neuromorphic-systems literature.

---

## 1. What Changed After Experiment 11

The original report's testbed measured *fixed* anatomies: placement, thermal, and traffic properties of networks whose wiring we specified. Experiments 12 onward invert the arrangement — the testbed specifies physics and experience, and the network specifies its own anatomy. Two mechanisms were added incrementally:

- **Structural plasticity** (Exp 12+): a fixed budget of 10,000 directed edges; every 40 steps, the 500 edges with the lowest retention score are pruned and 500 new edges are grown at the highest-scoring candidate pairs. The retention/growth score is the experimental variable: raw co-activity (a decaying coincidence matrix C), co-activity discounted by wire cost (C/(1+0.05d²)), accumulated value (V), or RPE-updated value.
- **Component migration** (Exp 15+): neuron positions become state. Each epoch, neurons step toward the co-activity-weighted centroid of their partners (attraction) and are displaced out of over-occupied cells (repulsion), with step size optionally scheduled over developmental time.

Methods otherwise follow the original report (LIF dynamics, 20% inhibition, three stimulus patterns on a 60-step cycle). One deliberate simplification carried through both arcs and flagged here once: coincidence is same-timestep and symmetric — a cartoon of STDP without causal asymmetry. Findings below should be read as properties of *correlation-gated* structural plasticity; the causal-timing refinement is future work.

## 2. Arc 2a: Experience Writes Structure — and So Does the Body (Experiments 12–13)

**Exp 12** initialized random wiring (a "newborn") and applied pure co-activity rewiring under a patterned world in which cluster pairs (0,3) and (1,4) always co-activate. The taught pairings became legible in the anatomy — their inter-cluster edge counts reached 2–3x matched controls — but wire energy improved only 17% (5.48M → 4.55M against a designed reference of 16,183): fire-together-wire-together contains no concept of distance, and implemented its learned truths as maximally expensive wiring.

**Exp 12B** added the wire-cost discount to candidate scoring, on the argument that in physical substrates the locality of growth is enforced by the medium (axons extend through space) rather than computed by the rule. Energy then fell twice as fast and locality rose past 50% — but the anatomy developed a spurious feature: the strongest bridge in the final network (C1↔C3, ~1,140 edges) connected clusters that were **never co-stimulated**. The die's geometry (seed-fixed since Exp 1) had placed C1 and C3 adjacent; mediocre correlation at trivial cost outcompeted strong correlation at high cost.

**Exp 13** tested whether this was geometry by control: the identical life (same stimulus schedule, same life-seed) re-run on a reshuffled die. Result — clean dissociation:

| Bridge | Taught? | Body A | Body B | dist A | dist B |
|---|---|---|---|---|---|
| C1–C4 | yes | 556 | 551 | 31 | 15 |
| C0–C3 | yes | 626 | 822 | 14 | 9 |
| C1–C3 | no | **1,144** | 398 | **5** | 19 |
| C0–C4 | no | 379 | **1,368** | 26 | **2** |

Taught bridges are body-invariant (C1–C4: 556 vs 551 across bodies, never adjacent in either). Spurious bridges track whichever pairs the geometry made cheap, and relocate wholesale when the geometry changes. **Anatomy = world-term + body-term, and the terms are separable by this two-body protocol.** The applied implication: connectivity read from any adaptive substrate (biological or artificial) confounds the statistics of its experience with the priors of its geometry, and the confound is resolvable by (in artificial systems) exactly this control.

## 3. Arc 2b: Geometry Governs Retention, Not Acquisition (Experiment 14)

Three hand-placed bodies lived one identical life: **aligned** (taught partners adjacent), **adversarial** (false partners adjacent, true partners across the die), **neutral** (equidistant ring). All three acquired the taught structure at identical rates for the first ~150 steps (~1,650 taught-edge mass). They then diverged monotonically: aligned *retained* (1,818, holding), adversarial and neutral *eroded* (~1,120, still falling at cutoff), because each rewiring epoch taxes expensive structure and the taught bridges were expensive everywhere except the aligned body.

Two corollaries. First, the adversarial body performed no worse than neutral: cheap-but-uncorrelated pairs received no co-activity support, so **geometric priors can starve truths but cannot fabricate falsehoods** under correlation gating — an asymmetry with safety relevance for adaptive hardware. Second, the aligned body won wire energy (1.57M vs 2.19M) *and* knowledge simultaneously: when body statistics match world statistics, efficiency and memory are the same physical quantity. We summarize the section's finding as: **geometry is a landlord, not a teacher** — it does not decide what is learned; it decides what can afford to persist.

## 4. Arc 2c: Migration, Its Failure Modes, and the Dream Cure (Experiments 15–17)

Allowing components to migrate (attraction toward correlation partners, repulsion from over-density) from an adversarial birth produced a taxonomy of outcomes as the constraint physics varied:

- **Collapse** (weak repulsion): four of five clusters merged into a radius-1 knot; the never-co-active cluster was excluded and atrophied. Wire energy became excellent and the learned structure re-expressed — a *degenerate* solution: geometry abolished rather than corrected. (The un-stimulated cluster's physical ostracism, emerging from two force rules, is noted as an unplanned reproduction of activity-dependent exclusion.)
- **Frustrated glass** (strong repulsion): cluster integrity preserved, but the false adjacencies of the adversarial birth remained and became load-bearing — no sequence of locally-favorable moves paid the transient cost of swapping clusters across the die. History froze into anatomy.
- **Stasis** (developmental schedule without signal, Exp 16): a hot-migration/cool-learning schedule modeled on critical periods *failed below both baselines* (940 vs frustrated 1,053) because the silent hot phase provided no correlation gradient — the network's most mobile era was wasted motionless, and the deprived loner cluster became the *richest* structure on the die (internal wiring 2,267), reproducing a deprivation pathology: plasticity with nothing to learn elaborates itself.
- **Contraction under dreams** (Exp 17): filling the hot phase with structured spontaneous activity ("retinal waves") rescued the schedule decisively. With waves rehearsing the true world statistics, the adversarial-born network reached **1,770 taught mass — 97% of the purpose-built aligned body** — via a fourth strategy: general contraction until the distance tax could no longer discriminate, rather than geometric rearrangement. The informed/uninformed control (waves rehearsing true pairings vs random pairings, equal activity) attributed ~93% of the benefit to the *presence* of structured pre-experience activity and ~7% to its *content* — with content buying two specific goods: a smooth dream-to-reality transition (no unlearning dip) and lower spurious structure (superstition 1,242 vs 2,240).

The arc's compressed lesson: **sequencing is not the mechanism; the coincidence of mobility and signal is.** A plastic substrate organizes correctly when its most mobile phase is supplied with activity bearing the statistics of its future workload — a procedure we refer to as *factory dreams* and identify in §8 as a candidate contribution.

## 5. Arc 3a: The Cosmological Detour and the Interleaving Result (Experiment 18)

Motivated by the observed morphological similarity between neural tissue and cosmic large-scale structure, we implemented cosmology's two-point correlation function ξ(r) (Peebles–Hauser DD/RR−1 estimator) as a structure-classification instrument, and swept the testbed's "constants" (attraction G × repulsion Λ) across 16 universes evolved from smooth initial conditions. Two results:

**Structure formation is generic.** The structured (filament–void) phase occupies a broad plateau of the constant space — gas requires attraction an order of magnitude below working values; collapse requires the strong-G/weak-Λ corner. The designed die's ξ(r) fingerprint (positive small-r excess, zero-crossing, negative void trough) is of the same family as galaxy-survey curves; we read the cortex–cosmos resemblance as shared membership in the universality class of attraction-under-anticollapse dynamics, requiring neither coincidence nor fine-tuning.

**The stowaway: identity interleaving.** The sweep's uniform identity-scattering caused every universe — including structureless gas — to out-learn every designed body from Arcs 1–2. A follow-up 2×2 audit (spatial arrangement × identity assignment, frozen bodies, full ledger) deconfounded the effect:

| Body | Taught | Wire energy |
|---|---|---|
| Columns (clustered, segregated) | 1,326 | 1,609,312 |
| **Salt-clustered (clustered, interleaved)** | **2,144** | 1,763,733 |
| Segregated-gas (uniform, segregated) | 981 | 2,811,144 |
| Salt-gas (uniform, interleaved) | 1,842 | 2,832,957 |

Interleaving wins learning in both spatial phases (+62% / +88%, single-seed; **+45% ± 6% across the five-seed audit**) at near-zero wire premium (+0.8–9.6%), because the rewiring rule implements each learned truth with whichever partner copies are nearby, and interleaving guarantees nearby copies of everything. Spatial clustering independently controls energy (−43%). **The levers are orthogonal: where the matter sits sets the energy; who sits where sets the learnability.** The best configuration — clustered matter, interleaved identity — we summarize as *clump the atoms, scatter the roles*. Segregation's residual asset is within-identity wiring density (~4,300 vs ~2,400 diagonal edges), i.e., it serves dense same-type computation; our taught-mass metric measures only cross-type association, and tasks weighting within-type computation would shift this ledger — flagged as the audit's open flank.

## 6. Arc 3b: Value-Gated Retention (Experiments 19–19d)

Replacing the retention signal with a three-factor rule — eligibility trace E (fast decay), broadcast reward R delivered one step post-pattern, value V += R·E gating prune and growth — on the salt-clustered substrate, against a correlation-rule control on the identical life:

**Selectivity (Exp 19).** Three patterns of identical frequency and intensity, differing only in value (+1 / 0 / −1). Correlation control: rewarded and neutral bridges statistically identical (ratio 1.09; audited 5-seed control mean ≈ 1.0). Utility rule: **18.7x** (audited 19.2x ± 10%). The neutral association — experienced as often as the rewarded one — was physically forgotten (753 → 184 edges). Punishment produced *active demolition*, not neglect: the punished cluster's internal wiring fell 348 → 3 against a control equilibrium of ~293, via monotonic prune-first ranking of negative-V edges. Two emergent pathologies: (i) *superstitious conditioning* — the never-co-active bridge adjacent to the rewarded pattern retained more structure than the genuinely-associated neutral bridge (403 vs 184), through noise co-firing inside reward windows; Skinner's phenomenon as a bookkeeping consequence; (ii) *cheap monomania* — 34% of all edges concentrated on the single valued association at *lower* total wire energy than the control, i.e., obsession is efficient, at the cost of anatomy that a changed world would require.

**Sunk cost (Exp 19b).** Reversing the economy at step 600 ((0,3)→0, (1,4)→+1): the obsolete structure *grew for 160 further steps* on stored value, was never overtaken within a 600-step window (5/5 audit seeds), and extrapolated crossover at ~step 1,430 — adaptation ~40% costlier than initial acquisition, consistent with extinction-slower-than-acquisition in the conditioning literature. Mechanism: prune-by-V cannot touch high-V stock; young edges are born at V≈0, the front of the prune queue, so the incumbent's ledger crowds out its successor. The superstitious bridge, adjacent to reward under *both* regimes, rose through the reversal (403 → 712): spurious credit is a survivor class.

**Prediction error (Exp 19c).** Paying V in surprise (δ = R − R̂, Rescorla–Wagner expectation R̂ per pattern) rather than raw reward produced regime change within the window (step 1,160; 5/5 audit seeds) and demolished the obsolete structure at twice the raw-rule rate. Three sharpened findings: disillusionment is metered by *re-encounter* — the V-stock of unrevisited associations is untouchable; the surprise economy defunds superstition in stable regimes (chance bridge 345 vs 403) but re-funds it under volatility (surge to 754 post-reversal): **spurious structure is financed by prediction-error flux**, the variable-ratio/gambling result as an identity; and punishment *flow* habituates (δ→0 as R̂→−1) while punishment *stock* persists — the phobia outlives its cause as an unforgiven ledger.

**Recovery (Exp 19d).** Removing the punishment at step 600 under three arms: RPE+therapy healed the demolished cluster 2 → ~93 edges — real, partial (one-third of the 293 natural baseline), plateauing, and relapse-wobbly; RPE-untreated stayed at floor; raw-rule+therapy stayed at floor (final: 4) despite passive debt decay, because growth slots are won by *positive* value — **acquittal is not advocacy**. Healing ran 3–5x slower than the original demolition, for an identified mechanistic reason: the destroyed assembly's residual activity carries small eligibility, so the repayments it can earn are whispers relative to the shouts that funded its destruction. Relief is additionally self-extinguishing — δ = 0 − R̂ shrinks as safety becomes expected — so exposure's active ingredient depletes because it works. Partial extinction with residuals and expectancy-violation as the engine are the textbook phenomenology of fear extinction, obtained here from four lines of bookkeeping.

## 7. The Statistical Audit

All headline claims were re-tested under pre-registered criteria (deliberately weaker than observed values, testing survival of the *phenomenon* rather than the number) across five seeds — geometry seeds for the placement claim, life seeds on fixed bodies for the rest. Harness: `src/audit.py`; wall-clock 1.5 minutes.

| Claim | Original | Audited (5 seeds) | Criterion | Verdict |
|---|---|---|---|---|
| Placement energy ratio (Exp 3) | 79x | 108.6x ± 4% | mean > 20x | **GRANITE** |
| Interleaving advantage (18c) | +62% | +45% ± 6%, all seeds > 1.0 | mean > 1.25x, all > 1 | **GRANITE** |
| Utility selectivity (19) | 18.7x (ctrl 1.1) | 19.2x ± 10% (ctrl ≈ 1.0) | mean > 5x, ctrl < 1.5 | **GRANITE** |
| Sunk-cost persistence (19b) | no crossover | 5/5 seeds | ≥ 4/5 | **GRANITE** |
| RPE regime change (19c) | step 1,160 | 5/5 seeds | ≥ 4/5 | **GRANITE** |

One honest correction surfaced: the interleaving effect's original single-seed magnitude (+62%) sat on the favorable side of its distribution (true center ~+45%). No claim inverted; one softened in size. The audit also revealed ~20x more compute headroom on the host machine than assumed, revising the scale-up ladder's cost estimates downward.

## 8. Revised Synthesis and Candidate Contributions

The original report's rulebook (cluster to the thermal boundary; sparse structured backbones; ~2 gateways; heterogeneous stacking only; provision inhibition) governed *static* fabrics. The two new arcs extend it to substrates that learn, and the extension is not additive but interactive — the static rules acquire new meanings:

1. **Placement is a prior on memory.** For a plastic substrate, layout does not merely set energy; it sets the maintenance economics that decide which learned structures survive (§3). Design placement from the expected correlation statistics of the workload, or —
2. **— dissolve the problem by interleaving.** Scattering functional identity across physical tiles makes every association locally implementable, decoupling learnability from layout at negligible wire cost (§5). *Clump the atoms, scatter the roles.*
3. **Commission plastic hardware with factory dreams.** A mobile/plastic phase supplied with synthetic activity bearing workload statistics organizes an arbitrarily mis-built substrate to near-optimal (§4). Presence of structured activity is most of the effect; matching content buys transition smoothness and purity.
4. **Value-gate retention, but with prediction error, and budget for its pathologies.** Raw reward gating produces efficient monomania, immortal sunk cost, and volatility-funded superstition; RPE fixes succession and defunds superstition in stable regimes but leaves trauma as stock and heals asymmetrically slowly (§6). Recovery requires computed relief; passive decay does not rebuild.

A preliminary literature search (neuromorphic placement, structural-plasticity hardware, deployment methodologies) located active work on on-chip plasticity platforms and hardware-aware training, and an explicitly acknowledged gap between computation and learning as co-designed processes, but did not locate principles (2), (3), or (1) as articulated design guidance for adaptive hardware. These are accordingly flagged as *candidate* contributions pending a proper review — with the explicit caveat that two searches constitute reconnaissance, not due diligence.

## 9. Limitations and Immediate Program

All prior limitations stand; new ones join them: same-step symmetric coincidence (no STDP causality); single reward source and three-pattern worlds (the monomania and superstition results should be re-examined under richer reward ecologies); migration physics with hand-set constants; the interleaving audit's within-type blind spot (§5); and scale (n=1,000 throughout — the audit's five seeds harden statistics, not scale). The immediate program, in order: sparse-engine rewrite with the audit as regression suite; the 1K/10K/100K ladder re-running every granite claim; ξ(r) and network-statistic comparison against the measured *Drosophila* connectome at the 100K rung; re-expression in Lava's CPU backend; INRC application with this document attached.

## 10. Coda

The original report closed by arguing that the prerequisite for contributing to architecture exploration may be shifting from credential-gated knowledge toward the ability to pose falsifiable questions and audit models. This addendum is evidence in both directions at once: every arc originated in the non-specialist author's conjecture (that substrate physicality co-authors learning; that the cortex–cosmos resemblance is lawful; that retention should follow utility), every arc was operationalized conversationally, and every load-bearing result then survived pre-registered replication. The findings themselves converge on a matching claim about the systems studied: in an adaptive substrate, *what the machine is shapes what the machine learns* — geometry taxes memory, interleaving democratizes it, dreams prime it, and value both builds and scars it. A research program that began by asking where to put the wires ended by measuring the anatomy of forgetting and the price of recovery, on the same laptop, with the same 90-line physics. We take that as the strongest available argument that the design space of learning hardware is legible to anyone willing to interrogate it honestly.

---

*Code, notebooks, audit harness, and full experimental records: `neuromorphic-sandbox` repository, Experiments 1–19d + audit, August 8–9, 2026.*
