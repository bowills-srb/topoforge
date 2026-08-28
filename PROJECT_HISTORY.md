# TopoForge — Project History & Working Conventions

This file supplements CLAUDE.md's auto-generated structural notes with
context that only exists in the project's history, not in the code
itself. Read this before making changes.

## What this project is, in one paragraph

TopoForge studies whether the physical placement of neurons on a
neuromorphic chip affects what the chip can learn — not just its
energy use, which is what every existing chip-mapping tool optimizes
for. The core finding: placing functionally-different neurons close
together ("interleaved") lets structural plasticity build cross-type
associations that placing them in function-grouped blocks
("segregated") makes much harder or impossible. This has now been
confirmed at multiple scales, with geometry removed entirely (pure
graph locality), and on real spike-encoded human speech data — not
just synthetic patterns.

## Two repositories — know which one you're in

- **`topoforge`** (public, github.com/bowills-srb/topoforge) — the
  main research repo. 37+ numbered experiments in `src/experiments/`,
  the core engine (`engine.py`, `engine_local.py`, `engine_fast.py`,
  `sparse_state.py`, `spatial.py`), the preprint draft, the CLI
  product (`topoforge_run.py`). **This repo needs to stay stable and
  citable** — it's what a collaborator (Henry, at Catalyst
  Neuromorphic) may test against, and what backs the filed provisional
  patent. Don't break working experiments; if you change shared engine
  code, re-run the affected experiments to confirm results still hold.

- **`topoforge-scale`** (private) — a from-scratch rewrite aimed at
  running the same physics at N=500,000 instead of N~1,000-5,000. Has
  its own `ScaleLife` engine, Numba-compiled kernels
  (`numba_kernels.py`, `numba_sparse_state.py`), and its own audit
  suite. This repo is allowed to be messy and in-progress. It only
  merges back into `topoforge` after full equivalence-audit passage.

## The single most important convention: audit before you trust

Every non-trivial change in this project's history has been validated
by comparing new code against known-good output on identical inputs
*before* it was trusted for anything else. This caught real,
consequential bugs — not hypothetical ones. Specific examples:

- Replacing the dense N×N distance matrix with sparse on-demand
  distance computation was proven bit-identical at N=500-5,000 before
  it touched the real engine.
- The Numba-rewritten sparse state (`NumbaSparsePairState`) was
  checked against the original dict-based `SparsePairState` on a
  shared shadow-test (dense matrix as ground truth) before integration.
- A full engine rewrite (`ScaleLife`) was checked against the
  known-good engine on 5 seeds, exact match required, before any
  scale claims were trusted.

**When you change engine code, write or reuse an equivalence test
before believing the new numbers.** Don't skip this because a change
"looks obviously correct" — several bugs tonight looked obviously
correct until checked.

## Known gotchas — bugs that cost real time, now documented so they don't recur

1. **Rewire trigger conditions must be checked for accidental
   always-true logic.** `(t // 40) % 1 == 0` is *always* true (mod 1
   is always 0) — this silently made a rewire step fire on 62% of
   steps instead of once per 40, and elsewhere, nesting the rewire
   block inside a reward `if` block made its trigger condition
   mathematically unreachable. Always sanity-check trigger frequency
   empirically (print a counter), don't just read the condition and
   assume it's right.

2. **Decay rates must match the life length.** `SparsePairState`
   values decay as `rate^steps_elapsed`. A rate of 0.999 is fine for
   an ~800-1,200 step synthetic life but decays to ~0 over an
   ~8,600-step real-data life (0.999^8600 ≈ 0.0002). If a result comes
   back suspiciously exactly zero, check decay-rate-vs-life-length
   arithmetic before assuming the physics is broken.

3. **Reward must be checked frequently relative to the eligibility
   trace's decay rate**, not just once at the end of a long
   stimulus window. The eligibility trace (typically decay=0.90)
   effectively vanishes after ~50-100 steps; checking reward only
   once after a 200-step presentation means the trace has already
   decayed to numerical noise by the time it's used.

4. **When comparing two placement conditions, verify they have
   comparable raw adjacency/opportunity counts before trusting the
   ratio.** An early real-data placement comparison put segregated
   clusters far enough apart that they had 57x fewer adjacent
   cross-type pairs than the interleaved condition — the resulting
   "1,448x" effect was mostly a rigged setup, not a real finding. Fixed
   by placing both conditions in the same disc, same density, differing
   only in whether types are arranged as contiguous wedges (segregated)
   or shuffled (interleaved).

5. **When editing a running script's control flow (e.g., moving a
   rewire block), verify the resulting indentation actually nests
   where intended** — `ast.parse()` catches syntax errors but not
   logical nesting mistakes. Print/count how often a block actually
   executes rather than assuming an edit landed correctly.

7. **Audit an algorithm on the input the deployed path actually
   feeds it, not a convenient variant of it.** Exp 38's SpiNeCluster
   (Kernighan-Lin partitioning) was validated by `--sanity-cluster`
   on type-*shuffled* neuron labels, where it reached the exact
   optimal cut. But `make_placement_spinemap` calls it with type-
   *sorted* labels, and on that input the same greedy stalled 6.4%
   above the optimum — at essentially the random-partition cut
   (KL/random = 0.9965, mean cluster type-purity 0.38, 1 of 60 cores
   pure). The published "SpiNeMap lands near interleaved" result was
   therefore measured on a partition that had barely partitioned. Two
   things made it easy to miss: the sanity check used a different
   input distribution than the deployment path, and the cut's dynamic
   range on this graph is narrow (even the exact optimum is only 6%
   below random), so an absolute cut number looks plausible on its
   own. Fixed by (a) exhaustive best-swap search within each cluster
   pair — the old top-gain pairing was unsound because the combined
   gain subtracts `2*W[na,nb]` and so is not monotone in the
   individual gains — and (b) an internal random node relabeling, so
   caller node order cannot determine the result. The check now runs
   both orderings, asserts gap == 0 vs the theoretical optimum, and
   prints KL/random and cluster purity — a ratio-to-baseline, not
   just an absolute score.

6. **If a new file's behavior differs from a known-working reference
   on identical inputs, A/B them directly** — call the same function
   from both contexts on the same data — rather than theorizing about
   what might differ. This isolated harness-vs-physics bugs faster
   than any other technique used tonight.

8. **A check only certifies the configuration it actually runs — and
   `audit.py` certifies nothing about this code at all.** An audit
   sweep (2026-08-27) mapped coverage against use and found the safety
   net sat over the wrong code. `audit.py` imports nothing from the
   project: it is a self-contained dense reimplementation, so it would
   print GRANITE with the engine deleted. `audit_engine.py` gates
   `engine.Life`, but NO preprint-backing experiment imports the engine
   — every one of them depends on just `SparsePairState`, `SpatialGrid`
   and its own inline physics loop. So the hours-long gate guarded code
   no published number touches, while the load-bearing path had no gate.
   Fixed by `src/audit_deployed.py` (~90s, run it). Same lesson as #7,
   one level up: ask what the deployed path calls, then test that.

9. **`FastLife` was never certified to match `LocalLife`, despite the
   docs saying "exactly".** Its self-test compares one scalar
   (taught_mass) on one seed under one rule, accepts any difference
   below 50 as "CLOSE = vectorization preserved logic", and asserts
   nothing. On a full edge-set comparison it differs by 3,078-7,077 of
   10,000 edges; `rule='rpe'`, seed 1 gives a taught-mass difference of
   65, which exceeds the test's own threshold and would print DIVERGED
   — but that configuration is never run. No published result uses
   `FastLife`; `topoforge-scale` plans to, so re-certify first.

10. **Learning numbers are environment-sensitive; record the
   interpreter.** `run_life` picks cold edges via `np.argsort` over a
   score array that is mostly exactly zero, using the default UNSTABLE
   sort, so tie order decides which connections are replaced. numpy
   1.26.4 vs 2.5.1 on the PLB, seed 0: interleaved 2955 vs 2979,
   segregated 682 vs 724 (6.2%), headline ratio 4.33x vs 4.03x. With
   `kind='stable'` both versions agree exactly (2978 / 702, ratio
   4.24x). This also explains a discrepancy previously mistaken for a
   stale code state: the preprint's old Section 4.6 table (686/2926)
   was simply run under numpy 1.26.4 while Section 4.1 (731.8/2952.9)
   was run under 2.5.1. Disclosed in preprint Section 3.4. The
   one-line fix (`kind='stable'`) is NOT yet applied because it shifts
   every published number and needs a deliberate regeneration pass.

## Current state (as of this handoff)

- **Provisional patent filed**: USPTO #64/134,008, filed 08/14/2026.
  Four claim areas: factory-dreams commissioning, value-gated
  structural plasticity, spatial consensus anomaly detection,
  interleaved placement for plastic hardware. 12-month window to
  decide on a full utility filing.
- **Preprint**: drafted (`topoforge_preprint.md`), not yet on arXiv.
  arXiv submission needs an endorser in cs.NE — no institutional
  affiliation, so the plan is to ask a cited author (e.g., Anup Das,
  SpiNeMap) once the paper is in strong shape.
- **Henry (Catalyst Neuromorphic)**: potential collaborator, contacted
  re: testing on his N4 hardware. His current chip doesn't have
  structural plasticity in the tapeout (cut before manufacture), so
  the collaboration is currently limited to an inference-only
  comparison he offered to run. Awaiting his results; no fixed
  timeline. A follow-up email was sent; no reply as of this handoff.
- **Real-data result (Exp 37b -> 37c)**: the placement effect confirmed
  on real SHD (Spiking Heidelberg Digits) speech data, closing the
  "is this only a synthetic-pattern artifact" gap. NOTE the headline
  number was corrected: Exp 37b's 9.31x (n=5, 4 samples/class) was a
  small-sample draw that does NOT reproduce from committed code (a fresh
  audit gives 7.22x). Exp 37c re-ran at n=12 with 20 distinct
  samples/class and confirmed the effect is not a small-sample artifact:
  7.25x [5.05x, 11.48x] at the validated life/decay regime, p < 1e-8.
  The magnitude is regime-dependent (~3.6x-7.3x with training exposure),
  not a single fixed number. Preprint, README, and the exp37b docstring
  were all reconciled to this.
- **Exp 38 (SpiNeMap), corrected**: closed the "invented baseline" gap.
  SpiNeMap (Balaji et al., arXiv:1909.01843) has no public code, so its
  published two-step algorithm was implemented from scratch —
  SpiNeCluster (Kernighan-Lin partitioning) + SpiNePlacer
  (particle-swarm placement) — and run through the PLB. The FIRST
  version of this result was wrong: the partitioner stalled on
  type-sorted input (the input the placement path actually uses) and
  was benchmarked as a near-random placement, which made SpiNeMap look
  like it landed near interleaved. See gotcha #7. With the partitioner
  fixed (exact optimal cut on both node orderings, purity 1.000 on the
  population graph), the finding REVERSES: on the population graph —
  the graph a mapper actually has at map time, since the associations
  do not exist as synapses until learned — SpiNeMap lands 7.7% of the
  way from segregated toward interleaved (903.8 +/- 61.2 vs 2952.9 +/-
  29.1 interleaved, 3.27x, p = 9.9e-20), i.e. it DOES reproduce the
  pathological placement. On the functional graph (associations
  declared up front) it lands at 101.2%, indistinguishable from
  interleaved. The decisive variable is the graph the optimizer is
  given, not the objective it minimizes. This strengthens the paper's
  claim: the segregated baseline is a stand-in for a deployed tool's
  output, not a strawman. Preprint abstract, Section 4.6, Discussion
  and Limitations were rewritten; a stale Table 4 that disagreed with
  Section 4.1 on the same conditions (686/2926 vs 731.8/2952.9) was
  reconciled at the same time.
- **Most recent completed result (Exp 42 + 42b)**: swept fabric core
  pitch against the plasticity radius (rho = 0.5-3.0), holding the
  radius fixed and rigidly respacing each core's disc (bit-identical to
  the original at rho = 1). Three findings. (1) The mapping-tool penalty
  SATURATES rather than switching on: population-graph SpiNeMap is
  already 1.90x below interleaved at rho = 0.5 and becomes
  indistinguishable from the segregated baseline at rho >= 1.5 (733 +/-
  3 vs 730 +/- 5). (2) Functional-graph SpiNeMap is flat across the
  whole sweep and BEATS interleaved by 1.46x at rho >= 1.25, because it
  co-locates the populations that must associate instead of mixing all
  types uniformly -- the most actionable result in the paper for tool
  builders. (3) The distance-discount confound is dead: interleaved
  learning is identical at rho = 1.5/2.0/3.0 (2035/2039/2039) across a
  4x change in wire distance, at pinned reach. A registered prediction
  FAILED -- interleaved is not flat in pitch (it falls 1.49x), because
  it also draws partners from neighbouring cores at low rho; reported
  as such in Section 4.8. Exp 42b then showed the pre-registered
  mean-COUNT reach metric fails across placement families (R^2 = 0.266)
  while COVERAGE -- the fraction of correlated neurons with any partner
  in reach -- gives R^2 = 0.832; count adds only +0.004 on top of
  coverage. Logged in Limitations as post-hoc metric selection; a
  confirmatory test needs a placement family built in advance to make
  count and coverage disagree (not yet run).
- **Most recent completed result (Exp 43)**: confirmatory test of the
  COVERAGE mediator, run because Exp 42b's coverage form was selected
  post-hoc. Three placements built in advance so count, partner-share
  (frac) and coverage predict DIFFERENT orderings, predictions and
  decision rules registered in the docstring before running. Blobs of
  constant density on a lattice spaced so every intra-blob pair is
  inside the plasticity radius and every inter-blob pair outside it --
  composition fixes reach by construction, verified with zero
  inter-blob neighbours. Result: the COUNT account is REFUTED (C3 has
  the highest mean count, 6.25, and the lowest learning, 2082, below
  both others at p = 3.6e-11 and 1.5e-16). Coverage confirmed in
  isolation (C2 vs C3, matched count and frac: 2739 -> 2082, 1.32x).
  But frac is NOT inert (C1 vs C2, matched count and coverage: 2909 ->
  2739, 1.06x, p = 5e-5), and a density control -- rescaling C2's blob
  radii to match C1's nearest-neighbour spacing -- moved learning only
  0.4% (p = 0.72), so that small frac effect is real rather than a
  boundary-density artifact. Standing claim: the mediator is
  fractional, not a pool size; primarily whether a correlated neuron
  has ANY reachable partner, secondarily what share of its
  neighbourhood they are. NOT calibrated -- the same relative coverage
  drop costs 1.32x here and 3.27x in Exp 38's geometry, so it predicts
  ordering, not magnitude. Logged as such in Limitations.
- **Most recent completed work (audit sweep, 2026-08-27)**: after the
  Exp 38 partitioner bug, swept every audit and self-test asking one
  question -- does this check exercise the configuration the deployed
  path uses? Four findings, logged as gotchas #8-#10 above. Nothing
  found invalidates a published result. `SpatialGrid.within` was
  brute-force verified exact at the deployed `cell == radius` setting
  (the zero-margin ring=1 case its self-test never ran) on every real
  placement plus adversarial boundary points; `SparsePairState` is
  exact to 1.67e-15 against a dense reference under the deployed call
  pattern. New gate `src/audit_deployed.py` passes under both numpy
  1.26.4 and 2.5.1 in ~60s. CLAUDE.md corrected where it asserted
  things now known false. Second half of the sweep found the
  EXPERIMENT-level audits to be in good shape, in contrast to the
  older src/ infrastructure: exp39's `--verify-equiv` is the model
  (feeds the original's placements to both implementations and
  requires bit-for-bit physics equality), and exp37c/40/41/42/43 each
  re-audit their own deployed configuration. Also verified clean:
  `shd_loader` checks downloads against official MD5 checksums (note
  it degrades to a warning if the checksum fetch fails), and
  `topoforge_run.py`'s independent re-implementation of the three
  placement strategies is BIT-IDENTICAL to the benchmark's -- benign
  today, but nothing enforced it, so the gate now pins it. A CLI that
  silently drifted from the benchmarked geometry would recommend a
  placement the paper never validated.
- **Most recent completed result (Exp 44)**: the energy-learnability
  frontier -- the first measurement in this project of the objective a
  mapper actually optimizes (summed squared wire length over REALIZED
  connections, post-plasticity) against what each placement learned.
  Three results. (1) Inference energy is not "negligibly different"
  between segregated and interleaved as Section 4.1 said, it is
  BIT-IDENTICAL (5.01906e6, all 8 seeds) -- they share coordinates and
  seeded connectivity and differ only in which type label sits where,
  so the zero-inference-cost claim can drop its hedge. (2) A registered
  prediction FAILED: interleaving does cost post-plasticity energy,
  1.0143x vs segregated, p = 3.6e-5. Real and unambiguous -- and 1.4%,
  against 4.27x more learned structure. Reported as a falsification,
  with the magnitude stated so nobody reads "significant" as "large".
  Per unit taught, segregated costs 3680 vs interleaved's 874. (3)
  SpiNeMap on the functional graph is PARETO-DOMINANT over everything
  tested: 0.972x energy at 1.006x learning vs interleaved, 0.986x
  energy at 4.30x learning vs segregated. With Exp 42's finding that it
  is also the only condition immune to fabric sparsity, the practical
  recommendation is now "give the mapper the association structure",
  not "interleave". Also noted: plasticity HALVES wire energy in every
  condition (5.0e6 -> 2.5e6), because the rewire rule's distance
  discount makes it pursue a communication objective on its own.
  Preprint gains Section 4.9; abstract and Discussion updated. Run
  under numpy 1.26.4 (venv).
- **Open, honestly-unresolved thread**: an "inverted-U" finding in
  Exp 35's timing-jitter sweep (moderate jitter beats both perfect
  synchrony and high jitter) survived three separate mechanism
  hypotheses being tested and refuted. The phenomenon is solid and
  reproducible; the underlying cause is still open.
- **Scale engineering**: `topoforge-scale`'s Phase 1 (dense-matrix
  elimination) and Phase 2 (Numba compilation of the hot loops) are
  both complete and audited. A crossover benchmark shows the
  Numba-compiled engine overtakes the original around N~50,000-70,000
  — right where Phase 3 (the actual N=10K-500K scale ladder) would
  need it. Phase 3 has not yet been run.

## Reference documents worth reading early

- `README.md` (repo root) — the public-facing summary, benchmark
  numbers, quickstart.
- `topoforge_preprint.md` — the fuller scientific write-up, including
  a Limitations section written adversarially (a self-directed
  skeptical-reviewer pass, not just a formality).
- `neuromorphic_sandbox_report.md` and `_addendum.md` — earlier-stage
  research notes, useful for understanding how the core architecture
  (spatial cores, structural plasticity, RPE-driven value gating) was
  derived from first principles across the early experiments.

## What this project's author is like to work with

Self-taught, working nights/weekends around a full-time job. Values
being told directly when something is wrong or risky rather than
being cushioned — the collaboration works best when you push back on
weak claims and flag real problems plainly, the same way you'd flag
them to a colleague. Prefers concrete next steps over open-ended
options when a call needs to be made.
