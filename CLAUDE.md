# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Also read `PROJECT_HISTORY.md`** — it holds context that only exists in the project's history (working conventions, the audit-first discipline, a catalog of bugs that cost real time, and current patent/collaboration status). This file covers structure and commands; that file covers judgment. The most load-bearing points from it are summarized below, but read the original before making changes.

## What this is

A research sandbox (project name "TopoForge") investigating one thesis: **on plastic neuromorphic hardware, where you place neurons determines what the network can learn**, not just communication energy. Placing functionally-different neurons close together ("interleaved") lets structural plasticity build cross-type associations that function-grouped block placement ("segregated") makes much harder or impossible. Existing chip-mapping tools optimize only for energy; the headline result is that segregated placement causes a learning penalty vs interleaved at zero-to-minimal inference-energy cost — 3.8x-4.2x on synthetic substrates, 2.92x with geometry removed entirely (pure graph locality), and 5.1x-8.2x (regime-dependent; 8.22x at the validated regime) on real spike-encoded speech data. See CLAUDE.md's "Current project status" below for what's been added since — this summary sentence is deliberately stable and doesn't track every new experiment.

The repo is organized as a lab notebook, not a product. Each numbered experiment in `src/experiments/` is a self-contained, independently runnable script that tests or refutes one hypothesis. **Git history is the primary record** — commit messages are experiment writeups (what was tested, what was found, what was refuted). Read `git log` before making claims about what is or isn't established; findings are deliberately hedged and several hypotheses have been cleanly refuted.

## Two repositories — know which one you're in

- **`topoforge`** (this repo, public) — the main research repo backing a filed provisional patent and a potential collaborator's testing. **It needs to stay stable and citable.** Don't break working experiments; if you change shared engine code, re-run the affected experiments to confirm results still hold.
- **`topoforge-scale`** (private, separate) — a from-scratch rewrite targeting N=500,000 with a `ScaleLife` engine and Numba kernels. Allowed to be messy; only merges back after full equivalence-audit passage.

## Running things

Everything is plain NumPy (plus `h5py` for the real-data loader). There is no build step and no `requirements.txt`; a `venv/` is checked in. Run from the **repo root** — scripts hardcode `sys.path.insert(0, "src")` and will not resolve imports from other working directories.

```powershell
# The CLI: describe hardware in JSON, get a placement recommendation (~90s)
python src/topoforge_run.py                       # runs default_config()
python src/topoforge_run.py src/example_config.json
python src/topoforge_run.py config.json --custom my_placement.csv   # CSV cols: neuron_id,x,y,cluster_identity

# The headline benchmark (PLB), 10 seeds, full stats
python src/experiments/exp32b_benchmark.py

# Any experiment — each prints its own "Run:" line in its module docstring
python src/experiments/exp37b_v2_real_data.py
```

### Tests / verification

There is no pytest suite. Verification is built into the modules themselves:

- **Core primitives self-test** when run directly: `python src/sparse_state.py`, `python src/spatial.py`, `python src/engine.py`, `python src/engine_fast.py`. These shadow-test the sparse/fast implementations against dense reference matrices and should be run after touching those files.
- **`src/audit_deployed.py`** — **the gate to run.** Covers the code the published results actually execute (`SparsePairState`, `SpatialGrid`, the SpiNeCluster partitioner, every placement's structure, and `run_life`'s learning values). ~90s. Most checks are exact and environment-independent; the learning checks carry a tolerance band and say why.
- **`src/audit.py`** — dense reference harness. Seed-sweeps the five load-bearing claims against pre-registered criteria, printing `GRANITE` or `SOFT` per claim. **It imports nothing from this project** — it is a self-contained reimplementation. That makes it valuable as an *independent replication of the findings*, but it is **not a regression test**: it would print GRANITE with the entire engine deleted. Do not use it to validate a code change.
- **`src/audit_engine.py`** — the same five claims re-run on `engine.Life` (legacy mode). Long-running (hours). Note the scope: **no preprint-backing experiment imports the engine at all**, so this gates code no published number depends on. Run it before trusting `Life`/`LocalLife`/`FastLife` (e.g. for `topoforge-scale`), not as a check on the paper's results.

**Audit before you trust — this is the project's single most important convention.** Every non-trivial change here has been validated against known-good output on identical inputs *before* being trusted. This has caught real, consequential bugs, not hypothetical ones (bit-identical sparse-vs-dense distance, shadow-tested Numba state, exact 5-seed engine-rewrite match). When you change engine or state code, write or reuse an equivalence test before believing the new numbers — even if the change "looks obviously correct." The bar for a change: primitive self-tests still pass, and **`audit_deployed.py` passes** — not `audit.py`, which shares no code with anything here and cannot detect a regression.

**Two cautions this convention learned the hard way (2026-08-27 audit sweep).** First, a check only certifies the configuration it actually runs: `SpiNeCluster` was validated on type-shuffled input and silently stalled on the type-sorted input the placement path feeds it, which inverted a published result, and `FastLife`'s equivalence test passes only in the one configuration it tests (see below). Always ask what the deployed path calls, and test *that*. Second, **learning numbers here were environment-sensitive**: `run_life` picked cold edges by sorting a heavily tie-degenerate score array with `np.argsort`'s default unstable sort, so numpy 1.26.4 and 2.5.1 gave different results (6.2% apart on the segregated condition; identical under `kind='stable'`). **This is now fixed** — `kind='stable'` was applied at 21 call sites and all 9 preprint sections were regenerated under it (2026-08-28/29), so results are numpy-version-independent going forward. Still record which interpreter produced a number, and prefer `venv/Scripts/python.exe` for anything published, as a matter of discipline.

## Architecture

The stack is three layers. The trick that makes everything tractable is at the bottom.

**1. Spatial + sparse-state primitives (`src/spatial.py`, `src/sparse_state.py`)**
- `SpatialGrid` — uniform grid bucketing for *exact* k-NN and radius queries (the grid only prunes candidates; distances decide). Rebuilt each epoch after neurons migrate.
- `SparsePairState` — decaying scalars on a dynamic set of `(i, j)` neuron pairs with **lazy exponential decay**: `value(t) = stored * decay^(t - last_touched)`, and `tick()` is O(1). This is what makes N=100K+ tractable instead of storing dense NxN matrices. Three instances are used throughout with different decay constants: **C** = co-firing correlation (~0.95), **E** = eligibility trace (~0.90), **V** = value / reward-attributed structure (~0.999+).

**2. Simulation engine (`src/engine.py` → `engine_local.py` → `engine_fast.py`)**
- `Life` (in `engine.py`) is the base simulation: LIF neurons (leak, threshold, refractory), a fixed input→pattern schedule, structural plasticity (periodically prune "cold" edges and rewire toward high-value candidate pairs), and RPE (reward-prediction-error) learning. Retention rule is pluggable via `rule='corr'|'util'|'rpe'`.
- `LocalLife` restricts plasticity deposits to spatially local pairs within `r_deposit` — this is the mechanism the whole thesis rests on (local rules can't associate neurons placed far apart).
- `FastLife` overrides only `rewire()` with a vectorized array scan. It is **not** an exact match to `LocalLife`, despite what this file previously claimed: on a full edge-set comparison it differs by 3,078-7,077 of 10,000 edges, and taught mass differs by 25-65. Its self-test accepts any difference under 50 as "CLOSE = vectorization preserved logic", asserts nothing, and runs only `rule='corr'`, seed 0 — the one configuration that passes (`rule='rpe'`, seed 1 gives 65 and would print DIVERGED). No published result uses `FastLife`, but `topoforge-scale` plans to, so re-certify it properly before relying on it.
- `r_grow=None` reproduces legacy global-candidate behavior; `r_grow=<float>` is the honest local-physics mode. Watch for which one an experiment uses.

**3. Experiments & the CLI (`src/experiments/`, `src/topoforge_run.py`)**
- Experiments are numbered roughly chronologically (exp20–exp49c as of 2026-08-30). Many **re-implement the sim loop inline** rather than importing `Life`, so they can vary the physics — do not assume an experiment shares code with the engine; read its top-of-file docstring, which states the hypothesis and the exact run command. `tinker_*` scripts are exploratory/tinkering-grade probes (reduced seed counts, no pre-registration) rather than formal numbered experiments — treat their results as suggestive, not preprint-grade, unless a later numbered experiment confirms them.
- The three placement strategies (`segregated`, `interleaved`, `random`) are generated in both `topoforge_run.py` and the benchmark experiments. Segregated = each neuron type in a contiguous region; interleaved = types mixed within each core/disc; random = global shuffle.

**Real data (`shd_loader.py`, `shd_data/`)** — `load_shd_samples()` downloads the Spiking Heidelberg Digits dataset directly from Zenke Lab (deliberately avoids `tonic`/`expelliarmus`, which need a C compiler on Windows; only `h5py` is required). Files cache in `shd_data/`. Used by the `exp37*` real-data experiments.

## Gotchas specific to this codebase

These are bugs that have already cost real time. See `PROJECT_HISTORY.md` for the full list with examples.

- **Lazy decay + long runs is a real trap.** Because `SparsePairState` decays continuously as `rate^steps_elapsed`, structure deposited early in a long simulation can decay below measurability before readout: V decay 0.999 over ~8,600 real-data steps ≈ 0.0002. This has produced spurious exact-"zero" results. Keep decay rate and total run length matched; when a result is unexpectedly zero, suspect decay-vs-life-length arithmetic before the physics.
- **Reward must be checked frequently relative to the eligibility trace's decay.** The E trace (decay ~0.90) vanishes after ~50–100 steps; checking reward only once after a 200-step presentation reads numerical noise.
- **Placement comparisons must be adjacency-matched** to be meaningful. Early segregated-vs-interleaved designs gave segregated ~57x fewer adjacent cross-type pairs, so the ratio reflected opportunity count, not arrangement (a bogus "1,448x"). Newer experiments equalize adjacency by placing both conditions in the same disc/density, differing only in contiguous-wedge vs shuffled type assignment.
- **Sanity-check rewire trigger frequency empirically** (print a counter). `(t // 40) % 1 == 0` is always true; nesting a rewire block inside a reward `if` can make its trigger mathematically unreachable. `ast.parse()` catches syntax errors, not logical nesting mistakes.
- `diagnose_exp37*.py` and `diff_check.py` in the repo root are debugging scripts, not experiments — they A/B an experiment's own `run_life()` against a known-good diagnostic loop to isolate harness-vs-physics bugs.
- `archive/` holds superseded Brian2 prototype notebooks; the project moved off Brian2 to pure NumPy (`src/simulation.py`). Don't mine it for current behavior.

## Current project status

*(Last refreshed 2026-08-30 at commit a573f62. This section drifts behind fast-moving research sessions — before citing a headline number or "most recent result" to anyone, check `git log` and the tail of `PROJECT_HISTORY.md` rather than trusting this snapshot at face value.)*

- **Provisional patent filed**: USPTO #64/134,008 (08/14/2026). Four claim areas: factory-dreams commissioning, value-gated structural plasticity, spatial consensus anomaly detection, interleaved placement for plastic hardware. 12-month window to decide on a full utility filing.
- **Preprint**: drafted (`topoforge_preprint.md`), not yet on arXiv — arXiv cs.NE needs an endorser and there's no institutional affiliation; plan is to ask a cited author once the paper is strong.
- **Henry (Catalyst Neuromorphic)** — potential collaborator contacted re: testing on his N4 hardware. His current chip's tapeout dropped structural plasticity, so collaboration is limited to an inference-only comparison he offered to run. Awaiting results; a follow-up email was sent with no reply as of the last handoff. No fixed timeline.
- **Headline number has been corrected downward from an earlier draft.** The previously reported 9.31x (Exp 37b v2, n=5 real-data seeds) was a small-sample draw that didn't reproduce (fresh audit gave 7.22x). The current, properly-powered reconciled range is **5.1x–8.2x** depending on training-exposure regime, with **8.22x (95% CI [7.03, 10.05], Welch p=3.5e-14)** as the validated-regime headline in the preprint abstract. The synthetic-substrate result (N=900–5,000) stands at 3.8x–4.2x, and the pure-graph-locality (no physical geometry) version at 2.92x.
- **The numpy `argsort` nondeterminism gotcha (see above) is resolved**, not open: `kind='stable'` applied at 21 sites, all 9 preprint sections regenerated under it (commits 3f8c502→cb9e9cc).
- **Most recent completed threads** (all folded into the preprint unless noted):
  - Exp 38 (SpiNeMap): a from-scratch reimplementation of a real published mapping tool shows the same segregation penalty on its own connectivity graph — the effect isn't an artifact of this project's own baseline.
  - Exp 46/47 (mechanism + counterexample): the locality mechanism runs through *two* channels (candidate-discovery radius and a distance-weighted rewiring score), not one; a hoped-for cheap fix (sparse long-range bridges, motivating a real-cortex counterexample) only recovers the gap roughly proportional to bridge budget — no shortcut.
  - Exp 45c (topology, confirmatory-grade): isolated/blob connectivity learns measurably more than connected/filament at matched local-reach stats (1.81x, pre-registered).
  - Exp 48/49/49b/49c (usefulness / partial knowledge): moderate, uncertain foreknowledge is close to useless to a mapper — the benefit concentrates in the 75–100% knowledge regime. Exp 49's first pass looked like a refutation (25% knowledge beat 100%) but that was a single-placement-seed artifact, caught and corrected by re-running with 10 seeds — a worked example of the audit discipline catching itself.
  - Exp 44 (energy-learnability frontier) and the fabric-sparsity sweep: the energy/learnability tradeoff is real but small (1.4%); the segregation penalty saturates past ~1.5x plasticity-radius core pitch; an association-aware mapper beats uniform interleaving by 1.43x on sparse fabrics.
  - Shape-optimization tinkering thread (`tinker_*` scripts, exploratory-grade, not a numbered preprint section): six systematic attempts — including an 18-shape search — to find a topology beating isolated-blob all failed. Strengthens confidence in Exp 45c's topology finding.
  - Section 5.1 (practical recommendations for mapping tool design) just added — the most recent commit as of this refresh.
- **Open, unresolved thread**: Exp 35's timing-jitter "inverted-U" (moderate jitter beats both perfect synchrony and high jitter) survived three refuted mechanism hypotheses. Phenomenon is solid and reproducible; cause is still open.
- **Scale engineering** (`topoforge-scale`): Phase 1 (dense-matrix elimination) and Phase 2 (Numba compilation) complete and audited; crossover benchmark shows Numba overtakes the original around N~50K–70K. Phase 3 (the N=10K–500K scale ladder) not yet run.

## Working with this project's author

Self-taught, working nights/weekends around a full-time job. Values being told directly when something is wrong or risky rather than cushioned — push back on weak claims and flag real problems plainly. Prefers concrete next steps over open-ended options when a call needs to be made.
