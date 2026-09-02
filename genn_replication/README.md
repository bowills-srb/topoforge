# GeNN replication — segregated vs interleaved placement on an independent engine

`topoforge_genn_colab.ipynb`

## Status: UNTESTED

**The GeNN portion of this notebook has never been executed.** It was written on a
machine with no GPU, no CUDA and no C++ compiler, so GeNN could not be installed
here, let alone run. Every GeNN API call was written by reading real source —
never from memory — but reading source is not running code. Expect to debug it on
first run.

There are no results in this directory. There are no placeholder numbers and no
example output anywhere in the notebook. If you want a number from this, you have
to run it.

## What this is, and what it is not

The paper's Limitations section flags that every result in TopoForge comes from
one simulator written by one person. This notebook addresses **that one weakness
and nothing else.**

- It **is** a same-rule-family replication on an **independent engine**: GeNN, a
  third-party GPU spiking simulator, running a third party's implementation of a
  published structural-plasticity rule.
- It is **not a hardware result** and **does not close the hardware gap**. Henry's
  N4 comparison and the inference-only limitation are untouched by this.
- It is **not an independent test of the learning claim.** The formation rule it
  uses is purely distance-dependent, so most of the segregated-vs-interleaved gap
  is a consequence of the distance kernel and is predictable analytically. That is
  also true of exp50, and it is exactly why the endpoint is framed as *suppression
  below a chance ceiling* rather than as a learning ratio. What the notebook tests
  is the **placement → connectivity** step, on an engine this project did not
  write.

## Design

Mirrors `src/experiments/exp50_published_rule_confirmatory.py` as closely as the
engine allows.

- **N = 900** (30×30 grid), **NC = 5** types, 180 neurons each — matching exp50.
- **Two conditions, one coordinate array.** GeNN's reference distance code derives
  position from neuron index (`x = id % grid_num_x`, `y = id / grid_num_x`), so
  both conditions share literally the same coordinates by construction, not by
  assertion. Segregated puts each type in a contiguous 6-row stripe; interleaved
  is the *same multiset* of types permuted across the *same* positions. The
  notebook asserts the multisets are identical, the per-type counts are identical,
  and the two assignments actually differ.
- **Primary endpoint:** cross-type fraction of grown synapses, autapses excluded,
  against the type-blind chance ceiling `(N − N/NC)/(N − 1) = 0.8009`.
- **The framing point, which is easy to invert:** interleaved sits *at* the ceiling
  and *cannot exceed it*. The finding is that **segregation suppresses cross-type
  wiring below chance**, not that interleaving enhances it above chance.
- **12 seeds per condition** (100–111, exp50's seeds), Welch t-test, Cohen's d.
- **Pre-registered thresholds are exp50's, lifted unchanged** (segregated < 0.60,
  interleaved > 0.70, p < 0.05). They were not chosen after looking at GeNN output,
  because there is no GeNN output.
- The GeNN model contains **no notion of neuron type at all.** Type enters in
  exactly two places, both outside the simulator: the host-side stimulus schedule
  and the Python-side readout.

### One thing that *was* run

The geometry cells are plain NumPy and do not need GeNN, so they were executed
locally. They produce the **analytic prediction from the formation kernel alone** —
what the distance kernel implies with the simulator switched off:

| | kernel-only cross-type fraction |
|---|---|
| segregated | 0.144 |
| interleaved | 0.800 |
| type-blind chance ceiling | 0.801 |

**This is a prediction, not a replication result.** It is a property of the
placement geometry, computed in NumPy. Its value is as a yardstick: when GeNN
produces a number, you can see how far activity-dependent retention moved it away
from pure geometry. Do not cite it as a GeNN result.

## How to run it

1. Upload `topoforge_genn_colab.ipynb` to [Google Colab](https://colab.research.google.com/).
2. **Runtime → Change runtime type → GPU** (T4 is fine). GeNN will fall back to
   its `single_threaded_cpu` backend without one, but it will be slow.
3. Run cells top to bottom.
4. **Stop at the smoke test (section 7) and read its output before continuing.**
   It runs one short simulation per condition and exists to catch the failure modes
   that would waste an hour of sweep time:
   - Does it compile and run at all?
   - Do neurons fire? Target roughly 2–30 Hz mean, few silent neurons. If the rate
     is ~0, raise `FF_WEIGHT`. If pinned near the refractory ceiling, lower it.
     `FF_WEIGHT` and `PEAK_RATE` are the only two parameters in the notebook not
     taken from a published source, and nothing guarantees the chosen values work.
   - Do synapses grow? Edge count well above zero.
   - Is there row headroom? `max_row_len` must stay clear of `MAX_ROW_LEN = 128`.
     At the cap, synapses are being silently dropped and every number is invalid.
   - Note the wall-clock time and extrapolate before launching the sweep.
5. Run the sweep (section 8) and the analysis (section 9).

### What to look for in the output

- **`[P1a]` structural parity** — must hold, or the comparison is confounded. This
  is the gotcha that produced the bogus "1,448×" early in this project.
- **`[P1b]`** — edge counts settled (last-quarter drift < 10%), non-zero, with row
  headroom, and a sane firing rate. If P1b fails the run is COMPROMISED; do not
  interpret P2.
- **`[P2]`** — the primary endpoint, plus the chance ceiling and the kernel-only
  analytic prediction side by side.
- **VERDICT** — decision rule copied from exp50. A `DID NOT REPRODUCE` verdict is a
  real outcome and should be reported as one, after checking the P1b diagnostics.

Runtime is unknown. 24 model builds each invoke `nvcc`, which dominates for short
runs. If it is too slow: lower `TSIM_MS`, or cut `SEEDS` — but not below 8, and say
so if you do.

## GeNN API: verified vs inferred

Sources fetched 2026-09-02. "Reference" = [`jhnnsnk/genn_structural_plasticity`](https://github.com/jhnnsnk/genn_structural_plasticity)
(MIT), accompanying Knight, Senk & Nowotny (2026), *A flexible framework for
structural plasticity in GPU-accelerated sparse spiking neural networks*,
Neuromorphic Computing and Engineering 6(1) 014019, doi:10.1088/2634-4386/ae4535.
"PyGeNN source" = `genn-team/genn` at `master`.

### Verified against real source

| Item | Source |
|---|---|
| Colab install cell (`gdown` + `pygenn-5.4.0-cp313-cp313-linux_x86_64.whl` + `%env CUDA_PATH`) | copied verbatim from the GeNN team's own tutorial notebooks: `genn-team/genn` `docs/tutorials/mushroom_body/*.ipynb` and `genn-team/ml_genn` `docs/tutorials/*.ipynb` |
| Source-build fallback (`pip install https://github.com/genn-team/genn/archive/refs/tags/5.3.0.zip`, `libffi-dev`, `CUDA_PATH`) | `genn-team/genn` `docs/installation.rst` |
| `BamfordStructuralPlasticity` custom connectivity update model — the whole rule, host and row update code | reference `topomap/custom_models_and_snippets.py`, **copied verbatim** |
| `GaussianProfileWithoutReplacement` sparse connectivity snippet | same file, **copied verbatim** except the two length-cap lambdas |
| `STDPAllToAllSpikePairing` weight update model | same file, **copied verbatim** |
| `PoissonSpatiallyCorrelatedInput` neuron model | same file, **copied verbatim** |
| `add_neuron_population`, `add_synapse_population`, `add_custom_connectivity_update` signatures | PyGeNN source `pygenn/genn_model.py`, cross-checked against the reference's calls — **no drift between 5.3 and master** |
| `init_weight_update`, `init_postsynaptic`, `init_sparse_connectivity`, `init_var`, `create_var_ref`, `create_wu_var_ref` | reference `topomap/parameters.py` and `topographic_map_model.py`; exports confirmed in `pygenn/__init__.py` |
| Connectivity readout `pull_connectivity_from_device()` → `get_sparse_pre_inds()` / `get_sparse_post_inds()` | reference `topographic_map_model.get_connectivity`; implementation read in `pygenn/genn_groups.py` |
| That `sg.vars["g"].values` returns an array **element-aligned** with the sparse pre/post inds | `pygenn/model_preprocessor.py` `SynapseVariable.values` — it and `get_sparse_post_inds` hstack rows using the same `_row_lengths` view |
| Neuron var write path `pull_from_device()` → `.values = arr` → `push_to_device()` | reference `topographic_map_model.set_stimulus_rates` |
| EGP init `extra_global_params[...].set_init_values(np.zeros(...))` (including the reference passing float arrays for `unsigned int*` / `uint32_t*` EGPs — mirrored, not "fixed") | reference `topographic_map_model.add_connectivity_update` |
| `model.custom_update("UpdateConnectivity")`, `step_time()`, `timestep`, `t`, `seed`, `dt`, `build()`, `load()`, `unload()` | reference `run_experiments.run_model` + `pygenn/genn_model.py` |
| Built-in `LIF` sim / threshold / reset code and derived params | GeNN C++ source `include/genn/genn/neuronModels.h` |
| Built-in `"ExpCond"`, `"StaticPulse"`, `"OneToOne"` usage and their parameter names | reference `topomap/parameters.py` |

### Inferred, or otherwise not verified

| Item | Why it is a risk |
|---|---|
| `LIFWithSpikeCount` | **Ours.** A mechanical transcription of GeNN's built-in `LIF` plus a `spikeCount` var incremented in `reset_code`. The transcription source is verified; the `spikeCount +=` line is not in any published source. We added it so firing rates come back through the already-verified variable path rather than through `spike_recording_data`, whose return shape could not be verified. |
| `FF_WEIGHT`, `PEAK_RATE` | The only two parameters not taken from a published source. Nothing guarantees the chosen values make the LIF population fire. **This is what the smoke test is for.** |
| Whether the network settles inside `TSIM_MS` | The rewiring rate was raised 10× over the reference's scaled value on an analytic estimate of the equilibration time constant. The `[P1b]` settling probes exist to check whether that estimate was right. |
| Wall-clock runtime | Unknown. Time the smoke test. |
| Whether GeNN accepts a `SPARSE` synapse group that starts nearly empty | The reference's own "Unconnected" option uses `prob = 0.0001` rather than 0, which suggests exactly-zero may be an edge case; we follow their spirit with `INIT_PEAK_PROB = 0.001` rather than 0. |

### Capability gaps found in GeNN's API

**None that block this design.** Everything the experiment needs exists:
per-neuron distance-dependent synapse formation and elimination
(`create_custom_connectivity_update_model` with `row_update_code` /
`host_update_code`, `add_synapse()` / `remove_synapse()` / `for_each_synapse`),
host-driven per-neuron input rates, and full readout of the final sparse
connectivity and its weights.

Two limitations worth knowing, neither fatal:

1. **GeNN's distance metric is index-derived, not coordinate-based.** The reference
   computes position from the neuron index on a regular grid with periodic boundary
   conditions. Arbitrary 2-D coordinates (like exp32b's random discs inside cores)
   would require passing a coordinate array as an extra global parameter and
   rewriting the distance code. We did not do that — we adapted the placement to a
   grid instead, which is a *cleaner* test of the invariant since positions become
   identical by construction. But it does mean this is not exp32b's geometry.
2. **Periodic boundary conditions are kept verbatim**, so in the segregated
   condition the top stripe wraps around and touches the bottom stripe, giving
   segregation one extra type boundary it would not have on a bounded sheet. This
   makes the segregated condition *less* extreme — conservative with respect to the
   hypothesis, so it was left alone.

## Deliberate deviations from the reference

All of these are marked in the notebook at the point they occur.

| Deviation | Reason |
|---|---|
| `dt = 1.0 ms` instead of `0.1 ms` | Colab runtime. The reference's own comment notes Bogdan et al. 2018, the model it follows, uses 1 ms. Set `DT = 0.1` to match the reference exactly if you have the time budget. |
| Lateral connectivity starts nearly empty (`INIT_PEAK_PROB = 0.001`) | So that essentially all measured connectivity is *grown by the rule*, as in exp50. The reference starts from a Gaussian-initialised network, which would put a placement-dependent baseline into the measurement — the same confound as gotcha #11. |
| `NUM_REWIRING_ATTEMPTS = 350` rather than the reference's ~35 for this N | Reaches steady state inside a Colab-length run. Changes the rewiring **rate**, not the rule. |
| Row/column length caps made configurable (128), reference hardcodes `2*32` / `32` | Structural plasticity needs room to grow rows; hitting the cap silently drops synapses. The notebook checks headroom. |
| 30×30 grid rather than the reference's 16×16 | N = 900, matching exp50 so the numbers are comparable. |
| Autapses excluded from the primary endpoint | The reference's formation rule does not forbid `i → i`, and an autapse is trivially same-type. The count is reported so the choice is visible rather than buried. |

## Licensing / attribution

The four models copied into the notebook (`BamfordStructuralPlasticity`,
`GaussianProfileWithoutReplacement`, `STDPAllToAllSpikePairing`,
`PoissonSpatiallyCorrelatedInput`, plus the `distance_squared` template) come from
`jhnnsnk/genn_structural_plasticity`, which is MIT-licensed. They are delimited in
the notebook by explicit `BEGIN verbatim copy` / `END verbatim copy` markers. If any
of this is used in the preprint, cite Knight, Senk & Nowotny (2026) for the rule and
its implementation, and GeNN for the simulator.
