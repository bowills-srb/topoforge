# TopoForge

**Placement affects learning.** On plastic neuromorphic hardware, the physical assignment of neurons to cores determines not just communication energy but what the network can learn. Every existing mapping tool optimizes for the first thing. Nobody has published on the second - until now.

**Headline result:** Wire-length-optimized segregated placement (the standard output of every current mapping tool) produces a **4.0x learning penalty** compared to interleaved placement, with **zero energy penalty** during frozen inference. Consistent from N=900 to N=5,000, 10 seeds each.

**Real-data generalization:** the same penalty appears on real spike-encoded human speech (Spiking Heidelberg Digits). A 12-seed, adjacency-matched comparison shows a **3.6x–7.3x** penalty depending on training-exposure regime (7.25x, 95% CI [5.05x, 11.48x], at the validated regime; Welch p = 3.2×10⁻¹⁴). See `src/experiments/exp37c_real_data_scaled.py`.

Preprint: `[arXiv link pending]`
Patent: Provisional filed, USPTO application #64/134,008 (08/14/2026)

---

## Run the benchmark yourself

```bash
git clone https://github.com/bowills-srb/topoforge.git
cd topoforge
pip install numpy

python src/topoforge_run.py
```

Takes about 90 seconds. You'll see three placement strategies compared on learning quality, wire energy, and adaptation after a workload change, with a plain-language recommendation at the end.

## Test your own placement strategy

```bash
python src/topoforge_run.py your_config.json --custom your_placement.csv
```

Your placement CSV needs columns: `neuron_id, x, y, cluster_identity`. See `src/example_config.json` for the hardware-spec format.

---

## What's in this repo

| Path | What it is |
|---|---|
| `src/topoforge_run.py` | The CLI - describe your hardware, get a placement recommendation |
| `src/spatial.py` | Exact k-NN spatial grid, O(N) queries |
| `src/sparse_state.py` | Lazy-decay pairwise state (the memory-efficient trick that makes N=100K+ tractable) |
| `src/engine.py`, `engine_local.py`, `engine_fast.py` | The simulation core - LIF neurons, structural plasticity, RPE learning |
| `src/experiments/` | All experiments (numbered through Exp 37c), each independently runnable |
| `src/experiments/exp32b_benchmark.py` | The Placement-Learning Benchmark (PLB) - the headline result |
| `neuromorphic_sandbox_report.md` | Original research report (Experiments 1-11) |
| `neuromorphic_sandbox_addendum.md` | Addendum (Experiments 12-19d, audit results, revised synthesis) |
| `topoforge_preprint.md` | Full paper draft |

## The three findings, in one paragraph each

**Placement.** Segregating neurons by function (the intuitive, wire-length-optimal choice) puts correlated partners beyond the reach of local plasticity rules - some associations become structurally impossible to learn. Interleaving avoids this at negligible energy cost. Confirmed on synthetic patterns (`exp32b_benchmark.py`) and on real speech data (`exp37c_real_data_scaled.py`).

**Persistence.** Three independent parameter sweeps (value-decay rate, correlation weight, edge turnover rate - `exp26` through `exp28`) show that none of them control how durably a network retains learned structure after its environment changes. Persistence is earned by the value hierarchy and defended by structural inertia - not tunable by any single constant.

**Detection.** Spatially coherent anomalies (Experiments 29-31) produce 4.7x more local correlation structure than scattered noise, discriminated by trajectory shape over time, at zero additional computational cost. The substrate's own geometry performs anomaly detection without a separate algorithm.

## Status

Built by one person on a laptop, in active development. Experiments numbered through Exp 37c, 5-seed-minimum replication on primary claims (real-data replication now at 12 seeds), GRANITE audit passed on both the original dense engine and the sparse rewrite. Numba optimization and cloud-scale runs (N=10K-100K) planned next.

## Citing this work

```
@misc{wills2026topoforge,
  title={Placement Affects Learning: Spatial Topology as a First-Class
         Design Variable for Plastic Neuromorphic Hardware},
  author={Wills, Bo},
  year={2026},
  note={Provisional patent filed, USPTO \#64/134,008},
  url={https://github.com/bowills-srb/topoforge}
}
```

## Contact

Bo Wills - issues and PRs welcome. For collaboration inquiries, open an issue or reach out directly.

