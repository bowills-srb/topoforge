"""Fast diagnostic: where does Exp 37's pipeline actually break?
Runs ONE short life with heavy instrumentation -- input firing rate,
output firing rate, edge counts input->output -- to find the break
point in under a minute instead of re-running the full 3-4 hour
experiment blindly.

Run: python diagnose_exp37.py
"""
import numpy as np
import sys
sys.path.insert(0, "src")
from sparse_state import SparsePairState
from spatial import SpatialGrid
from shd_loader import load_shd_samples

N_INPUT = 140
N_OUTPUT = 60
N = N_INPUT + N_OUTPUT
INPUT_TYPE, OUTPUT_TYPE = 0, 1

print("Loading a few real samples...")
all_stimuli, all_labels = load_shd_samples(
    n_samples=200, n_channels_out=N_INPUT, bin_ms=4.0, max_steps=200)
sample = all_stimuli[0]
print("Sample shape: {}, max spike count in one bin: {}, mean nonzero value: {:.2f}".format(
    sample.shape, sample.max(), sample[sample > 0].mean()))

rng = np.random.default_rng(11)
coords, cids = [], []
for _ in range(N_INPUT):
    th, r = rng.uniform(0, 2*np.pi), rng.uniform(0, 8)
    coords.append([5 + r*np.cos(th), 5 + r*np.sin(th)])
    cids.append(INPUT_TYPE)
for _ in range(N_OUTPUT):
    th, r = rng.uniform(0, 2*np.pi), rng.uniform(0, 8)
    coords.append([35 + r*np.cos(th), 35 + r*np.sin(th)])
    cids.append(OUTPUT_TYPE)
coords = np.array(coords); cids = np.array(cids)

rng2 = np.random.default_rng(7)
src = rng2.integers(0, N, N * 10); dst = rng2.integers(0, N, N * 10)
keep = src != dst; src, dst = src[keep], dst[keep]
inhib = rng2.random(N) < 0.20

input_idx = np.where(cids == INPUT_TYPE)[0]
output_idx = np.where(cids == OUTPUT_TYPE)[0]
input_set = set(input_idx.tolist())
output_set = set(output_idx.tolist())

# HOW MANY initial random edges actually go input->output or output->input?
io_edges = sum(1 for s, d in zip(src, dst)
               if (s in input_set and d in output_set) or
                  (s in output_set and d in input_set))
print("\nInitial random edges: {} total, {} are input<->output ({:.1f}%)".format(
    len(src), io_edges, 100 * io_edges / len(src)))

v = np.zeros(N); refrac = np.zeros(N, dtype=int)
out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
for s2, d2b in zip(src, dst):
    out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)

rng3 = np.random.default_rng(0)
input_fire_count = 0
output_fire_count = 0
max_output_v = 0.0
input_gain = 0.35

print("\nRunning {} steps of ONE real sample, tracking firing...".format(sample.shape[0]))
for local_t in range(sample.shape[0]):
    inp = rng3.uniform(0, 0.01, N)
    inp[input_idx] += sample[local_t] * input_gain

    v_ = v * 0.90 + inp
    fired = (v_ >= 1.0) & (refrac == 0); f = np.where(fired)[0]
    if len(f):
        for fi in f:
            for ti, wi in zip(out_t[fi], out_w[fi]): v_[ti] += wi

    max_output_v = max(max_output_v, v_[output_idx].max())

    fired_input = len(set(int(x) for x in f) & input_set)
    fired_output = len(set(int(x) for x in f) & output_set)
    input_fire_count += fired_input
    output_fire_count += fired_output

    v = np.maximum(v_, 0); v[fired] = 0
    refrac[fired] = 3; refrac[refrac > 0] -= 1

print("\nRESULTS:")
print("  Total input neuron firings across {} steps: {}".format(sample.shape[0], input_fire_count))
print("  Total output neuron firings across {} steps: {}".format(sample.shape[0], output_fire_count))
print("  Max output membrane potential reached: {:.3f} (threshold=1.0)".format(max_output_v))
print("\nDIAGNOSIS:")
if input_fire_count == 0:
    print("  INPUT never fires -- input_gain ({}) is too weak, or sample data is empty.".format(input_gain))
elif output_fire_count == 0:
    print("  INPUT fires but OUTPUT never fires -- insufficient input->output")
    print("  connectivity/weight to cross threshold. Max output potential reached")
    print("  {:.3f} of the 1.0 threshold needed.".format(max_output_v))
    print("  Likely fix: increase input_gain, lower threshold, or increase the")
    print("  number/weight of initial input->output edges.")
else:
    print("  Both input and output fire -- the bug must be elsewhere (reward logic,")
    print("  epoch structure, or the rewire step).")
