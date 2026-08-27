"""Fast diagnostic v3: do the smaller dual output clusters (30 each)
ever fire at all? Checks for the degenerate zero-reward trap: if
output never fires, base=0.0=Rhat forever, delta stays exactly 0,
and zero deposits happen -- indefinitely, regardless of how many
trials pass.

Run: python diagnose_exp37c.py
"""
import numpy as np
import sys
sys.path.insert(0, "src")
from spatial import SpatialGrid
from shd_loader import load_shd_samples

N_INPUT = 140
N_OUTPUT_A = 30
N_OUTPUT_B = 30
N = N_INPUT + N_OUTPUT_A + N_OUTPUT_B
INPUT_TYPE, OUTPUT_A_TYPE, OUTPUT_B_TYPE = 0, 1, 2

print("Loading a real sample...")
all_stimuli, all_labels = load_shd_samples(
    n_samples=200, n_channels_out=N_INPUT, bin_ms=4.0, max_steps=200)
sample = all_stimuli[0]

# INTERLEAVED placement (most favorable case for cross-type firing)
rng = np.random.default_rng(11)
coords, cids = [], []
all_types = [INPUT_TYPE]*N_INPUT + [OUTPUT_A_TYPE]*N_OUTPUT_A + [OUTPUT_B_TYPE]*N_OUTPUT_B
rng.shuffle(all_types)
for t_ in all_types:
    th, r = rng.uniform(0, 2*np.pi), rng.uniform(0, 10)
    coords.append([20 + r*np.cos(th), 20 + r*np.sin(th)])
    cids.append(t_)
coords = np.array(coords); cids = np.array(cids)

rng2 = np.random.default_rng(7)
src = rng2.integers(0, N, N * 10); dst = rng2.integers(0, N, N * 10)
keep = src != dst; src, dst = src[keep], dst[keep]
inhib = rng2.random(N) < 0.20

input_idx = np.where(cids == INPUT_TYPE)[0]
output_a_idx = np.where(cids == OUTPUT_A_TYPE)[0]
output_b_idx = np.where(cids == OUTPUT_B_TYPE)[0]
a_set = set(output_a_idx.tolist()); b_set = set(output_b_idx.tolist())

io_a = sum(1 for s, d in zip(src, dst) if (s in a_set) != (d in a_set) and
           ((s in a_set) or (d in a_set)) and not ((s in b_set) or (d in b_set)))
print("Initial random edges touching output_A specifically: checking density...")
a_touch = sum(1 for s, d in zip(src, dst) if s in a_set or d in a_set)
b_touch = sum(1 for s, d in zip(src, dst) if s in b_set or d in b_set)
print("  Edges touching output_A: {} of {} total".format(a_touch, len(src)))
print("  Edges touching output_B: {} of {} total".format(b_touch, len(src)))

v = np.zeros(N); refrac = np.zeros(N, dtype=int)
out_t = [[] for _ in range(N)]; out_w = [[] for _ in range(N)]
for s2, d2b in zip(src, dst):
    out_t[s2].append(d2b); out_w[s2].append(-0.60 if inhib[s2] else 0.30)

rng3 = np.random.default_rng(0)
a_fire_count = 0
b_fire_count = 0
max_a_v = 0.0
max_b_v = 0.0
input_gain = 0.35

print("\nRunning {} steps of ONE real sample...".format(sample.shape[0]))
for local_t in range(sample.shape[0]):
    inp = rng3.uniform(0, 0.01, N)
    inp[input_idx] += sample[local_t] * input_gain
    v_ = v * 0.90 + inp
    fired = (v_ >= 1.0) & (refrac == 0); f = np.where(fired)[0]
    if len(f):
        for fi in f:
            for ti, wi in zip(out_t[fi], out_w[fi]): v_[ti] += wi
    max_a_v = max(max_a_v, v_[output_a_idx].max())
    max_b_v = max(max_b_v, v_[output_b_idx].max())
    fs = set(int(x) for x in f)
    a_fire_count += len(fs & a_set)
    b_fire_count += len(fs & b_set)
    v = np.maximum(v_, 0); v[fired] = 0
    refrac[fired] = 3; refrac[refrac > 0] -= 1

print("\nRESULTS:")
print("  Output_A firings: {}  (max potential reached: {:.3f} of 1.0)".format(a_fire_count, max_a_v))
print("  Output_B firings: {}  (max potential reached: {:.3f} of 1.0)".format(b_fire_count, max_b_v))
print("\nDIAGNOSIS:")
if a_fire_count == 0 and b_fire_count == 0:
    print("  CONFIRMED: neither output cluster EVER fires. This IS the zero-reward")
    print("  trap -- base always 0.0, Rhat stays 0.0, delta always exactly 0,")
    print("  zero deposits forever. The 30-neuron clusters are too small/sparse")
    print("  to reliably activate from the input drive at this gain level.")
    print("  FIX: increase input_gain, increase output cluster size, or increase")
    print("  the initial edge density specifically targeting output neurons.")
else:
    print("  At least one output cluster does fire -- the trap theory is wrong,")
    print("  need to look elsewhere (reward formula logic, epoch structure).")
