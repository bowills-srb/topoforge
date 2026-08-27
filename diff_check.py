"""Run the experiment's OWN run_life() with interleaved placement and
the diagnostic's parameters, instrumented at the same checkpoints.
If this produces 0 while diagnose_exp37e produced 722, the difference
is inside run_life itself and we compare directly.
Run: python diff_check.py
"""
import numpy as np, sys
sys.path.insert(0, "src"); sys.path.insert(0, ".")
sys.path.insert(0, "src/experiments")
from shd_loader import load_shd_samples
import importlib.util
spec = importlib.util.spec_from_file_location(
    "exp37bv2", "src/experiments/exp37b_v2_real_data.py")
m = importlib.util.module_from_spec(spec)
sys.modules["exp37bv2"] = m
spec.loader.exec_module(m)

all_stimuli, all_labels = load_shd_samples(
    n_samples=200, n_channels_out=140, bin_ms=4.0, max_steps=200)
from collections import Counter
counts = Counter(all_labels); top = [c for c, _ in counts.most_common(2)]
tgt = [s for s, l in zip(all_stimuli, all_labels) if l == top[0]][:2]
dis = [s for s, l in zip(all_stimuli, all_labels) if l == top[1]][:2]

print("Calling the EXPERIMENT's run_life with 2 samples, 2 epochs,")
print("interleaved placement -- same as the diagnostic that gave 722.\n")
coords, cids = m.make_placement_interleaved()
ba, bb = m.run_life(coords, cids, 0, tgt, dis, n_epochs=2)
print("Experiment run_life result: bridge_A={:.3f} bridge_B={:.3f}".format(ba, bb))
print("Diagnostic result for equivalent config: ~750")
print()
if ba + bb == 0:
    print("CONFIRMED: the experiment's run_life returns 0 where the")
    print("diagnostic's inline loop returns 750, on identical inputs.")
    print("The bug is INSIDE run_life. Next: instrument it directly.")
else:
    print("run_life works here -- so the issue is in the harness/config")
    print("(n_epochs=15, 8 samples, or placement fn), not run_life itself.")
