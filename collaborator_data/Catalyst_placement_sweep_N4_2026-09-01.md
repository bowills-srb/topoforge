# Multicast fan-out against placement, N4 48-core deployed model

Catalyst Neuromorphic, 1 September 2026. Prepared for Bo Wills.

## What was measured

Placement cannot change accuracy on N4, because delivery is exact and a neuron
performs identical arithmetic wherever it sits. The only quantity placement moves
is routed traffic, so the metric is **multicast fan-out**: the number of distinct
cores a spiking neuron's events must be delivered to. That is the hardware cost
of a placement.

Model: the deployed 48-core SHD build. 1,024 recurrent adLIF hidden units plus 20
readout, 1,486,656 quantised connections, 1,724 distinct sources, 47 occupied
cores. Connectivity is 84.2% of the dense matrix.

Placements compared, all on the same connection set:

- **deployed** — first-fit by fan-in, what the compiler actually produces
- **interleaved** — neuron *n* to core *n* mod 48
- **blocked** — contiguous index blocks
- **random** — uniform assignment

## Result at full density

Every one of the 1,724 sources reaches all 47 occupied cores. Minimum, mean and
maximum fan-out are all exactly 47. Placement has no room to act.

## Under magnitude pruning

Mean cores reached per source:

| Sparsity | Edges | deployed | interleaved | blocked | random |
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

## Reading

Blocked, interleaved and random sit within 0.2 cores of each other at every
sparsity level from 0 to 99.9%. A deliberately segregated arrangement costs
nothing on this network.

The deployed placement is modestly better than all three, but not because it
exploits community structure. It is first-fit by fan-in, so it correlates with
which neurons retain edges under magnitude pruning.

The conclusion is that a backpropagation-trained recurrent network, even stripped
to 0.1% of its weights, carries no community structure for a placement to exploit.
The penalty requires connectivity with locality already in it.

## Reproducing

`catalyst-n4/fpga/f2/placement_study.py`, run against `fpga_bundle_48/model.npz`.
Fan-out is read from the compiled CSR delivery tables, so it is the mapping the
hardware would actually execute, not a simulation of one.
