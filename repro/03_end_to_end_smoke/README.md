# 03_end_to_end_smoke: planned, not implemented

The future test should exercise real data loading, bounded training/validation,
checkpoint save/reload, inference, inverse transforms and evaluation. Its workload
and acceptance checks have not been implemented. The launcher deliberately exits 2;
an empty script must not be mistaken for a passing training lifecycle test.

Use level 01/02 for inference checks. Level 04 runs an actual reduced training
experiment and is not an automatic substitute for this bounded test.
