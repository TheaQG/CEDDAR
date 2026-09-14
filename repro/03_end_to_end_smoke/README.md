Real training pipeline

Tests the complete training lifecycle.

Purpose: Can CEDDAR start from real training data, optimise a model, save it, reload it, generate from it, and evaluate the generated output?

Keep real data/preprocessing pipeline but deliberately reduced neural network and workload

Real input --> preprocessing --> training --> validation --> checkpoint save --> checkpoint reload --> generation --> inverse transform --> evaluation --> external artifact + provenance