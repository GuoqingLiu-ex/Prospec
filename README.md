# ProSpec core code

This repository contains the source-only Jetson snapshot used by ProSpec.

## Included modules

- `jetson_core_code_20261118/fad_vispec/`: verification-guided dynamic draft-budget controller and its controller-side helpers.
- `jetson_core_code_20261118/training_free/`: reordered visual compressor implementation.
- `jetson_core_code_20261118/training_free_opt/`: compressor integration and reorder comparison code.
- `jetson_core_code_20261118/direct_opt/`: memory-saving visual-KV, head-bank, head-scale, overlay, and attention operator implementations.

The main entry points are `fad_vispec/full_controller.py`, `training_free/compressors.py`, and `training_free_opt/run_native_vs_reorder.py`. The `direct_opt` directory contains the operator variants, including the visual-KV and CUDA/Triton operator wrappers.

This is a source-only export from the Jetson experiment tree. Model weights, datasets, virtual environments, caches, generated plots, and test-only/reference-only files are excluded. The source manifest is in `jetson_core_code_20261118/MANIFEST.txt`.
