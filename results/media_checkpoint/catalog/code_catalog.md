# Code Lineage Catalog

Purpose: identify the code paths behind the generated media and experiment claims. This is not the report.

| Path | Role | Lines | SHA-256 prefix |
| --- | --- | ---: | --- |
| `scripts/io/render_cmu_trusted_replay.py` | source mocap replay renderer | 223 | `f3d5881319be` |
| `scripts/runpod/export_cmu12407_mocapact_hdf5.py` | CMU 124_07 export into MoCapAct HDF5 format | 175 | `7ee40ee2c640` |
| `scripts/runpod/train_cmu_samebody_imitation_ppo.py` | same-body physical imitation PPO experiments | 649 | `f60b18947c65` |
| `scripts/runpod/search_mocapact_residual_bias_cem.py` | early CEM residual tracking search | 510 | `2421070483cf` |
| `scripts/runpod/search_mocapact_speed_residual_cem.py` | bat-speed oriented residual search | 540 | `80e1312ae50b` |
| `scripts/runpod/render_mocapact_old_bat_asset.py` | massful old-bat scene rendering and telemetry | 669 | `44c006a816dd` |
| `scripts/runpod/search_mocapact_virtual_tball_cem.py` | virtual tee-ball task-objective search | 637 | `403c743fc659` |
| `scripts/runpod/search_mocapact_physical_tball_cem.py` | massful physical tee-ball distance CEM | 664 | `f64ad941f243` |
| `scripts/runpod/search_mocapact_batspeed_imitation_cem.py` | no-ball bat-speed imitation gate search | 641 | `e63bd7345be2` |
| `src/motion/cmu_replay.py` | CMU ASF/AMC playback and qpos construction | 198 | `2758a79e2139` |
| `src/motion/mocapact_rollout.py` | local rollout utilities for MoCapAct-style outputs | 219 | `785925ec6d58` |
| `src/optim/kinematic_evaluator.py` | kinematic batting evaluator | 279 | `a88ab5954790` |
| `src/env/contacts.py` | Nathan bat-ball model and flight helpers | 236 | `050fac0e5d13` |
| `assets/mujoco/cmu_batting_scene.xml` | kinematic batting scene with bat sites | 433 | `01ae0948b7fb` |
