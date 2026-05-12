# Media Checkpoint Artifact Catalog

Purpose: organize local media and data for later report writing. This is not the report.

## Selected Videos

| Slug | Role | Video | Still | Contact sheet | Notes |
| --- | --- | --- | --- | --- | --- |
| 01_source_mocap_trusted_replay | source | `results\media_checkpoint\videos\01_source_mocap_trusted_replay.mp4` | `results\media_checkpoint\stills\01_source_mocap_trusted_replay_still.png` | `results\media_checkpoint\stills\01_source_mocap_trusted_replay_contact_sheet.png` | Canonical batting motion segment used as the human reference. |
| 02_physical_imitation_clean | imitation | `results\media_checkpoint\videos\02_physical_imitation_clean.mp4` | `results\media_checkpoint\stills\02_physical_imitation_clean_still.png` | `results\media_checkpoint\stills\02_physical_imitation_clean_contact_sheet.png` | MoCapAct physical tracker candidate without bat or ball. |
| 03_physical_imitation_overlay | imitation audit | `results\media_checkpoint\videos\03_physical_imitation_overlay.mp4` | `results\media_checkpoint\stills\03_physical_imitation_overlay_still.png` | `results\media_checkpoint\stills\03_physical_imitation_overlay_contact_sheet.png` | Overlay used to visually approve that the physical body tracks the swing. |
| 04_virtual_bat_tball | task warmup | `results\media_checkpoint\videos\04_virtual_bat_tball.mp4` | `results\media_checkpoint\stills\04_virtual_bat_tball_still.png` | `results\media_checkpoint\stills\04_virtual_bat_tball_contact_sheet.png` | Early task-objective stage before massful bat/contact was introduced. |
| 05_massful_old_bat_clean | asset validation | `results\media_checkpoint\videos\05_massful_old_bat_clean.mp4` | `results\media_checkpoint\stills\05_massful_old_bat_clean_still.png` | `results\media_checkpoint\stills\05_massful_old_bat_clean_contact_sheet.png` | Original bat mesh attached to the physical hand with no ball. |
| 06_physical_tball_launchfloor | contact result | `results\media_checkpoint\videos\06_physical_tball_launchfloor.mp4` | `results\media_checkpoint\stills\06_physical_tball_launchfloor_still.png` | `results\media_checkpoint\stills\06_physical_tball_launchfloor_contact_sheet.png` | Best tee-ball carry checkpoint: launch angle becomes slightly positive. |
| 07_physical_tball_speedrecover | negative result | `results\media_checkpoint\videos\07_physical_tball_speedrecover.mp4` | `results\media_checkpoint\stills\07_physical_tball_speedrecover_still.png` | `results\media_checkpoint\stills\07_physical_tball_speedrecover_contact_sheet.png` | CEM can shape launch but cannot recover enough bat speed with the current tracker. |
| 08_batspeed_imitation_scale030 | diagnostic result | `results\media_checkpoint\videos\08_batspeed_imitation_scale030.mp4` | `results\media_checkpoint\stills\08_batspeed_imitation_scale030_still.png` | `results\media_checkpoint\stills\08_batspeed_imitation_scale030_contact_sheet.png` | No-ball speed gate run improves contact speed to 7.55 m/s. |
| 09_batspeed_imitation_scale035_cont_iter2 | diagnostic result | `results\media_checkpoint\videos\09_batspeed_imitation_scale035_cont_iter2.mp4` | `results\media_checkpoint\stills\09_batspeed_imitation_scale035_cont_iter2_still.png` | `results\media_checkpoint\stills\09_batspeed_imitation_scale035_cont_iter2_contact_sheet.png` | Stopped checkpoint improves contact speed to 8.30 m/s, still below the 10 m/s gate. |

## Key Metrics

- Kinematic bat sweet-spot contact target: `17.35 m/s`.
- Current best no-ball physical contact speed: `8.30 m/s`.
- Resume-gate target before tee-ball distance optimization: `10.00 m/s`.
- Best physical tee-ball carry checkpoint: `7.35 ft`.

## Generated Figures

- `results/media_checkpoint/figures/tee_ball_checkpoint_metrics.png`
- `results/media_checkpoint/figures/bat_speed_gate.png`
- `results/media_checkpoint/figures/batspeed_cem_trace.png`
- `results/media_checkpoint/figures/pipeline_media_map.png`

## Data Files

- JSON summaries copied to `results/media_checkpoint/data/`.
- Residual checkpoints copied to `results/media_checkpoint/data/residuals/`.
- Consolidated metrics snapshot: `results/media_checkpoint/catalog/metrics_snapshot.json`.
