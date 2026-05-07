# RunPod / GPU setup

Cloud setup notes for pod-based work. **Today's pipeline does NOT need
a pod** - kinematic eval runs in 2-3 minutes on a laptop CPU. This doc
is for the next stage: vectorized PPO, hybrid physics fixes, full
DeepMimic-style RL, or MJX ports.

## When to use a pod

| Work | Local OK? | Pod recommended? |
|---|---|---|
| Tier 2A CMA-ES (3-d, ~3 min) | yes | no |
| Tier 2B PPO 50k steps single env | yes (~25 min) | no |
| Tier 2B PPO 1M+ steps with 8+ vec envs | painful | yes |
| Hybrid physics debugging | yes (debug locally) | no |
| Full physics RL (DeepMimic / ASAP) | no | yes |
| MJX / JAX port + GPU training | no | yes |
| Long hyperparam sweep | no | yes |

## Pod selection

Verify rates at https://www.runpod.io/pricing - these are May 2026.

| Pod | $/hr | When |
|---|---|---|
| **RTX 4090 community** | ~$0.34 | dm_control + MuJoCo (CPU-bound). Start here. |
| RTX 4090 secure cloud | ~$0.69 | Long runs where preemption hurts |
| A6000 48GB | ~$1.22 | Larger replay buffers, bigger nets |
| A100 80GB | $1.89-4.18 | Only worth it if env is ported to MJX/JAX |

dm_control's MuJoCo backend is **CPU-bound** - env stepping doesn't use
the GPU. The 4090 only helps for the policy net forward/backward, which
is tiny. Save the A100 budget for a JAX rewrite.

## Recommended template

Pick `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (or
NVIDIA's `nvcr.io/nvidia/pytorch:24.10-py3`). Both have torch + CUDA +
cuDNN preinstalled. We add the rest with `pip`.

For the pod's network volume, allocate at least 10 GB so mocap data,
result artifacts, and PPO checkpoints survive pod restarts.

## First-time setup (after SSH into the pod)

```bash
# 1. Clone fresh - do not rsync a CRLF working tree from Windows
cd /workspace
git clone https://github.com/Dexin-Huang/exitvelo.git
cd exitvelo

# 2. Install python deps
pip install -U pip
pip install mujoco dm_control gymnasium 'stable-baselines3[extra]' \
            cma rich numpy scipy matplotlib imageio

# 3. Headless rendering for MuJoCo (no display on the pod)
export MUJOCO_GL=egl       # try this first
# if EGL errors: export MUJOCO_GL=osmesa

# 4. Download mocap (one-time, ~10 MB)
python scripts/io/download_cmu_data.py

# 5. Verify everything works
python -c "import torch, mujoco, dm_control, cma, stable_baselines3 as sb3; \
           print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available()); \
           print('mujoco:', mujoco.__version__); \
           print('sb3:', sb3.__version__)"
python scripts/smoke/smoke_extracted_physics.py
python scripts/smoke/smoke_env_contact.py
```

If both smokes print `PASS`, the pod is ready.

## Persistent storage

Mount a RunPod network volume at `/workspace/persistent` and symlink the
output directories so they survive pod restarts:

```bash
mkdir -p /workspace/persistent/{data,results,models}
cd /workspace/exitvelo
ln -sfn /workspace/persistent/data data
ln -sfn /workspace/persistent/results results
```

## Running real jobs

```bash
# Tier 2A CMA-ES (full): ~3 min on CPU, no GPU benefit
python scripts/run/run_tier2a_full.py

# Tier 2B PPO 50k steps: ~25 min single env on CPU.
# When the env is wrapped for vectorization (8 envs), this drops to ~5 min.
python scripts/run/run_tier2b_ppo.py

# Future: full DeepMimic-style physics RL - not implemented yet.
# Estimated 6 hours on a 4090 community pod for 10M env steps,
# total cost ~$2-3.
```

For long runs use `tmux` or `nohup` so an SSH disconnect doesn't kill the
job:

```bash
tmux new -s ppo
python scripts/run/run_tier2b_ppo.py 2>&1 | tee logs/ppo.log
# Ctrl+B then D to detach. tmux attach -t ppo to come back.
```

## MoCapAct physical-tracking branch

Use the pod for the next-stage MoCapAct probe. The local Windows/Python
3.13 environment imports the repo only until `dm_control.locomotion` asks
for `labmaze`; `pip install labmaze` then fails on the native/Bazel build.
Do not spend time forcing that stack locally.

Local readiness preflight from the Exitvelo repo root:

```bash
.venv/Scripts/python.exe scripts/runpod/check_runpod_readiness.py
```

This writes:

```text
results/final/runpod_readiness_report.json
```

It records platform, required files, basic tool availability, and whether a
RunPod auth variable or `~/.runpod/config.toml` is configured. Credential
values are never printed or written. On this Windows shell the latest report
shows `runpodctl.exe`, the Python SDK, SSH, and `~/.runpod/config.toml` are
present, so the shell can launch a pod. The public GitHub clone check passes
without credential helpers or terminal prompts, but the generated launch plan
also checks whether the current local probe payload is actually present in the
public clone. If the probe files have uncommitted local changes, push them or
explicitly approve the two-file minimal upload path before any paid pod
creation.

Static local MoCapAct checkout audit:

```bash
.venv/Scripts/python.exe scripts/analysis/audit_mocapact_static.py
.venv/Scripts/python.exe scripts/smoke/smoke_mocapact_static_audit.py
```

This writes `results/analysis/mocapact_static_audit/summary.json`. The latest
audit found the expected MoCapAct tracking env, clip-expert train/evaluate
entrypoints, rollout exporter, clip-length utility, and config files in the
configured local MoCapAct checkout. It also found that `CMU_124_07` is absent from bundled
MoCapAct split files while nearby `CMU_124_06` and known control
`CMU_016_22` are present. The pod probe should therefore test
loader/custom-`mocap_path` availability for `CMU_124_07`, not assume a
pretrained snippet exists.
The audit also writes a probe candidate matrix with ready single-clip
commands for `CMU_124_07`, `CMU_124_08`, split-present `CMU_124_06`, and
known control `CMU_016_22`.

### Pod-side code access decision

Use one of these paths before launching another paid pod:

1. Use the public GitHub clone path for the pod, then run the pushed
   `scripts/runpod/` files from that clone.
2. Explicitly approve uploading only these minimal probe files to the pod:

```text
scripts/runpod/mocapact_probe.py
scripts/runpod/mocapact_probe.sh
```

To inspect the exact candidate payload without uploading it, run:

```bash
.venv/Scripts/python.exe scripts/runpod/prepare_probe_payload_manifest.py
```

This writes `results/final/runpod_probe_payload_manifest.json` with byte
counts, line counts, and SHA-256 hashes for only those two files.

To verify whether the repo is cloneable without local GitHub credentials:

```bash
.venv/Scripts/python.exe scripts/runpod/check_github_clone_access.py
```

This writes `results/final/runpod_clone_access_report.json`. The check
disables Git credential helpers and terminal prompts to approximate a clean
RunPod pod. The current report records
`clone_accessible_without_credentials=true` for
`https://github.com/Dexin-Huang/exitvelo.git`.

To generate the no-action runbook for the first approved pod attempt, run:

```bash
.venv/Scripts/python.exe scripts/runpod/build_probe_runbook.py
```

This writes `results/final/runpod_probe_runbook.json`. It includes both code
access paths, the pod-create template, expected remote outputs, local adapter
gates, and cleanup checks. It does not launch a pod or transfer files.

The payload manifest and launch plan also check whether the local probe files
have uncommitted changes or unpushed payload-touching commits relative to the
configured upstream branch. If `scripts/runpod/mocapact_probe.py` or
`scripts/runpod/mocapact_probe.sh` differs from the public clone, the
public-clone path is treated as stale for the current payload: commit and push
the probe changes before using `git clone` on the pod, or explicitly approve
the two-file minimal upload path instead.

To convert the runbook into exact no-action launch parameters for the
installed CLI/SDK, run:

```bash
.venv/Scripts/python.exe scripts/runpod/plan_mocapact_probe_launch.py
```

This writes `results/final/runpod_probe_launch_plan.json`. It records the
`runpodctl pod create ...` command, equivalent `runpod.create_pod(...)`
keyword arguments for the installed Python SDK, prelaunch checks, the selected
code-access path, and the current blocked/ready decision. Public-clone code
access is ready, but `ready_to_launch_paid_pod` remains false until explicit
paid-pod approval is supplied. It still does not launch a pod, upload files,
open network connections, or write credential values.

After explicit paid-cloud approval, regenerate this plan with
`--paid-pod-creation-approved` before using the guarded launcher.

To render the launch plan into a human-readable no-action approval packet, run:

```bash
.venv/Scripts/python.exe scripts/runpod/render_probe_launch_packet.py
```

This writes `results/final/runpod_probe_launch_packet.md`. It keeps the exact
pod-create command, public-clone code-access status, paid-approval blocker,
allowed two-file upload scope, and post-probe gates in one reviewable file
without local absolute paths or credential values.

To generate the guarded launcher report without creating a paid pod, run:

```bash
.venv/Scripts/python.exe scripts/runpod/launch_mocapact_probe.py
.venv/Scripts/python.exe scripts/smoke/smoke_runpod_probe_guarded_launcher.py
```

This writes `results/final/runpod_probe_launcher_dryrun.json`. The default
mode is a dry run. Actual pod creation is blocked unless the launch plan is
generated with explicit paid-pod approval, the caller passes
`--execute --i-understand-paid-cloud`, and the
caller also pins the selected code-access path with `--expected-code-access`.
The launcher only covers pod creation; it does not transfer local files or run
remote probe commands.

To audit any local historical pod-create response without bundling raw RunPod
or SSH environment values, run:

```bash
.venv/Scripts/python.exe scripts/runpod/audit_runpod_create_response.py
.venv/Scripts/python.exe scripts/smoke/smoke_runpod_create_response_audit.py
```

This writes `results/final/runpod_create_response_audit.json`. If
`results/final/runpod_create_response.json` exists, the audit records only
redacted metadata such as pod name, id, GPU type, creation-time status, and
cost. It also keeps the raw response out of the submission bundle. A live
`runpodctl pod list --name exitvelo-mocapact-probe --all -o json` check should
still be run before any paid cloud work.

To make that live check repeatable without printing the API key or bundling
raw RunPod output, run:

```bash
.venv/Scripts/python.exe scripts/runpod/check_live_pod_status.py
.venv/Scripts/python.exe scripts/smoke/smoke_runpod_live_pod_status.py
```

This writes `results/final/runpod_live_pod_status.json`. The script performs
only `runpodctl pod list --name exitvelo-mocapact-probe --all -o json`,
summarizes matching pod counts, redacts credential values, omits raw stdout,
and records that it did not create, stop, delete, upload, or execute remote
commands.

To prepare the future copy-back step without copying anything, run:

```bash
.venv/Scripts/python.exe scripts/runpod/build_probe_artifact_handoff.py
.venv/Scripts/python.exe scripts/smoke/smoke_runpod_probe_artifact_handoff.py
```

This writes `results/final/runpod_probe_artifact_handoff.json`. It lists the
expected remote files under
`/workspace/exitvelo/results/runpod_mocapact_probe/`, placeholder `scp`
templates, clip priority, and the local validation sequence that must pass
before adapter analysis trusts copied-back rollout files.

To prepare teardown without stopping or deleting any pod, run:

```bash
.venv/Scripts/python.exe scripts/runpod/build_probe_cleanup_plan.py
.venv/Scripts/python.exe scripts/smoke/smoke_runpod_probe_cleanup_plan.py
.venv/Scripts/python.exe scripts/runpod/cleanup_mocapact_probe.py
.venv/Scripts/python.exe scripts/smoke/smoke_runpod_probe_guarded_cleanup.py
```

This writes `results/final/runpod_probe_cleanup_plan.json` and
`results/final/runpod_probe_cleanup_dryrun.json`. The plan is derived from
the latest live pod-status artifact. The guarded cleanup helper defaults to a
dry run; actual `runpodctl pod stop <pod-id>` and
`runpodctl pod delete <pod-id>` execution requires an explicit pod id from
the cleanup plan, matching pod name, and `--execute --i-understand-delete-pod`.

Do not upload local workspace files to a RunPod pod without that explicit
approval. After the probe finishes, delete the pod promptly.

Recommended isolated setup:

```bash
cd /workspace
git clone https://github.com/microsoft/MoCapAct.git
cd MoCapAct
python3.8 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -e .
```

Minimum useful probe:

1. Confirm the installed MoCapAct examples import on Linux.
2. Check whether CMU subject 124 / swing-adjacent clips are exposed through
   the dm_control mocap loader. Try `CMU_124_07` first, then `CMU_124_06`
   and `CMU_016_22` to separate target-clip absence from install failure.
3. Create a `MocapTrackingGymEnv` around the batting contact window.
4. Train or load a clip expert and export the tracked humanoid rollout.
5. Compare bat sweet-spot trajectory against Exitvelo's kinematic evaluator.

Success means MoCapAct becomes the physical-tracking base for residual RL.
Failure means keep MoCapAct as a reference and build a smaller DeepMimic-style
tracker directly around `assets/mujoco/cmu_batting_scene.xml`.

Automated probe path from the Exitvelo repo root:

```bash
bash scripts/runpod/mocapact_probe.sh
```

Custom exact-target probe path:

```bash
BUILD_CUSTOM_MOCAP_HDF5=1 bash scripts/runpod/mocapact_probe.sh
```

This path converts tracked raw
`data/raw/cmu_subject_124/124_07.amc` into
`results/mocapact_custom_hdf5/CMU_124_07_from_amc_30hz.h5`, pins the
MoCapAct venv to `numpy==1.26.4`, `protobuf==3.20.3`, and `h5py`, then probes
`CMU_124_07` through `--mocap_path`. The target window is `87-120` because
the custom file is exported at `dt=0.03`; this corresponds to Exitvelo's
100 Hz frames `260-360`.

Use the public-clone path for this custom branch. The older minimal upload
path was scoped for the two RunPod probe files and is not sufficient by itself
for regenerating the target HDF5 from tracked raw AMC data.

Default probe candidates now use clip-specific windows so fallback clips are
not accidentally tested with the batting contact window:

| candidate | window | role |
|---|---:|---|
| `CMU_124_07` | `260-360` | exact Exitvelo batting target |
| `CMU_124_08` | `260-360` | second local subject-124 raw clip |
| `CMU_124_06` | `0-189` | subject-124 clip present in MoCapAct split files |
| `CMU_016_22` | `0-82` | known MoCapAct control clip |

Override this with:

```bash
PROBE_CANDIDATES="CMU_124_07:260:360:target CMU_016_22:0:82:control" \
  bash scripts/runpod/mocapact_probe.sh
```

Set `CLIP_IDS="..." START_STEP=... END_STEP=...` only when intentionally
applying one global window to every clip.

Expected outputs:

```text
results/runpod_mocapact_probe/mocapact_probe_report.json
results/runpod_mocapact_probe/<clip_id>_rollout.json
```

After copying `mocapact_probe_report.json` back locally, classify the result
without importing MoCapAct:

```bash
.venv/Scripts/python.exe scripts/runpod/validate_probe_artifact_intake.py
.venv/Scripts/python.exe scripts/smoke/smoke_runpod_probe_artifact_intake.py
.venv/Scripts/python.exe scripts/smoke/smoke_runpod_post_probe_end_to_end.py
.venv/Scripts/python.exe scripts/analysis/analyze_mocapact_probe_report.py
.venv/Scripts/python.exe scripts/smoke/smoke_mocapact_probe_decision.py
```

The intake gate writes `results/final/runpod_probe_artifact_intake.json` and
`results/analysis/runpod_probe_artifact_intake/summary.*`. It checks the
copied-back report, referenced rollout files, rollout schema, hashes, and
adapter-readiness. It performs no remote actions and does not copy or upload
files.

This writes `results/analysis/mocapact_probe_decision/summary.json`. Without
a runtime report it records `pending_external_probe_report` /
`await_linux_probe`. With a runtime report it decides whether to use the exact
`CMU_124_07` rollout, retarget/debug around `CMU_124_08`, validate the adapter
on `CMU_124_06`, treat `CMU_016_22` as install-only success, or debug the
MoCapAct install before more batting work.

After copying a successful `*_rollout.json` back locally, run the adapter
gates:

```bash
.venv/Scripts/python.exe scripts/io/export_mocapact_proxy_trajectory.py \
  --rollout results/runpod_mocapact_probe/CMU_124_07_rollout.json \
  --out results/runpod_mocapact_probe/CMU_124_07_proxy.json

.venv/Scripts/python.exe scripts/analysis/compare_mocapact_proxy_to_kinematic.py \
  --proxy results/runpod_mocapact_probe/CMU_124_07_proxy.json \
  --start-frame 260 \
  --max-hand-rmse-m 0.20 \
  --max-any-error-m 0.50 \
  --fail-on-threshold

.venv/Scripts/python.exe scripts/analysis/evaluate_proxy_batting.py \
  --proxy results/runpod_mocapact_probe/CMU_124_07_proxy.json \
  --out results/runpod_mocapact_probe/CMU_124_07_proxy_batting.json
```

## Cost estimates

| Job | Local RTX 2060 | RunPod 4090 community ($0.34/hr) |
|---|---|---|
| Tier 2A CMA-ES full | ~3 min | ~3 min (about $0.02) - local is faster + free |
| Tier 2B PPO 50k single-env | ~25 min | ~25 min ($0.14) |
| Tier 2B PPO 50k x 8 vec envs | ~3.5 min | ~3.5 min ($0.02) |
| Future: PPO 1M steps x 8 vec | ~8 hr | ~50 min ($0.30) |
| Future: physics RL 10M steps | impractical | ~6 hr ($2.04) |
| Hyperparam sweep (10 configs x 1M) | impractical | ~8 hr ($2.72) |

## Gotchas

- **Line endings**: always `git clone` fresh on the pod. Do not rsync a
  Windows working tree - CRLF endings break shell scripts and confuse
  Python's `__pycache__`.
- **Mocap data**: `data/raw/cmu_subject_124/{124.asf,124_07.amc,124_08.amc}`
  must exist before any rollout. `scripts/io/download_cmu_data.py`
  fetches them.
- **MuJoCo viewer**: do NOT try `mujoco.viewer.launch()` on a headless
  pod. Render offscreen with `mujoco.Renderer` and dump frames to GIF
  via `imageio`.
- **EGL vs OSMesa**: `MUJOCO_GL=egl` works on most NVIDIA pods. If you
  get a `Failed to create OpenGL context` error, fall back to
  `MUJOCO_GL=osmesa` (slower but reliable).
- **GPU detection without GPU work**: `torch.cuda.is_available()` should
  return `True` on the pod. If it doesn't, the pod was provisioned
  without a GPU - re-launch.
- **Pod auto-shutdown**: community pods can be preempted. For runs that
  matter, use secure-cloud or save checkpoints frequently.
- **Don't pay for idle**: stop the pod when you're not using it. RunPod
  bills by the second.

## Snapshot back to git

After a successful run, push the artifacts you want to keep. Big files
(`*.mp4`, `*.zip` model checkpoints) should go to a release or a
separate storage bucket - not into the git repo.

```bash
git add results/<your_run_dir>/best.json results/<your_run_dir>/generations.csv
git commit -m "Tier 2X: <run name> results"
git push
```
