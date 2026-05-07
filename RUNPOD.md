# RunPod / GPU setup

Cloud setup notes for pod-based work. **Today's pipeline does NOT need
a pod** — kinematic eval runs in 2–3 minutes on a laptop CPU. This doc
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

Verify rates at https://www.runpod.io/pricing — these are May 2026.

| Pod | $/hr | When |
|---|---|---|
| **RTX 4090 community** | ~$0.34 | dm_control + MuJoCo (CPU-bound). Start here. |
| RTX 4090 secure cloud | ~$0.69 | Long runs where preemption hurts |
| A6000 48GB | ~$1.22 | Larger replay buffers, bigger nets |
| A100 80GB | $1.89–4.18 | Only worth it if env is ported to MJX/JAX |

dm_control's MuJoCo backend is **CPU-bound** — env stepping doesn't use
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
# 1. Clone fresh — do not rsync a CRLF working tree from Windows
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

# Future: full DeepMimic-style physics RL — not implemented yet.
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

## Cost estimates

| Job | Local RTX 2060 | RunPod 4090 community ($0.34/hr) |
|---|---|---|
| Tier 2A CMA-ES full | ~3 min | ~3 min (≈ $0.02) — local is faster + free |
| Tier 2B PPO 50k single-env | ~25 min | ~25 min ($0.14) |
| Tier 2B PPO 50k × 8 vec envs | ~3.5 min | ~3.5 min ($0.02) |
| Future: PPO 1M steps × 8 vec | ~8 hr | ~50 min ($0.30) |
| Future: physics RL 10M steps | impractical | ~6 hr ($2.04) |
| Hyperparam sweep (10 configs × 1M) | impractical | ~8 hr ($2.72) |

## Gotchas

- **Line endings**: always `git clone` fresh on the pod. Do not rsync a
  Windows working tree — CRLF endings break shell scripts and confuse
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
  without a GPU — re-launch.
- **Pod auto-shutdown**: community pods can be preempted. For runs that
  matter, use secure-cloud or save checkpoints frequently.
- **Don't pay for idle**: stop the pod when you're not using it. RunPod
  bills by the second.

## Snapshot back to git

After a successful run, push the artifacts you want to keep. Big files
(`*.mp4`, `*.zip` model checkpoints) should go to a release or a
separate storage bucket — not into the git repo.

```bash
git add results/<your_run_dir>/best.json results/<your_run_dir>/generations.csv
git commit -m "Tier 2X: <run name> results"
git push
```
