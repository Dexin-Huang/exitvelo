"""Build a local media checkpoint package for the batting project.

This script deliberately does not generate the final report. It collects the
videos, JSON summaries, residual checkpoints, thumbnails, plots, and catalogs
needed to write the report later in a separate fixed format.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[2]
DOWNLOADS = REPO / "results" / "downloads"
OUT = REPO / "results" / "media_checkpoint"
VIDEOS = OUT / "videos"
STILLS = OUT / "stills"
FIGURES = OUT / "figures"
DATA = OUT / "data"
RESIDUALS = DATA / "residuals"
CATALOG = OUT / "catalog"


@dataclass(frozen=True)
class VideoSpec:
    slug: str
    source: str
    title: str
    role: str
    notes: str


VIDEOS_TO_PACKAGE = [
    VideoSpec(
        "01_source_mocap_trusted_replay",
        "canonical_source_CMU_124_07_trusted_replay_clean_frames260_359.mp4",
        "Source CMU mocap replay",
        "source",
        "Canonical batting motion segment used as the human reference.",
    ),
    VideoSpec(
        "02_physical_imitation_clean",
        "CMU_124_07_stride1_residual_knots12_speedkick3860_physical_only.mp4",
        "Approved physical imitation",
        "imitation",
        "MoCapAct physical tracker candidate without bat or ball.",
    ),
    VideoSpec(
        "03_physical_imitation_overlay",
        "CMU_124_07_stride1_residual_knots12_speedkick3860_best_with_ghost.mp4",
        "Physical imitation with ghost reference",
        "imitation audit",
        "Overlay used to visually approve that the physical body tracks the swing.",
    ),
    VideoSpec(
        "04_virtual_bat_tball",
        "CMU_124_07_virtual_bat_tball_speedkick3860_twohand_visible_kinematic_bat.mp4",
        "Virtual bat tee-ball search",
        "task warmup",
        "Early task-objective stage before massful bat/contact was introduced.",
    ),
    VideoSpec(
        "05_massful_old_bat_clean",
        "CMU_124_07_old_bat_asset_geom_stage2_twohand_stride1_clean.mp4",
        "Massful old bat attachment",
        "asset validation",
        "Original bat mesh attached to the physical hand with no ball.",
    ),
    VideoSpec(
        "06_physical_tball_launchfloor",
        "CMU_124_07_physical_tball_distance_cont_pop14_iter4_launchfloor.mp4",
        "Physical tee-ball launch correction",
        "contact result",
        "Best tee-ball carry checkpoint: launch angle becomes slightly positive.",
    ),
    VideoSpec(
        "07_physical_tball_speedrecover",
        "CMU_124_07_physical_tball_distance_speedrecover_pop18_iter5.mp4",
        "Physical tee-ball speed-recovery attempt",
        "negative result",
        "CEM can shape launch but cannot recover enough bat speed with the current tracker.",
    ),
    VideoSpec(
        "08_batspeed_imitation_scale030",
        "CMU_124_07_batspeed_imitation_scale030.mp4",
        "No-ball bat-speed imitation pass",
        "diagnostic result",
        "No-ball speed gate run improves contact speed to 7.55 m/s.",
    ),
    VideoSpec(
        "09_batspeed_imitation_scale035_cont_iter2",
        "CMU_124_07_batspeed_imitation_scale035_cont_iter2.mp4",
        "No-ball bat-speed continuation checkpoint",
        "diagnostic result",
        "Stopped checkpoint improves contact speed to 8.30 m/s, still below the 10 m/s gate.",
    ),
]


DATA_FILES = [
    "canonical_source_CMU_124_07_trusted_replay_clean_frames260_359_manifest.json",
    "CMU_124_07_stride1_residual_knots12_speedkick3860_summary.json",
    "CMU_124_07_stride1_residual_knots12_speedkick3860_physical_only_summary.json",
    "CMU_124_07_virtual_bat_tball_speedkick3860_twohand_summary.json",
    "CMU_124_07_virtual_bat_tball_speedkick3860_twohand_visible_kinematic_bat_summary.json",
    "CMU_124_07_old_bat_asset_geom_stage2_twohand_stride1_clean_summary.json",
    "CMU_124_07_physical_tball_distance_pop12_iter4_summary.json",
    "CMU_124_07_physical_tball_distance_cont_pop14_iter4_launchfloor_summary.json",
    "CMU_124_07_physical_tball_distance_speedrecover_pop18_iter5_summary.json",
    "CMU_124_07_batspeed_kinematic_target.json",
    "CMU_124_07_batspeed_imitation_scale030_summary.json",
    "CMU_124_07_batspeed_imitation_scale035_cont_partial_summary.json",
    "CMU_124_07_batspeed_imitation_scale035_cont_iter2_summary.json",
]


RESIDUAL_FILES = [
    "CMU_124_07_stride1_residual_knots12_speedkick3860_best_residual.npy",
    "CMU_124_07_virtual_bat_tball_speedkick3860_twohand_best_residual.npy",
    "CMU_124_07_physical_tball_distance_cont_pop14_iter4_launchfloor_best_residual.npy",
    "CMU_124_07_physical_tball_distance_speedrecover_pop18_iter5_best_residual.npy",
    "CMU_124_07_batspeed_imitation_scale030_best_residual.npy",
    "CMU_124_07_batspeed_imitation_scale035_cont_iter2_best_residual.npy",
]


CODE_PATHS: list[tuple[str, str]] = [
    ("scripts/io/render_cmu_trusted_replay.py", "source mocap replay renderer"),
    ("scripts/runpod/export_cmu12407_mocapact_hdf5.py", "CMU 124_07 export into MoCapAct HDF5 format"),
    ("scripts/runpod/train_cmu_samebody_imitation_ppo.py", "same-body physical imitation PPO experiments"),
    ("scripts/runpod/search_mocapact_residual_bias_cem.py", "early CEM residual tracking search"),
    ("scripts/runpod/search_mocapact_speed_residual_cem.py", "bat-speed oriented residual search"),
    ("scripts/runpod/render_mocapact_old_bat_asset.py", "massful old-bat scene rendering and telemetry"),
    ("scripts/runpod/search_mocapact_virtual_tball_cem.py", "virtual tee-ball task-objective search"),
    ("scripts/runpod/search_mocapact_physical_tball_cem.py", "massful physical tee-ball distance CEM"),
    ("scripts/runpod/search_mocapact_batspeed_imitation_cem.py", "no-ball bat-speed imitation gate search"),
    ("src/motion/cmu_replay.py", "CMU ASF/AMC playback and qpos construction"),
    ("src/motion/mocapact_rollout.py", "local rollout utilities for MoCapAct-style outputs"),
    ("src/optim/kinematic_evaluator.py", "kinematic batting evaluator"),
    ("src/env/contacts.py", "Nathan bat-ball model and flight helpers"),
    ("assets/mujoco/cmu_batting_scene.xml", "kinematic batting scene with bat sites"),
]


def ensure_dirs() -> None:
    for path in [VIDEOS, STILLS, FIGURES, DATA, RESIDUALS, CATALOG]:
        path.mkdir(parents=True, exist_ok=True)


def load_json(name: str) -> dict[str, Any]:
    with (DOWNLOADS / name).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return max(float(result.stdout.strip()), 0.1)


def extract_frame(video: Path, output: Path, time_s: float) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{time_s:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output),
        ]
    )


def make_contact_sheet(video: Path, output: Path, label: str, frames: int = 6) -> None:
    tmp = output.parent / f".{output.stem}_frames"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    duration = ffprobe_duration(video)
    frame_paths: list[Path] = []
    for i in range(frames):
        t = duration * (i + 0.5) / frames
        frame_path = tmp / f"frame_{i:02d}.png"
        extract_frame(video, frame_path, t)
        frame_paths.append(frame_path)

    images = [Image.open(p).convert("RGB") for p in frame_paths]
    target_w = 320
    target_h = int(images[0].height * (target_w / images[0].width))
    thumbs = [img.resize((target_w, target_h), Image.Resampling.LANCZOS) for img in images]
    pad = 16
    title_h = 42
    sheet = Image.new("RGB", (frames * target_w + (frames + 1) * pad, target_h + title_h + 2 * pad), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((pad, pad), label, fill=(20, 24, 33), font=font)
    for i, img in enumerate(thumbs):
        x = pad + i * (target_w + pad)
        y = pad + title_h
        sheet.paste(img, (x, y))
        draw.text((x, y + target_h + 2), f"t{i + 1}", fill=(70, 78, 92), font=font)
    sheet.save(output)
    shutil.rmtree(tmp)


def copy_media() -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for spec in VIDEOS_TO_PACKAGE:
        source = DOWNLOADS / spec.source
        if not source.exists():
            raise FileNotFoundError(source)
        dest = VIDEOS / f"{spec.slug}.mp4"
        shutil.copy2(source, dest)
        still = STILLS / f"{spec.slug}_still.png"
        sheet = STILLS / f"{spec.slug}_contact_sheet.png"
        duration = ffprobe_duration(dest)
        extract_frame(dest, still, duration * 0.55)
        make_contact_sheet(dest, sheet, spec.title)
        manifest.append(
            {
                "slug": spec.slug,
                "title": spec.title,
                "role": spec.role,
                "notes": spec.notes,
                "source": str(source.relative_to(REPO)),
                "video": str(dest.relative_to(REPO)),
                "still": str(still.relative_to(REPO)),
                "contact_sheet": str(sheet.relative_to(REPO)),
                "duration_s": duration,
                "bytes": dest.stat().st_size,
            }
        )

    for name in DATA_FILES:
        source = DOWNLOADS / name
        if source.exists():
            shutil.copy2(source, DATA / name)

    for name in RESIDUAL_FILES:
        source = DOWNLOADS / name
        if source.exists():
            shutil.copy2(source, RESIDUALS / name)

    return manifest


def diag(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get("diagnostics") or {}


def baseline_diag(summary: dict[str, Any]) -> dict[str, Any]:
    return (summary.get("baseline") or {}).get("diagnostics") or {}


def build_metrics_snapshot() -> dict[str, Any]:
    tball_pop12 = load_json("CMU_124_07_physical_tball_distance_pop12_iter4_summary.json")
    tball_launch = load_json("CMU_124_07_physical_tball_distance_cont_pop14_iter4_launchfloor_summary.json")
    speedrecover = load_json("CMU_124_07_physical_tball_distance_speedrecover_pop18_iter5_summary.json")
    speed030 = load_json("CMU_124_07_batspeed_imitation_scale030_summary.json")
    speed035 = load_json("CMU_124_07_batspeed_imitation_scale035_cont_iter2_summary.json")
    target = load_json("CMU_124_07_batspeed_kinematic_target.json")
    partial035 = load_json("CMU_124_07_batspeed_imitation_scale035_cont_partial_summary.json")

    metrics = {
        "tee_ball_stages": [
            {
                "stage": "physical tee baseline",
                "carry_ft": baseline_diag(tball_pop12).get("carry_ft"),
                "exit_speed_mph": baseline_diag(tball_pop12).get("exit_speed_mph"),
                "launch_angle_deg": baseline_diag(tball_pop12).get("launch_angle_deg"),
                "contact_step": baseline_diag(tball_pop12).get("first_bat_ball_contact_step"),
            },
            {
                "stage": "first physical CEM",
                "carry_ft": diag(tball_pop12).get("carry_ft"),
                "exit_speed_mph": diag(tball_pop12).get("exit_speed_mph"),
                "launch_angle_deg": diag(tball_pop12).get("launch_angle_deg"),
                "contact_step": diag(tball_pop12).get("first_bat_ball_contact_step"),
            },
            {
                "stage": "launch correction",
                "carry_ft": diag(tball_launch).get("carry_ft"),
                "exit_speed_mph": diag(tball_launch).get("exit_speed_mph"),
                "launch_angle_deg": diag(tball_launch).get("launch_angle_deg"),
                "contact_step": diag(tball_launch).get("first_bat_ball_contact_step"),
            },
            {
                "stage": "speed recovery",
                "carry_ft": diag(speedrecover).get("carry_ft"),
                "exit_speed_mph": diag(speedrecover).get("exit_speed_mph"),
                "launch_angle_deg": diag(speedrecover).get("launch_angle_deg"),
                "contact_step": diag(speedrecover).get("first_bat_ball_contact_step"),
                "bat_speed_mps": diag(speedrecover).get("bat_speed_mps"),
            },
        ],
        "bat_speed_gate": {
            "kinematic_contact_mps": target.get("bat_sweet_contact_speed_mps"),
            "kinematic_peak_mps": target.get("bat_sweet_peak_speed_mps"),
            "gate_mps": 10.0,
            "physical_tball_speedrecover_mps": diag(speedrecover).get("bat_speed_mps"),
            "no_ball_scale030_contact_mps": diag(speed030).get("current_contact_speed_mps"),
            "no_ball_scale030_peak_mps": diag(speed030).get("current_peak_speed_mps"),
            "no_ball_scale035_contact_mps": diag(speed035).get("current_contact_speed_mps"),
            "no_ball_scale035_peak_mps": diag(speed035).get("current_peak_speed_mps"),
            "physical_contact_step": target.get("physical_contact_step"),
            "kinematic_contact_step": target.get("kinematic_contact_step"),
        },
        "traces": {
            "tball_launch": tball_launch.get("trace", []),
            "tball_speedrecover": speedrecover.get("trace", []),
            "batspeed_scale030": speed030.get("trace", []),
            "batspeed_scale035_partial": partial035.get("trace", []),
        },
    }
    write_json(CATALOG / "metrics_snapshot.json", metrics)
    return metrics


def plot_tee_ball(metrics: dict[str, Any]) -> None:
    stages = metrics["tee_ball_stages"]
    labels = [s["stage"] for s in stages]
    carry = [s["carry_ft"] for s in stages]
    exit_speed = [s["exit_speed_mph"] for s in stages]
    angle = [s["launch_angle_deg"] for s in stages]
    x = range(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    fig.suptitle("Physical tee-ball checkpoints", fontsize=15, fontweight="bold")
    axes[0].bar(x, carry, color="#3f6f5f")
    axes[0].set_ylabel("Carry (ft)")
    axes[1].bar(x, exit_speed, color="#5f6f9f")
    axes[1].set_ylabel("Exit speed (mph)")
    axes[2].bar(x, angle, color="#9f6f3f")
    axes[2].axhline(0, color="#222222", linewidth=1)
    axes[2].set_ylabel("Launch angle (deg)")
    axes[2].set_xticks(list(x), labels, rotation=18, ha="right")
    for ax in axes:
        ax.grid(axis="y", color="#d7dbe2", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "tee_ball_checkpoint_metrics.png", dpi=200)
    plt.close(fig)


def plot_bat_speed_gate(metrics: dict[str, Any]) -> None:
    gate = metrics["bat_speed_gate"]
    labels = [
        "physical tee-ball",
        "no-ball pass 1",
        "no-ball continuation",
        "kinematic target",
    ]
    values = [
        gate["physical_tball_speedrecover_mps"],
        gate["no_ball_scale030_contact_mps"],
        gate["no_ball_scale035_contact_mps"],
        gate["kinematic_contact_mps"],
    ]
    colors = ["#8b6f47", "#5b7c99", "#315f72", "#333333"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, values, color=colors)
    ax.axhline(gate["gate_mps"], color="#9f3333", linewidth=2, linestyle="--", label="10 m/s resume gate")
    ax.set_ylabel("Bat sweet-spot contact speed (m/s)")
    ax.set_title("Bat-speed bottleneck before distance optimization", fontsize=14, fontweight="bold")
    ax.grid(axis="y", color="#d7dbe2", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    for tick in ax.get_xticklabels():
        tick.set_rotation(12)
        tick.set_ha("right")
    fig.tight_layout()
    fig.savefig(FIGURES / "bat_speed_gate.png", dpi=200)
    plt.close(fig)


def plot_batspeed_trace(metrics: dict[str, Any]) -> None:
    trace030 = metrics["traces"]["batspeed_scale030"]
    trace035 = metrics["traces"]["batspeed_scale035_partial"]
    rows = []
    for row in trace030:
        rows.append(("scale030", row["iteration"], row))
    offset = len(rows)
    for row in trace035:
        rows.append(("scale035", offset + row["iteration"], row))

    x = [r[1] for r in rows]
    peak = [r[2]["best_peak_speed_mps"] for r in rows]
    contact = [r[2]["best_contact_speed_mps"] for r in rows]
    target = rows[-1][2]["best_target_contact_speed_mps"] if rows else 17.35

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x, peak, marker="o", color="#315f72", label="peak physical speed")
    ax.plot(x, contact, marker="o", color="#8b6f47", label="contact physical speed")
    ax.axhline(10.0, color="#9f3333", linewidth=2, linestyle="--", label="10 m/s resume gate")
    ax.axhline(target, color="#333333", linewidth=1.5, linestyle=":", label="kinematic target")
    ax.set_xlabel("CEM iteration index")
    ax.set_ylabel("Bat sweet-spot speed (m/s)")
    ax.set_title("No-ball bat-speed imitation progress", fontsize=14, fontweight="bold")
    ax.grid(axis="y", color="#d7dbe2", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "batspeed_cem_trace.png", dpi=200)
    plt.close(fig)


def plot_pipeline_map() -> None:
    stages = [
        ("CMU mocap", "human reference"),
        ("physical imitation", "stable body motion"),
        ("bat attachment", "massful old asset"),
        ("tee-ball contact", "MuJoCo ball contact"),
        ("speed gate", "bat-site velocity bottleneck"),
    ]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axis("off")
    x0 = 0.05
    width = 0.16
    gap = 0.035
    for i, (title, subtitle) in enumerate(stages):
        x = x0 + i * (width + gap)
        rect = plt.Rectangle((x, 0.32), width, 0.36, transform=ax.transAxes, facecolor="#f4f1ea", edgecolor="#9aa3ad")
        ax.add_patch(rect)
        ax.text(x + width / 2, 0.55, title, transform=ax.transAxes, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x + width / 2, 0.43, subtitle, transform=ax.transAxes, ha="center", va="center", fontsize=8, color="#5b6470")
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + width + gap * 0.75, 0.50),
                xytext=(x + width + gap * 0.15, 0.50),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "color": "#4d5560", "lw": 1.5},
            )
    ax.text(0.05, 0.82, "Experiment media sequence", transform=ax.transAxes, fontsize=14, fontweight="bold")
    ax.text(
        0.05,
        0.16,
        "The package separates visual proof of tracking/contact from the unresolved speed bottleneck.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4d5560",
    )
    fig.savefig(FIGURES / "pipeline_media_map.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def code_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path_str, role in CODE_PATHS:
        path = REPO / path_str
        row: dict[str, Any] = {"path": path_str, "role": role, "exists": path.exists()}
        if path.exists() and path.is_file():
            data = path.read_bytes()
            row["bytes"] = len(data)
            row["lines"] = data.count(b"\n") + 1
            row["sha256_12"] = hashlib.sha256(data).hexdigest()[:12]
        rows.append(row)
    write_json(CATALOG / "code_inventory.json", rows)
    return rows


def write_markdown_catalogs(manifest: list[dict[str, Any]], metrics: dict[str, Any], inventory: list[dict[str, Any]]) -> None:
    artifact_lines = [
        "# Media Checkpoint Artifact Catalog",
        "",
        "Purpose: organize local media and data for later report writing. This is not the report.",
        "",
        "## Selected Videos",
        "",
        "| Slug | Role | Video | Still | Contact sheet | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in manifest:
        artifact_lines.append(
            f"| {item['slug']} | {item['role']} | `{item['video']}` | `{item['still']}` | `{item['contact_sheet']}` | {item['notes']} |"
        )
    artifact_lines += [
        "",
        "## Key Metrics",
        "",
        f"- Kinematic bat sweet-spot contact target: `{metrics['bat_speed_gate']['kinematic_contact_mps']:.2f} m/s`.",
        f"- Current best no-ball physical contact speed: `{metrics['bat_speed_gate']['no_ball_scale035_contact_mps']:.2f} m/s`.",
        f"- Resume-gate target before tee-ball distance optimization: `{metrics['bat_speed_gate']['gate_mps']:.2f} m/s`.",
        f"- Best physical tee-ball carry checkpoint: `{metrics['tee_ball_stages'][2]['carry_ft']:.2f} ft`.",
        "",
        "## Generated Figures",
        "",
        "- `results/media_checkpoint/figures/tee_ball_checkpoint_metrics.png`",
        "- `results/media_checkpoint/figures/bat_speed_gate.png`",
        "- `results/media_checkpoint/figures/batspeed_cem_trace.png`",
        "- `results/media_checkpoint/figures/pipeline_media_map.png`",
        "",
        "## Data Files",
        "",
        "- JSON summaries copied to `results/media_checkpoint/data/`.",
        "- Residual checkpoints copied to `results/media_checkpoint/data/residuals/`.",
        "- Consolidated metrics snapshot: `results/media_checkpoint/catalog/metrics_snapshot.json`.",
    ]
    (CATALOG / "artifact_catalog.md").write_text("\n".join(artifact_lines) + "\n", encoding="utf-8")

    code_lines = [
        "# Code Lineage Catalog",
        "",
        "Purpose: identify the code paths behind the generated media and experiment claims. This is not the report.",
        "",
        "| Path | Role | Lines | SHA-256 prefix |",
        "| --- | --- | ---: | --- |",
    ]
    for item in inventory:
        if item["exists"]:
            code_lines.append(
                f"| `{item['path']}` | {item['role']} | {item.get('lines', '')} | `{item.get('sha256_12', '')}` |"
            )
        else:
            code_lines.append(f"| `{item['path']}` | {item['role']} | missing | missing |")
    (CATALOG / "code_catalog.md").write_text("\n".join(code_lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    manifest = copy_media()
    write_json(CATALOG / "media_manifest.json", manifest)
    metrics = build_metrics_snapshot()
    plot_tee_ball(metrics)
    plot_bat_speed_gate(metrics)
    plot_batspeed_trace(metrics)
    plot_pipeline_map()
    inventory = code_inventory()
    write_markdown_catalogs(manifest, metrics, inventory)
    print(f"Wrote media checkpoint to {OUT}")


if __name__ == "__main__":
    main()
