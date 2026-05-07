"""Preflight the RunPod/MoCapAct branch without exposing credentials.

This script answers a narrow handoff question: can this shell launch or
support the next-stage MoCapAct pod workflow? It deliberately records only
booleans and tool paths, never API key values.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "results" / "final" / "runpod_readiness_report.json"
AUTH_ENV_NAMES = ("RUNPOD_API_KEY", "RUNPOD_API_TOKEN")
TOOL_NAMES = ("runpodctl", "git", "ssh", "bash")


def _tool_report(name: str) -> dict:
    path = shutil.which(name)
    return {
        "name": name,
        "available": path is not None,
        "path": path,
    }


def _auth_report() -> dict:
    present = {name: bool(os.environ.get(name)) for name in AUTH_ENV_NAMES}
    return {
        "any_runpod_auth_env_present": any(present.values()),
        "checked_env_names": list(AUTH_ENV_NAMES),
        "present": present,
        "values_redacted": True,
    }


def build_report() -> dict:
    tools = {name: _tool_report(name) for name in TOOL_NAMES}
    auth = _auth_report()
    required_files = {
        "mocapact_probe_py": PROJECT_ROOT / "scripts" / "runpod" / "mocapact_probe.py",
        "mocapact_probe_sh": PROJECT_ROOT / "scripts" / "runpod" / "mocapact_probe.sh",
        "runpod_notes": PROJECT_ROOT / "RUNPOD.md",
        "adapter_spec": PROJECT_ROOT / "MOCAPACT_ADAPTER_SPEC.md",
    }
    files = {
        name: {"path": str(path), "exists": path.exists()}
        for name, path in required_files.items()
    }

    can_launch_from_shell = (
        auth["any_runpod_auth_env_present"]
        and tools["runpodctl"]["available"]
        and tools["git"]["available"]
        and tools["ssh"]["available"]
    )
    can_run_probe_locally = platform.system().lower() == "linux"
    ready_for_mocapact_probe = can_launch_from_shell or can_run_probe_locally

    blockers: list[str] = []
    if not auth["any_runpod_auth_env_present"]:
        blockers.append("No RUNPOD_API_KEY or RUNPOD_API_TOKEN is configured in this shell.")
    if not tools["runpodctl"]["available"]:
        blockers.append("The runpod CLI is not available on PATH.")
    if not tools["git"]["available"]:
        blockers.append("git is not available on PATH.")
    if not tools["ssh"]["available"]:
        blockers.append("ssh is not available on PATH.")
    missing_files = [name for name, item in files.items() if not item["exists"]]
    for name in missing_files:
        blockers.append(f"Missing required RunPod handoff file: {name}.")

    next_actions = [
        "Install/configure the runpod CLI or launch a pod manually from the RunPod UI.",
        "Set RUNPOD_API_KEY or RUNPOD_API_TOKEN in the launch shell if using the CLI.",
        "On the pod, run: bash scripts/runpod/mocapact_probe.sh",
        "Copy any *_rollout.json back and run the local adapter gates from RUNPOD.md.",
    ]

    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "project_root": str(PROJECT_ROOT),
        "auth": auth,
        "tools": tools,
        "required_files": files,
        "can_launch_from_shell": can_launch_from_shell,
        "can_run_probe_locally": can_run_probe_locally,
        "ready_for_mocapact_probe": ready_for_mocapact_probe,
        "blockers": blockers,
        "next_actions": next_actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit nonzero unless this shell can launch/run the MoCapAct probe.",
    )
    args = parser.parse_args()

    report = build_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(
        "runpod readiness: "
        f"launch_from_shell={report['can_launch_from_shell']} "
        f"run_probe_locally={report['can_run_probe_locally']} "
        f"ready={report['ready_for_mocapact_probe']}"
    )
    if report["blockers"]:
        print("blockers:")
        for blocker in report["blockers"]:
            print(f"- {blocker}")
    if args.require_ready and not report["ready_for_mocapact_probe"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
