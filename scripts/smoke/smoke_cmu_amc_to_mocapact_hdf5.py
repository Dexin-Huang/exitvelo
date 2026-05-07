"""Smoke test for the AMC -> MoCapAct HDF5 exporter."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out = tmp_path / "CMU_124_07_smoke.h5"
        summary_path = tmp_path / "summary.json"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/io/export_cmu_amc_to_mocapact_hdf5.py",
                "--out",
                str(out),
                "--summary",
                str(summary_path),
                "--max-frames",
                "48",
                "--overwrite",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        assert "Wrote MoCapAct HDF5" in result.stdout
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["status"] == "written"
    assert summary["clip_id"] == "CMU_124_07"
    assert summary["num_steps"] == 48
    assert summary["control_dt"] == 0.03
    assert summary["feature_shapes"]["joints"] == [48, 56]
    assert summary["feature_shapes"]["body_positions"] == [48, 93]
    assert summary["feature_shapes"]["body_quaternions"] == [48, 124]
    assert summary["validation"]["tracking_reset_ok"] is True
    assert summary["validation"]["start_step"] == 0
    assert summary["validation"]["tracking_initial_termination_error"] < 1e-8
    assert summary["validation"]["dict_shapes"]["walker/joints"] == [48, 56]
    assert summary["validation"]["dict_shapes"]["walker/body_positions"] == [48, 31, 3]
    assert "CMU_124_07" in summary["validation"]["loader_keys"]
    assert summary["target_window_validation"] is None

    print("PASS cmu amc to mocapact hdf5 smoke")


if __name__ == "__main__":
    main()
