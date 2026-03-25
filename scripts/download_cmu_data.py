"""Download CMU Motion Capture Database Subject 124 (baseball motions).

Downloads:
  - 124.asf  : skeleton definition
  - 124_07.amc : baseball swing (primary clip)
  - 124_08.amc : baseball bunt (backup)

Sources tried in order:
  1. Direct CMU server
  2. GitHub mirror (una-dinosauria/cmu-mocap)
"""

from pathlib import Path
import urllib.request
import urllib.error
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cmu_subject_124"

# Ordered list of (filename, [url_option1, url_option2, ...])
FILES = [
    (
        "124.asf",
        [
            "http://mocap.cs.cmu.edu/subjects/124/124.asf",
            "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/all_asfamc/subjects/124/124.asf",
        ],
    ),
    (
        "124_07.amc",
        [
            "http://mocap.cs.cmu.edu/subjects/124/124_07.amc",
            "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/all_asfamc/subjects/124/124_07.amc",
        ],
    ),
    (
        "124_08.amc",
        [
            "http://mocap.cs.cmu.edu/subjects/124/124_08.amc",
            "https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/all_asfamc/subjects/124/124_08.amc",
        ],
    ),
]


def download_file(filename: str, urls: list[str], dest_dir: Path) -> Path:
    """Try each URL in order until one succeeds. Returns path to saved file."""
    dest = dest_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {filename} already exists ({dest.stat().st_size:,} bytes)")
        return dest

    for url in urls:
        print(f"  [try]  {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            dest.write_bytes(data)
            print(f"  [ok]   {filename} downloaded ({len(data):,} bytes)")
            return dest
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            print(f"  [fail] {exc}")
            time.sleep(1)

    raise RuntimeError(f"All download sources failed for {filename}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Download directory: {DATA_DIR}\n")

    for filename, urls in FILES:
        print(f"Downloading {filename} ...")
        download_file(filename, urls, DATA_DIR)
        print()

    print("All files downloaded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
