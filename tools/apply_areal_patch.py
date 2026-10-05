"""Apply the Nibble adapter only to the exact pinned source files, or verify it.

Usage: python tools/apply_areal_patch.py /path/to/areal-040cbfd5 [--verify]
No shared source tree is selected automatically.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    manifest = json.loads((project / "patches/manifest.json").read_text())
    key = "patched" if args.verify else "original"
    for rel, hashes in manifest["files"].items():
        actual = hashlib.sha256((args.source / rel).read_bytes()).hexdigest()
        if actual != hashes[key]:
            raise RuntimeError(
                f"{rel} does not match the pinned {key} hash; refusing to proceed"
            )
    if not args.verify:
        subprocess.run(
            [
                "patch",
                "-p1",
                "--batch",
                "--forward",
                "-i",
                str(project / "patches/areal-score-centering.patch"),
            ],
            cwd=args.source,
            check=True,
        )
        (args.source / "NIBBLE_PATCHED").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    print(f"Verified {len(manifest['files'])} adapter source files")


if __name__ == "__main__":
    main()
