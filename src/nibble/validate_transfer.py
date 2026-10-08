"""Require every requested replay transition; preserve partial artifacts on failure."""

import argparse
import json
from pathlib import Path


def validate(root, updates, count):
    for version in range(updates):
        directory = Path(root) / f"v{version}-v{version + 1}"
        summary = json.loads((directory / "summary.json").read_text())
        rows = json.loads((directory / "rows.json").read_text())
        if (
            summary["source_version"],
            summary["target_version"],
            summary["requested"],
        ) != (version, version + 1, count):
            raise ValueError(f"{directory}: incomplete or incorrect transition request")
        if len(rows) != summary["interrupted"] or not rows:
            raise ValueError(f"{directory}: no complete interrupted-request comparison")
    # Cache usage is a separate empirical finding, not manufactured by validation.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    validate(args.root, args.updates, args.count)


if __name__ == "__main__":
    main()
