"""One native training process; launched only by the bounded supervisor."""

import argparse
import json
from pathlib import Path
from .runtime import Runtime
from .spec import TrainSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    runtime = Runtime.load(require_token=False)
    spec = TrainSpec.from_dict(json.loads((args.run_dir / "spec.json").read_text()))
    from .native import run

    run(spec, runtime, args.run_dir)


if __name__ == "__main__":
    main()
