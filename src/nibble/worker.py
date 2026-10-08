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
    import os

    os.environ["NIBBLE_OBJECTIVE"] = spec.objective
    os.environ["NIBBLE_SCORE_CENTERING"] = "1" if spec.score_centering else "0"
    os.environ["NIBBLE_MAPPER_GPU"] = (
        runtime.gpu_ids[3]
        if spec.target == "slurm" and len(runtime.gpu_ids) == 4
        else ""
    )
    if spec.execution == "fork_arm":
        os.environ["NIBBLE_PACKED_SDPA"] = "1"
        from .fork_arm import run
    elif spec.execution == "trl_sync":
        from .trl_sync import run
    elif spec.execution == "standalone_sync":
        from .sync_baseline import run
    else:
        from .native import run

    run(spec, runtime, args.run_dir)


if __name__ == "__main__":
    main()
