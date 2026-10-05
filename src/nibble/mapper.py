"""Fit adjacent-policy KV maps on the reserved GPU and charge the barrier cost.

The original fitter polled saved checkpoints after an update. For this bounded
pilot, we wait at the trainer's pre-resume hook so artifacts exist before decode
continues. This overhead is included in training wall time, never subtracted.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from .report import event


def fitting_trainer(base, spec, runtime, run_dir):
    """Return a trainer that fits v(i)->v(i+1) before rollout resume."""

    class FittingTrainer(base):
        def _wait_for_mapper_artifact(self, *, new_version):
            matches = list(
                (run_dir / "areal/checkpoints").rglob(
                    f"*globalstep{new_version - 1}/config.json"
                )
            )
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one saved policy v{new_version}, found {len(matches)}"
                )
            target = matches[0].parent
            if new_version == 1:
                source = runtime.models[spec.model]
            else:
                previous = list(
                    (run_dir / "areal/checkpoints").rglob(
                        f"*globalstep{new_version - 2}/config.json"
                    )
                )
                if len(previous) != 1:
                    raise RuntimeError(
                        "previous policy checkpoint missing for mapper fit"
                    )
                source = str(previous[0].parent)
            registry = run_dir / "registry"
            final = registry / f"v{new_version - 1}-v{new_version}"
            staging = registry / f".pending-v{new_version - 1}-v{new_version}"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = os.environ["NIBBLE_MAPPER_GPU"]
            start = time.perf_counter()
            command = [
                str(runtime.python),
                str(
                    Path(
                        os.environ.get(
                            "NIBBLE_CODE_ROOT", os.environ["NIBBLE_PROJECT_ROOT"]
                        )
                    )
                    / "tools/fit_mapper_hf.py"
                ),
                "--source",
                str(source),
                "--target",
                str(target),
                "--prefixes",
                os.environ["NIBBLE_CALIBRATION"],
                "--limit",
                "200",
                "--max-tokens",
                "2048",
                "--batch-size",
                "8",
                "--lambda",
                "0.01",
                "--stride",
                "4",
                "--selection",
                "same_layer",
                "--out",
                str(staging),
                "--provenance",
                json.dumps(
                    {"source_version": new_version - 1, "target_version": new_version}
                ),
            ]
            with (run_dir / f"mapper-v{new_version}.log").open("w") as log:
                subprocess.run(
                    command,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=180,
                )
            staging.rename(final)
            event(
                run_dir,
                "mapper_ready",
                source_version=new_version - 1,
                target_version=new_version,
                seconds=time.perf_counter() - start,
            )

    return FittingTrainer
