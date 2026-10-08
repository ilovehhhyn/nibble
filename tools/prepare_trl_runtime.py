#!/usr/bin/env python3
"""Create and import-check the pinned, isolated HF TRL runtime on Della."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import venv

TRL_VERSION = "0.28.0"
ROOT = Path("/scratch/gpfs/ARORA/hh9077/nibble")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--base-python",
        type=Path,
        default=Path(
            os.environ.get(
                "NIBBLE_PYTHON",
                "/scratch/gpfs/ARORA/hh9077/areal-kv/AReaL-arm/.venv/bin/python",
            )
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    base_python = args.base_python.absolute()
    env_dir = root / "work" / "venvs" / f"trl-{TRL_VERSION}"
    env_python = env_dir / "bin" / "python"
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    if not env_python.exists():
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(env_dir)
    shared_sites = json.loads(
        subprocess.check_output(
            [
                str(base_python),
                "-c",
                "import json,site; print(json.dumps(site.getsitepackages()))",
            ],
            text=True,
        )
    )
    env_sites = next(env_dir.glob("lib/python*/site-packages"))
    (env_sites / "nibble_base_runtime.pth").write_text(
        "".join(f"{p}\n" for p in shared_sites)
    )
    subprocess.run(
        [
            str(env_python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-deps",
            f"trl=={TRL_VERSION}",
        ],
        check=True,
    )
    check = r"""
import sys, torch, transformers, trl
import trl.import_utils as iu
iu.is_weave_available = lambda: False
from trl.experimental.ppo import PPOConfig, PPOTrainer
from trl.trainer import GRPOConfig, GRPOTrainer
assert trl.__version__ == "0.28.0", trl.__version__
assert transformers.__version__ == "5.3.0", transformers.__version__
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"trl={trl.__version__}")
print("imports=PPOConfig,PPOTrainer,GRPOConfig,GRPOTrainer")
"""
    env = os.environ.copy()
    env["TRL_EXPERIMENTAL_SILENCE"] = "1"
    subprocess.run([str(env_python), "-c", check], check=True, env=env)
    print(f"isolated_runtime={env_python}")
    print("base environment packages were not modified")


if __name__ == "__main__":
    main()
