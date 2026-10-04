"""Runner configuration, trusted plugin registry, and conservative placement."""

from dataclasses import dataclass
from pathlib import Path
import json
import os
import subprocess


@dataclass
class Runtime:
    root: Path
    areal_root: Path
    cado_src: Path
    python: Path
    models: dict
    datasets: dict
    rewards: dict
    environments: dict
    gpu_ids: list[str]
    token: str

    @classmethod
    def load(cls, require_token=True):
        config = json.loads(Path(os.environ["NIBBLE_RUNNER_CONFIG"]).read_text())
        token = os.environ.get("NIBBLE_TOKEN", "")
        if require_token and len(token) < 24:
            raise ValueError("NIBBLE_TOKEN must contain at least 24 characters")
        return cls(
            root=Path(config["root"]).resolve(),
            areal_root=Path(config["areal_root"]).resolve(),
            cado_src=Path(config["cado_src"]).resolve(),
            python=Path(config["python"]).absolute(),
            models=config["models"],
            datasets=config["datasets"],
            rewards=config["rewards"],
            environments=config["environments"],
            gpu_ids=[str(x) for x in config["gpu_ids"]],
            token=token,
        )

    def plan(self, spec):
        for label, registry, key in [
            ("model", self.models, spec.model),
            ("dataset", self.datasets, spec.dataset),
            ("reward", self.rewards, spec.reward),
            ("environment", self.environments, spec.environment),
        ]:
            if key not in registry:
                raise ValueError(
                    f"unknown {label} {key!r}; runner administrator must register it"
                )
        if not Path(self.models[spec.model]).is_dir():
            raise ValueError("registered model snapshot is unavailable")
        required = 1 if spec.target == "login_pilot" else 2
        if required > len(self.gpu_ids) or required > spec.max_gpus:
            raise ValueError(
                f"profile needs {required} configured GPUs; cap/available devices are insufficient"
            )
        if spec.mapper_registry:
            registry = Path(spec.mapper_registry).resolve()
            if not registry.is_relative_to(self.root) or not registry.is_dir():
                raise ValueError(
                    "mapper_registry must be an existing directory inside Nibble's root"
                )
        return {
            "physical_gpus": required,
            "gpu_ids": self.gpu_ids[:required],
            "placement": "shared-one-GPU-feasibility"
            if required == 1
            else "separate-trainer-rollout",
            "rl_schedule": "synchronous"
            if spec.max_staleness == 0
            else "bounded-asynchronous",
            "api_execution": "background job",
            "weight_update": "disk",
            "cache_mode": spec.cache_mode,
            "note": "No speed or learning-quality claim follows from a tiny pilot.",
        }

    def environment(self, run_dir):
        env = os.environ.copy()
        env["PATH"] = str(self.python.parent) + os.pathsep + env.get("PATH", "")
        repo_src = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = os.pathsep.join(
            [repo_src, str(self.areal_root), str(self.cado_src)]
        )
        env["AREAL_FORCE_HOST_IP"] = "127.0.0.1"
        env["WANDB_MODE"] = "disabled"
        env["OMP_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        env["OPENBLAS_NUM_THREADS"] = "2"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["HF_HUB_OFFLINE"] = "1"
        env["HF_DATASETS_OFFLINE"] = "1"
        env["AREAL_CACHE_DIR"] = str(self.root / "work" / "cache" / "areal")
        env["NIBBLE_JOB_DIR"] = str(run_dir)
        env["PYTHONUNBUFFERED"] = "1"
        for key in ("NIBBLE_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
            env.pop(key, None)
        return env


def check_idle(gpu_ids):
    """Refuse occupied GPUs. This is a courtesy check, not a cluster reservation."""
    for gpu_id in gpu_ids:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                gpu_id,
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=True,
        )
        if result.stdout.strip():
            raise RuntimeError(
                f"GPU {gpu_id} has existing compute processes; job was not launched"
            )
