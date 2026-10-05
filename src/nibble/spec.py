"""Strict public job contract. Limits deliberately keep this feasibility pilot small."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class TrainSpec:
    model: str
    dataset: str = "gsm8k"
    environment: str = "math"
    reward: str = "gsm8k"
    algorithm: str = "grpo"
    max_gpus: int = 1
    rollouts_per_prompt: int = 2
    prompts_per_step: int = 2
    max_steps: int = 1
    max_new_tokens: int = 64
    max_prompt_tokens: int = 256
    max_concurrent_rollouts: int = 2
    max_staleness: int = 0
    audit_on: bool = True
    seed: int = 1
    learning_rate: float = 6e-6
    timeout_seconds: int = 180
    target: str = "login_pilot"
    cache_mode: str = "recompute"
    mapper_registry: str | None = None
    score_centering: bool = False
    objective: str = "legacy_grpo"
    execution: str = "areal"
    fit_mappers: bool = False

    def __post_init__(self):
        for name, low, high in [
            ("max_gpus", 1, 8),
            ("rollouts_per_prompt", 2, 8),
            ("prompts_per_step", 1, 8),
            ("max_steps", 1, 15),
            ("max_new_tokens", 1, 1024),
            ("max_prompt_tokens", 16, 1024),
            ("max_concurrent_rollouts", 1, 64),
            ("max_staleness", 0, 2),
            ("seed", 0, 2**31 - 1),
            ("timeout_seconds", 10, 1800),
        ]:
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        for name in ("audit_on", "score_centering", "fit_mappers"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        for name in (
            "model",
            "dataset",
            "environment",
            "reward",
            "algorithm",
            "target",
            "cache_mode",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 256:
                raise ValueError(
                    f"{name} must be a non-empty string of at most 256 characters"
                )
        if self.mapper_registry is not None and not isinstance(
            self.mapper_registry, str
        ):
            raise ValueError("mapper_registry must be a string or null")
        if self.algorithm != "grpo":
            raise ValueError("this MVP implements GRPO only")
        if self.target not in ("login_pilot", "local", "slurm"):
            raise ValueError(
                "target must be login_pilot, local, or slurm (inside an allocation)"
            )
        if self.cache_mode not in ("recompute", "map"):
            raise ValueError("cache_mode must be recompute or map")
        if self.cache_mode == "map" and not (self.mapper_registry or self.fit_mappers):
            raise ValueError(
                "map mode requires a pre-fitted mapper_registry; no fitting is launched"
            )
        if self.cache_mode == "recompute" and self.mapper_registry is not None:
            raise ValueError("mapper_registry only applies to map mode")
        if self.objective not in ("legacy_grpo", "group_pg_tis"):
            raise ValueError("objective must be legacy_grpo or group_pg_tis")
        if self.execution not in ("areal", "standalone_sync"):
            raise ValueError("execution must be areal or standalone_sync")
        if self.score_centering and self.objective != "group_pg_tis":
            raise ValueError(
                "score centering requires the paper group_pg_tis objective"
            )
        if self.target == "slurm" and self.max_gpus != 4:
            raise ValueError("the Slurm comparison requires exactly four GPUs")
        if self.fit_mappers and (
            self.target != "slurm" or self.cache_mode != "map" or self.mapper_registry
        ):
            raise ValueError(
                "live fitting requires slurm map mode with a fresh automatic registry"
            )
        if self.execution == "standalone_sync" and (
            self.target != "slurm"
            or self.max_staleness
            or self.score_centering
            or self.cache_mode != "recompute"
        ):
            raise ValueError(
                "standalone_sync requires slurm, zero staleness, recompute, no SC"
            )
        if self.target != "slurm" and self.timeout_seconds > 240:
            raise ValueError("longer jobs require a Slurm allocation")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(self.learning_rate)
            or not 0 < self.learning_rate <= 1e-3
        ):
            raise ValueError("learning_rate must be finite and in (0, 0.001]")
        if self.target == "login_pilot":
            if (
                self.max_gpus != 1
                or self.score_centering
                or self.max_staleness != 0
                or self.cache_mode != "recompute"
            ):
                raise ValueError(
                    "login_pilot is one GPU, synchronous RL, exact cache recomputation"
                )
            if (
                self.max_steps > 2
                or self.max_new_tokens > 64
                or self.prompts_per_step > 2
            ):
                raise ValueError(
                    "login_pilot is limited to 2 steps, 2 prompts/step and 64 output tokens"
                )
        if self.target == "local" and self.max_gpus < 2:
            raise ValueError("local async profile needs at least 2 GPUs")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("job specification must be an object")
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
