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
            ("prompts_per_step", 1, 256),
            ("max_steps", 1, 15),
            ("max_new_tokens", 1, 1024),
            ("max_prompt_tokens", 16, 1024),
            ("max_concurrent_rollouts", 1, 256),
            ("max_staleness", 0, 2),
            ("seed", 0, 2**31 - 1),
            ("timeout_seconds", 10, 21600),
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
        if self.algorithm not in ("grpo", "ppo"):
            raise ValueError(
                "implemented algorithms are grpo and ppo; DPO is not implemented"
            )
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
        if self.objective not in (
            "legacy_grpo",
            "group_pg_tis",
            "fork_ppo",
            "fork_grpo",
        ):
            raise ValueError("unsupported training objective")
        if self.execution not in ("areal", "standalone_sync", "fork_arm", "trl_sync"):
            raise ValueError("execution must be areal, standalone_sync or fork_arm")
        if self.score_centering and self.objective not in (
            "group_pg_tis",
            "fork_ppo",
            "fork_grpo",
        ):
            raise ValueError("score centering requires a supported policy objective")
        if self.target == "slurm":
            required = 4 if self.cache_mode == "map" else 3
            if self.execution in ("fork_arm", "trl_sync"):
                if self.max_gpus != required:
                    raise ValueError(f"this comparison profile needs {required} GPUs")
            elif self.max_gpus != 4:
                raise ValueError("the legacy Slurm comparison requires four GPUs")
        if self.execution in ("fork_arm", "trl_sync"):
            if (
                self.dataset != "gsm8k"
                or self.reward != "gsm8k"
                or self.environment != "math"
            ):
                raise ValueError(
                    "the fork comparison preset implements GSM8K with the original math reward"
                )
            expected = "fork_ppo" if self.algorithm == "ppo" else "fork_grpo"
            expected_lr = 1.7e-5 if self.algorithm == "ppo" else 6e-6
            if self.objective != expected or self.target != "slurm":
                raise ValueError(
                    "comparison execution requires its matching fork objective in Slurm"
                )
            if (
                self.max_steps != 15
                or self.prompts_per_step != 256
                or self.rollouts_per_prompt != 4
                or self.max_concurrent_rollouts != 256
                or self.max_staleness != (0 if self.execution == "trl_sync" else 2)
                or self.learning_rate != expected_lr
                or self.max_new_tokens != 1024
                or self.max_prompt_tokens != 1024
                or self.seed != 1
            ):
                raise ValueError(
                    "comparison preset is 15 updates, 256 prompts, four samples, original token caps and seed"
                )
        if self.execution == "trl_sync" and (
            self.score_centering or self.cache_mode != "recompute" or self.fit_mappers
        ):
            raise ValueError(
                "TRL baseline is synchronous, recompute, and score-centering off"
            )
        if self.algorithm == "ppo" and self.execution not in ("fork_arm", "trl_sync"):
            raise ValueError("ppo requires a comparison execution")
        if self.objective in ("fork_ppo", "fork_grpo") and self.execution not in (
            "fork_arm",
            "trl_sync",
        ):
            raise ValueError("fork objectives require their comparison execution")
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
