"""Nibble's CPU-only client; training dependencies stay on the GPU host."""

from .sdk import Client, Job
from .spec import TrainSpec

__all__ = ["Client", "Job", "TrainSpec"]
