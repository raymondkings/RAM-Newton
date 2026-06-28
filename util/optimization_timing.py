from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


_JAX_SYNC_UNAVAILABLE = False


@dataclass(frozen=True)
class OptimizationTiming:
    """Wall-clock breakdown of a single optimize_morphology[_and_trajectory] call."""

    optimizer_seconds: float
    validation_seconds: float


class OptimizationTimer:
    """Wall-clock timer that separates cuRobo validation from optimizer work."""

    def __init__(
        self,
        device: torch.device | str | None = None,
        *,
        sync_jax: bool = False,
    ) -> None:
        self.device = torch.device(device) if device is not None else None
        self.sync_jax = sync_jax
        self._total_start: float | None = None
        self._total_elapsed: float | None = None
        self.validation_elapsed = 0.0

    def _sync(self) -> None:
        if torch.cuda.is_available():
            if self.device is not None and self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            else:
                torch.cuda.synchronize()
        if self.sync_jax:
            self._sync_jax()

    def _sync_jax(self) -> None:
        global _JAX_SYNC_UNAVAILABLE
        if _JAX_SYNC_UNAVAILABLE:
            return
        try:
            import jax
        except Exception:
            _JAX_SYNC_UNAVAILABLE = True
            return
        try:
            for device in jax.local_devices():
                sync = getattr(device, "synchronize_all_activity", None)
                if sync is not None:
                    sync()
        except Exception:
            return

    def start(self) -> None:
        self._sync()
        self._total_start = time.perf_counter()
        self._total_elapsed = None

    @contextmanager
    def validation(self) -> Iterator[None]:
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.validation_elapsed += time.perf_counter() - start

    def result(self) -> OptimizationTiming:
        if self._total_start is None:
            return OptimizationTiming(0.0, self.validation_elapsed)
        if self._total_elapsed is None:
            self._sync()
            self._total_elapsed = time.perf_counter() - self._total_start
        optimizer_elapsed = max(0.0, self._total_elapsed - self.validation_elapsed)
        return OptimizationTiming(optimizer_elapsed, self.validation_elapsed)
