from collections.abc import Callable

import drjit as dr


class DrTimer:
    def __init__(self, fn: callable, warmup: int = 1, step_kwargs: list[str] | Callable | None = None):
        self._fn = fn
        self._warmup = warmup
        self._step_kwargs = step_kwargs

    def _kwargs_for_step(self, i: int) -> dict:
        if self._step_kwargs is None:
            return {}
        if callable(self._step_kwargs):
            return self._step_kwargs(i)
        return {k: i for k in self._step_kwargs}

    def timeit(self, n: int) -> float:
        for w in range(self._warmup):
            self._fn(**self._kwargs_for_step(0))

        times = []
        for i in range(n):
            dr.kernel_history_clear()
            with dr.scoped_set_flag(dr.JitFlag.KernelHistory):
                self._fn(**self._kwargs_for_step(i))
            kh = dr.kernel_history()
            jit_ks = [k for k in kh if k["type"] is dr.KernelType.JIT]
            times.append(sum(k["execution_time"] for k in jit_ks))

        return sum(times) / n