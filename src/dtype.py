import drjit as dr
import mitsuba as mi
import numpy as np
import torch


class DType:
    def __init__(self, dtype: str):
        if dtype not in ("half", "float"):
            raise ValueError(f"Unknown dtype: {dtype}")
        self.is_half = dtype == "half"

    @property
    def torch(self):
        return torch.float16 if self.is_half else torch.float32

    @property
    def numpy(self):
        return np.float16 if self.is_half else np.float32

    @property
    def np(self):
        return np.float16 if self.is_half else np.float32

    @property
    def mitsuba(self):
        return mi.Struct.Type.Float16 if self.is_half else mi.Struct.Type.Float32

    @property
    def mi(self):
        return mi.Struct.Type.Float16 if self.is_half else mi.Struct.Type.Float32

    @property
    def drjit(self):
        return dr.scalar.Float16 if self.is_half else dr.scalar.Float

    @property
    def dr(self):
        return dr.scalar.Float16 if self.is_half else dr.scalar.Float

    def __repr__(self):
        return f"DType({'half' if self.is_half else 'float'})"
