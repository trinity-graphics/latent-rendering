import dataclasses
import json

import numpy as np

from pipeline.outputs import Image, Latent, OutputMeta, Scalar

METRICS = {"mse", "mse_lin", "psnr", "ssim", "mean", "var"}
METRICS_RGB = {"lpips", "flip"}


def _expected_metrics(meta: dict) -> set[str]:
    return METRICS | (METRICS_RGB if meta.dtype is Image else set())


class _MetricsEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return super().default(obj)


###################################################################################
#   LoggerMetrics: A strict dictionary that requires pre-defined LoggerEntries.   #
###################################################################################
# TODO add support for per-channel metrics logging.
class LoggerMetrics:
    def __init__(
        self,
        *output_classes: type,
        num_refiner_layers: int = 0,
        exclude: set[str] = None,
    ):
        self._schema: dict[str, OutputMeta] = self._build_schema(
            *output_classes, num_refiner_layers=num_refiner_layers, exclude=exclude
        )
        self._data: dict[str, float | dict | None] = {}
        self.logged_entries: set[str] = set()
        self.reset()

    #
    #   Schema construction
    #
    @staticmethod
    def _build_schema(
        *output_classes: type,
        num_refiner_layers: int = 0,
        exclude: set[str] = None,
    ) -> dict[str, OutputMeta]:
        exclude = exclude or set()
        schema = {}
        for cls in output_classes:
            for f in dataclasses.fields(cls):
                if f.name in exclude:
                    continue
                meta: OutputMeta = f.metadata.get("meta")
                if meta is None:
                    continue
                if meta.key_template is not None:
                    for i in range(num_refiner_layers):
                        key = meta.key_template.format(i=i)
                        schema[key] = dataclasses.replace(
                            meta,
                            pretty_name=meta.pretty_name.format(i=i),
                            description=meta.description.format(i=i),
                        )
                else:
                    schema[f.name] = meta

            # Expand implicit decoded-latent entries after each class
            for key, meta in list(schema.items()):
                if meta.dtype is Latent and meta.log_decoded:
                    decoded_key = f"decoded_{key}"
                    if decoded_key not in schema:
                        schema[decoded_key] = dataclasses.replace(
                            meta,
                            dtype=Image,
                            pretty_name=f"Decoded {meta.pretty_name}",
                            description=f"{meta.description} Decoded to RGB.",
                            log_decoded=False,
                            key_template=None,
                            has_ground_truth=True,
                        )

        return schema

    #
    #   Data access
    #
    def all_keys(self) -> list[str]:
        return list(self._schema.keys())

    def reset(self) -> None:
        self._data = {k: None for k in self._schema}
        self.logged_entries = set()

    def __getitem__(self, key: str) -> dict:
        if key not in self._schema:
            raise KeyError(f"No schema entry for '{key}'")
        return self._schema[key], self._data.get(key)

    def __setitem__(self, key: str, value: float | dict) -> None:
        meta = self._schema[key]
        if key in self.logged_entries:
            raise RuntimeError(
                f"Already logged to '{key}' this step. Call reset() before logging again."
            )
        if meta.dtype is Scalar and not isinstance(value, (int, float)):
            raise TypeError(f"'{key}' is a scalar entry, expected float or int.")
        if meta.dtype is Image or meta.dtype is Latent:
            if not isinstance(value, dict):
                raise TypeError(f"'{key}' is an image entry, expected a metrics dict.")
            expected = _expected_metrics(meta)
            if value.keys() != expected:
                missing = expected - value.keys()
                unexpected = value.keys() - expected
                raise ValueError(
                    f"Metrics mismatch for '{key}'. Missing: {missing or 'none'}. Unexpected: {unexpected or 'none'}."
                )
        self._data[key] = value
        self.logged_entries.add(key)
