"""Common data contract for meter and public accelerated-aging datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DegradationSeries:
    device_type: str
    sample_id: str
    times: np.ndarray
    stresses: np.ndarray
    responses: np.ndarray
    target: float

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=np.float32).reshape(-1)
        self.stresses = np.asarray(self.stresses, dtype=np.float32)
        self.responses = np.asarray(self.responses, dtype=np.float32)
        if self.stresses.ndim != 2 or self.responses.ndim != 2:
            raise ValueError("stresses and responses must be two-dimensional")
        if len(self.times) != len(self.stresses) or len(self.times) != len(self.responses):
            raise ValueError("times, stresses, and responses must share sequence length")
        if len(self.times) == 0:
            raise ValueError("series cannot be empty")
        order = np.argsort(self.times, kind="stable")
        self.times = self.times[order]
        self.stresses = self.stresses[order]
        self.responses = self.responses[order]


class SameDeviceSeriesDataset(Dataset):
    """Dataset for one device type; use one DataLoader per public dataset."""

    def __init__(self, series: Sequence[DegradationSeries]):
        self.series = list(series)
        if not self.series:
            raise ValueError("series cannot be empty")
        self.device_type = self.series[0].device_type
        response_dim = self.series[0].responses.shape[1]
        stress_dim = self.series[0].stresses.shape[1]
        for item in self.series:
            if item.device_type != self.device_type:
                raise ValueError("SameDeviceSeriesDataset accepts one device type")
            if item.responses.shape[1] != response_dim or item.stresses.shape[1] != stress_dim:
                raise ValueError("feature dimensions must be consistent within a device dataset")

    def __len__(self):
        return len(self.series)

    def __getitem__(self, index):
        return self.series[index]


def collate_degradation_series(items: Sequence[DegradationSeries]) -> dict[str, torch.Tensor | list[str]]:
    """Pad variable-length trajectories and create the model mask."""
    if not items:
        raise ValueError("cannot collate an empty batch")
    device_type = items[0].device_type
    if any(item.device_type != device_type for item in items):
        raise ValueError("a model batch must contain one device type")

    batch_size = len(items)
    max_length = max(len(item.times) for item in items)
    response_dim = items[0].responses.shape[1]
    stress_dim = items[0].stresses.shape[1]
    responses = torch.zeros(batch_size, max_length, response_dim, dtype=torch.float32)
    stresses = torch.zeros(batch_size, max_length, stress_dim, dtype=torch.float32)
    times = torch.zeros(batch_size, max_length, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    targets = torch.zeros(batch_size, dtype=torch.float32)
    sample_ids = []

    for row, item in enumerate(items):
        length = len(item.times)
        responses[row, :length] = torch.from_numpy(item.responses)
        stresses[row, :length] = torch.from_numpy(item.stresses)
        times[row, :length] = torch.from_numpy(item.times)
        if length < max_length:
            times[row, length:] = float(item.times[-1])
        mask[row, :length] = True
        targets[row] = float(item.target)
        sample_ids.append(item.sample_id)

    return {
        "device_type": device_type,
        "sample_ids": sample_ids,
        "responses": responses,
        "stresses": stresses,
        "times": times,
        "mask": mask,
        "target": targets,
    }
