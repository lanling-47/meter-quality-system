"""Minimal runnable demo for UDM-Net using three synthetic device datasets."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.services.unified_degradation_data import (
    DegradationSeries,
    SameDeviceSeriesDataset,
    collate_degradation_series,
)
from app.services.unified_degradation_model import (
    UnifiedDegradationConfig,
    UnifiedDegradationNet,
    train_multidevice_epoch,
)


def make_series(device_type: str, response_dim: int, count: int, seed: int):
    rng = np.random.default_rng(seed)
    items = []
    for index in range(count):
        length = int(rng.integers(5, 9))
        times = np.linspace(0, 100, length, dtype=np.float32)
        stress_level = float(rng.uniform(0.4, 1.0))
        stresses = np.column_stack([
            np.full(length, stress_level),
            np.full(length, rng.uniform(0.3, 1.0)),
            times / 100.0,
        ]).astype(np.float32)
        baseline = rng.uniform(0.8, 1.2, size=response_dim)
        direction = np.ones(response_dim)
        if device_type == "capacitor":
            direction[0] = -1.0  # capacitance decreases while ESR increases
        degradation = stress_level * (times / 100.0)[:, None] * rng.uniform(0.08, 0.35, size=response_dim)
        noise = rng.normal(0.0, 0.01, size=(length, response_dim))
        responses = baseline + direction * degradation + noise
        target = float(np.clip(stress_level * 0.55 + np.max(degradation) * 1.4, 0.0, 1.0))
        items.append(DegradationSeries(
            device_type=device_type,
            sample_id=f"{device_type}-{index:03d}",
            times=times,
            stresses=stresses,
            responses=responses,
            target=target,
        ))
    return items


def main():
    torch.manual_seed(42)
    response_dims = {"capacitor": 2, "igbt": 2, "mosfet": 3}
    directions = {"capacitor": (-1, 1), "igbt": (1, 1), "mosfet": (1, 1, 1)}
    loaders = {}
    for index, (device_type, response_dim) in enumerate(response_dims.items()):
        dataset = SameDeviceSeriesDataset(make_series(device_type, response_dim, 24, seed=100 + index))
        loaders[device_type] = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=collate_degradation_series,
        )

    config = UnifiedDegradationConfig(
        response_dims=response_dims,
        response_directions=directions,
        stress_dim=3,
        d_model=32,
        nhead=4,
        num_layers=1,
    )
    model = UnifiedDegradationNet(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    for epoch in range(1, 6):
        metrics = train_multidevice_epoch(model, loaders, optimizer)
        print(f"epoch={epoch} total={metrics['total']:.4f} data={metrics['data']:.4f} "
              f"mono={metrics['monotonicity']:.4f} domain={metrics['domain']:.4f}")

    batch = next(iter(loaders["capacitor"]))
    model.eval()
    uncertainty = model.predict_with_uncertainty(
        "capacitor",
        batch["responses"],
        batch["stresses"],
        batch["times"],
        batch["mask"],
        mc_samples=8,
    )
    print("prediction:", uncertainty["prediction_mean"][:3].tolist())
    print("total uncertainty:", uncertainty["total_variance"][:3].tolist())


if __name__ == "__main__":
    main()
