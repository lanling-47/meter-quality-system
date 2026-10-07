"""Compare a compact device-gated neural regressor with the frozen RF protocol.

This is an exploratory model. Test groups are never used for scaling, early
stopping, or architecture selection. Run from the meter-quality-system root.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from torch import nn

from experiments.unified_degradation.experiment_runner import (
    _classical_vector, run_random_forest_baseline, split_groups,
)


class GatedResidualNet(nn.Module):
    """Shared trend, device residuals and an input-dependent shrinkage gate."""

    def __init__(self, features: int, devices: int, width: int = 24):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(features, width), nn.GELU(), nn.Linear(width, 1))
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(features, width // 2), nn.GELU(), nn.Linear(width // 2, 1))
            for _ in range(devices)
        ])
        self.gates = nn.ModuleList([nn.Linear(features, 1) for _ in range(devices)])

    def forward(self, x: torch.Tensor, device: torch.Tensor) -> torch.Tensor:
        shared = self.shared(x).squeeze(-1)
        residual = torch.zeros_like(shared)
        for index, (expert, gate) in enumerate(zip(self.experts, self.gates)):
            selected = device == index
            if selected.any():
                # Small initial gate limits overfitting when a device has few groups.
                residual[selected] = (torch.sigmoid(gate(x[selected]).squeeze(-1) - 2.0)
                                      * expert(x[selected]).squeeze(-1))
        return torch.sigmoid(shared + residual)


def make_arrays(datasets, splits):
    names = sorted(datasets)
    partitions = {part: [] for part in ("train", "validation", "test")}
    for device_index, name in enumerate(names):
        dataset = datasets[name]
        split = splits[name]
        for series, group in zip(dataset.series, dataset.groups):
            for part in partitions:
                if group in getattr(split, f"{part}_groups"):
                    partitions[part].append((
                        _classical_vector(series, dataset.response_directions),
                        device_index, float(series.target), name, series.sample_id, group,
                    ))
                    break
    return partitions


def fit_seed(partitions, seed: int, epochs: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = partitions["train"]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = RobustScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(np.stack([r[0] for r in train])))
    def tensors(rows):
        x = scaler.transform(imputer.transform(np.stack([r[0] for r in rows])))
        x = np.clip(np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0), -10, 10)
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor([r[1] for r in rows], dtype=torch.long),
                torch.tensor([r[2] for r in rows], dtype=torch.float32))
    x_train = np.clip(np.nan_to_num(x_train), -10, 10)
    train_t = (torch.tensor(x_train, dtype=torch.float32),
               torch.tensor([r[1] for r in train], dtype=torch.long),
               torch.tensor([r[2] for r in train], dtype=torch.float32))
    valid_t = tensors(partitions["validation"])
    test_t = tensors(partitions["test"])
    model = GatedResidualNet(train_t[0].shape[1], max(r[1] for r in train) + 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    best_mae, best_state, stale, best_epoch = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(train))
        for batch in permutation.split(32):
            optimizer.zero_grad()
            prediction = model(train_t[0][batch], train_t[1][batch])
            loss = nn.functional.smooth_l1_loss(prediction, train_t[2][batch], beta=0.1)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_mae = (model(train_t[0], train_t[1]) - train_t[2]).abs().mean().item()
            mae = (model(valid_t[0], valid_t[1]) - valid_t[2]).abs().mean().item()
        history.append({"epoch": epoch, "train_mae": train_mae, "validation_mae": mae})
        if mae < best_mae - 1e-5:
            best_mae, best_epoch, stale = mae, epoch, 0
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 25:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_train_mae = (model(train_t[0], train_t[1]) - train_t[2]).abs().mean().item()
        predicted = model(test_t[0], test_t[1]).numpy()
    rows = [{"device_type": row[3], "sample_id": row[4], "group": row[5],
             "target": row[2], "prediction": float(value)}
            for row, value in zip(partitions["test"], predicted)]
    return {"seed": seed, "validation_mae": best_mae, "best_epoch": best_epoch,
            "best_train_mae": best_train_mae, "stopped_epoch": history[-1]["epoch"],
            "history": history,
            "test_mae": float(np.mean(np.abs(predicted - test_t[2].numpy()))),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "predictions": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/public/nasa/processed"))
    parser.add_argument("--output", type=Path, default=Path("experiments/unified_degradation/results/gated_residual_exploratory.json"))
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    datasets = {}
    for name in ("capacitor", "igbt", "mosfet"):
        with (args.data_root / f"{name}.pkl").open("rb") as handle:
            datasets[name] = pickle.load(handle)
    splits = {name: split_groups(dataset, 42) for name, dataset in datasets.items()}
    partitions = make_arrays(datasets, splits)
    results = [fit_seed(partitions, seed, args.epochs) for seed in (17, 42, 73)]
    rf = {str(seed): run_random_forest_baseline(datasets, splits, seed)[0]["overall"]["mae"]
          for seed in (17, 42, 73)}
    output = {"status": "exploratory; single fixed group split", "split_seed": 42,
              "model": "GatedResidualNet", "neural": results, "random_forest_mae": rf}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("neural MAE:", [round(r["test_mae"], 5) for r in results])
    print("RF MAE:", rf)


if __name__ == "__main__":
    main()
