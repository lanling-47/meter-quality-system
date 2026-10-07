"""Group-held-out study of a statistics/trajectory neural regressor against RF.

The NASA labels are relative within-run prefix positions, not measured damage.
Architecture is selected only on the development split; evaluation splits are
fixed before training and are not used to change model settings.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from torch import nn

from experiments.unified_degradation.experiment_runner import (
    _classical_vector, run_random_forest_baseline, split_groups,
)


DEV_SPLIT = 101
EVAL_SPLITS = (202, 303, 404)
MODEL_SEEDS = (17, 73)
DEVICE_NAMES = ("capacitor", "igbt", "mosfet")


class TrajectoryFusionNet(nn.Module):
    def __init__(self, feature_count: int, variant: str):
        super().__init__()
        self.variant = variant
        self.device_embedding = nn.Embedding(len(DEVICE_NAMES), 8)
        self.statistics = nn.Sequential(
            nn.Linear(feature_count + 8, 96), nn.LayerNorm(96), nn.GELU(),
            nn.Linear(96, 64), nn.GELU(),
        )
        if variant == "fusion":
            self.trajectory = nn.Sequential(
                nn.Conv1d(11, 32, 5, padding=2), nn.GELU(),
                nn.Conv1d(32, 48, 5, padding=2), nn.GELU(),
                nn.Conv1d(48, 48, 3, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            )
        self.head = nn.Sequential(
            nn.Linear(64 + (48 if variant == "fusion" else 0) + 8, 64),
            nn.GELU(), nn.Dropout(0.12), nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, features, sequence, device):
        embedding = self.device_embedding(device)
        statistical = self.statistics(torch.cat((features, embedding), dim=1))
        parts = [statistical]
        if self.variant == "fusion":
            parts.append(self.trajectory(sequence))
        parts.append(embedding)
        return torch.sigmoid(self.head(torch.cat(parts, dim=1)).squeeze(1))


def rows_for_split(datasets, split_seed):
    splits = {name: split_groups(datasets[name], split_seed) for name in DEVICE_NAMES}
    rows = {part: [] for part in ("train", "validation", "test")}
    for device, name in enumerate(DEVICE_NAMES):
        data = datasets[name]
        memberships = {group: part for part in rows
                       for group in getattr(splits[name], f"{part}_groups")}
        for sample, group in zip(data.series, data.groups):
            rows[memberships[group]].append({
                "feature": _classical_vector(sample, data.response_directions),
                "responses": sample.responses.copy(),
                "device": device, "name": name, "group": group,
                "sample_id": sample.sample_id, "target": float(sample.target),
            })
    return rows, splits


class Preprocessor:
    def fit(self, rows):
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = RobustScaler()
        self.scaler.fit(self.imputer.fit_transform(np.stack([r["feature"] for r in rows])))
        self.response_center = np.zeros((len(DEVICE_NAMES), 5), dtype=np.float32)
        self.response_scale = np.ones((len(DEVICE_NAMES), 5), dtype=np.float32)
        for device in range(len(DEVICE_NAMES)):
            response = np.concatenate([r["responses"] for r in rows if r["device"] == device])
            center = np.median(response, axis=0)
            scale = np.percentile(response, 75, axis=0) - np.percentile(response, 25, axis=0)
            width = response.shape[1]
            self.response_center[device, :width] = center
            self.response_scale[device, :width] = np.maximum(scale, 1e-5)
        return self

    def transform(self, rows):
        feature = self.scaler.transform(self.imputer.transform(
            np.stack([r["feature"] for r in rows])))
        feature = np.clip(np.nan_to_num(feature, nan=0, posinf=10, neginf=-10), -10, 10)
        sequences = []
        for row in rows:
            response = row["responses"]
            width = response.shape[1]
            normalized = np.clip((response - self.response_center[row["device"], :width]) /
                                 self.response_scale[row["device"], :width], -8, 8)
            sequence = np.zeros((response.shape[0], 11), dtype=np.float32)
            sequence[:, :width] = normalized
            sequence[:, 5:5 + width] = np.diff(normalized, axis=0, prepend=normalized[:1])
            sequence[:, 10] = np.linspace(0, 1, response.shape[0])
            sequences.append(sequence.T)
        return (torch.tensor(feature, dtype=torch.float32),
                torch.tensor(np.stack(sequences), dtype=torch.float32),
                torch.tensor([r["device"] for r in rows], dtype=torch.long),
                torch.tensor([r["target"] for r in rows], dtype=torch.float32))


def group_mae(rows, predictions):
    errors = defaultdict(list)
    devices = defaultdict(list)
    for row, value in zip(rows, predictions):
        errors[(row["name"], row["group"])].append(abs(float(value) - row["target"]))
    for (name, _), values in errors.items():
        devices[name].append(float(np.mean(values)))
    return {
        "group_macro_mae": float(np.mean([np.mean(x) for x in errors.values()])),
        "device_mae": {name: float(np.mean(devices[name])) for name in DEVICE_NAMES},
        "worst_group_mae": float(max(np.mean(x) for x in errors.values())),
        "per_group": {f"{name}/{group}": float(np.mean(values))
                      for (name, group), values in errors.items()},
    }


def fit_model(rows, variant, model_seed, epochs, evaluate=True):
    torch.manual_seed(model_seed)
    processor = Preprocessor().fit(rows["train"])
    train = processor.transform(rows["train"])
    valid = processor.transform(rows["validation"])
    model = TrajectoryFusionNet(train[0].shape[1], variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
    best = float("inf")
    best_state = None
    history = []
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for indices in torch.randperm(len(train[0])).split(48):
            prediction = model(*(x[indices] for x in train[:3]))
            loss = nn.functional.smooth_l1_loss(prediction, train[3][indices], beta=0.12)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_mae = (model(*train[:3]) - train[3]).abs().mean().item()
            valid_mae = (model(*valid[:3]) - valid[3]).abs().mean().item()
        history.append({"epoch": epoch, "train_mae": train_mae,
                        "validation_mae": valid_mae})
        if valid_mae < best - 1e-5:
            best, best_epoch, stale = valid_mae, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale == 35:
            break
    model.load_state_dict(best_state)
    model.eval()
    metrics = {"validation_mae": best, "best_epoch": best_epoch,
                    "stopped_epoch": epoch, "training_mae_at_best": history[best_epoch - 1]["train_mae"],
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "history": history}
    if evaluate:
        test = processor.transform(rows["test"])
        with torch.no_grad():
            predicted = model(*test[:3]).numpy()
        metrics.update(group_mae(rows["test"], predicted))
    return metrics


def rf_metrics(datasets, splits, rows, seed):
    _, predictions = run_random_forest_baseline(datasets, splits, seed)
    keyed = {(r["device_type"], r["sample_id"]): r["prediction"] for r in predictions}
    return group_mae(rows["test"],
                     [keyed[(r["name"], r["sample_id"])] for r in rows["test"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/public/nasa/processed"))
    parser.add_argument("--output", type=Path, default=Path(
        "experiments/unified_degradation/results/trajectory_fusion_study.json"))
    parser.add_argument("--epochs", type=int, default=220)
    args = parser.parse_args()
    datasets = {}
    for name in DEVICE_NAMES:
        with (args.data_root / f"{name}.pkl").open("rb") as handle:
            datasets[name] = pickle.load(handle)
    development_rows, _ = rows_for_split(datasets, DEV_SPLIT)
    development = {variant: [fit_model(development_rows, variant, seed, args.epochs, evaluate=False)
                             for seed in MODEL_SEEDS] for variant in ("statistics", "fusion")}
    selected = min(development, key=lambda variant: np.mean(
        [r["validation_mae"] for r in development[variant]]))
    evaluation = {}
    for split_seed in EVAL_SPLITS:
        rows, splits = rows_for_split(datasets, split_seed)
        neural = [fit_model(rows, selected, seed, args.epochs) for seed in MODEL_SEEDS]
        forest = [rf_metrics(datasets, splits, rows, seed) for seed in MODEL_SEEDS]
        evaluation[str(split_seed)] = {"neural": neural, "random_forest": forest,
                                       "test_group_count": len(set((r["name"], r["group"])
                                                                   for r in rows["test"]))}
        print(f"split {split_seed}: neural {[round(x['group_macro_mae'], 4) for x in neural]}, "
              f"RF {[round(x['group_macro_mae'], 4) for x in forest]}", flush=True)
    output = {"label": "relative within-run prefix stage, not RUL",
              "protocol": {"development_split": DEV_SPLIT, "evaluation_splits": EVAL_SPLITS,
                           "model_seeds": MODEL_SEEDS, "selection": "lowest mean validation MAE",
                           "no_test_based_tuning": True},
              "development": development, "selected_variant": selected,
              "evaluation": evaluation}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("selected:", selected, "output:", args.output, flush=True)


if __name__ == "__main__":
    main()
