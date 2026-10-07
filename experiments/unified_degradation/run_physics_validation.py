"""Validate the interpretable physics layer against cached NASA datasets."""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

from app.services.physics_degradation import physics_consistency_metrics


def main() -> None:
    project = Path(__file__).resolve().parents[2]
    data_root = (project / "data" / "public" / "nasa").resolve()
    output = (project / "experiments" / "unified_degradation" / "results" / "nasa_real").resolve()
    if data_root.drive.upper() != "D:" or output.drive.upper() != "D:":
        raise ValueError("physics validation must run entirely on D:")
    temp = data_root.parent / "tmp"
    for name in ("TEMP", "TMP", "MPLCONFIGDIR"):
        os.environ[name] = str(temp / name.lower())
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    processed = data_root / "processed"
    results = {}
    for device_type in ("capacitor", "igbt", "mosfet"):
        with (processed / f"{device_type}.pkl").open("rb") as handle:
            dataset = pickle.load(handle)
        results[device_type] = physics_consistency_metrics(dataset)
    (output / "physics_validation.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8"
    )
    lines = [
        "# 物理退化层验证",
        "",
        "物理层预测的是归一化损伤阶段 D，不把电容、IGBT 和 MOSFET 的物理量强行视为同一变量。",
        "应力历史先通过器件专用方程得到 D，再与观测响应构造的健康代理进行一致性检查。",
        "",
        "| 器件 | 曲线组数 | 物理代理 MAE | Spearman | 物理阶段单调性违反率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for device_type, metrics in results.items():
        lines.append(
            f"| {device_type} | {metrics['groups']} | {metrics['mean_proxy_physics_mae']:.4f} | "
            f"{metrics['mean_spearman']:.4f} | {metrics['mean_monotonicity_violation']:.4f} |"
        )
    lines.extend([
        "",
        "## 方程",
        "",
        "- 电容：r = A exp(-Ea/(kT)) (V/Vrated)^n。",
        "- IGBT/MOSFET：r = A exp(-Ea/(kTj)) (DeltaT)^m (S/S0)^n。",
        "- 损伤阶段：D(t) = clip(cumsum(r(t) Delta t) / cumsum(r(t) Delta t)_final, 0, 1)。",
        "",
        "这些参数是公开数据上的先验起点，不等同于具体厂家材料参数；正式论文应通过器件批次数据或数据表重新标定。",
    ])
    (output / "physics_validation.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: {m: v for m, v in value.items() if m != "rows"} for k, value in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
