"""Audit and summarize the Bosch GBH 2-24 MES / repair data.

The raw workbooks are intentionally supplied through command-line arguments or
environment variables. The committed outputs contain aggregate statistics only
and do not expose product serial numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import hashlib
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median

from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.services.bosch_observation_adapter import build_bosch_observation


PARAMS = [
    "WP2RunInFullVolt",
    "WP2RunInHalfCurrent",
    "WP2RunInHalfVolt",
    "WP2RunInFullCurrent",
    "WP3RunInCurrent",
    "WP3RunInVolt",
    "WP4ImpactEnergy",
    "WP1DummyCurrent10",
    "WP5ClutchTorq",
    "WP6NoloadCurrent",
    "WP6NoloadVolt",
    "WP7LeakageCurrent1",
    "WP7LeakageCurrent2",
    "WP7Similarity",
    "WP7Volt",
    "WP7Current",
]

# WP1 and the WP7 voltage/current channels require the Bosch test dictionary.
# They are retained for completeness but excluded from the usable physical
# range summary below.
DIAGNOSTIC_ONLY = {"WP1DummyCurrent10", "WP7Volt", "WP7Current", "WP7Similarity"}

SEMANTICS = {
    "WP2RunInFullVolt": ("WP2", "full-voltage run-in voltage", "V"),
    "WP2RunInHalfCurrent": ("WP2", "half-voltage run-in current", "A"),
    "WP2RunInHalfVolt": ("WP2", "half-voltage run-in voltage", "V"),
    "WP2RunInFullCurrent": ("WP2", "full-voltage run-in current", "A"),
    "WP3RunInCurrent": ("WP3", "post-run current", "A"),
    "WP3RunInVolt": ("WP3", "post-run test voltage", "V"),
    "WP4ImpactEnergy": ("WP4", "impact performance indicator; source unit m/ss", "m/ss"),
    "WP1DummyCurrent10": ("WP1", "dummy current channel; dictionary required", "mA0"),
    "WP5ClutchTorq": ("WP5", "clutch torque", "Nm"),
    "WP6NoloadCurrent": ("WP6", "no-load current", "A"),
    "WP6NoloadVolt": ("WP6", "no-load test voltage", "V"),
    "WP7LeakageCurrent1": ("WP7", "high-voltage leakage current channel 1", "mA"),
    "WP7LeakageCurrent2": ("WP7", "high-voltage leakage current channel 2", "mA"),
    "WP7Similarity": ("WP7", "similarity / reference matching indicator", "-"),
    "WP7Volt": ("WP7", "voltage channel; dictionary required", "V"),
    "WP7Current": ("WP7", "current channel; dictionary required", "A"),
}


def parse_mes_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    text = str(value).replace(" ASIA/SHANGHAI", "")
    # MES has nine fractional-second digits; six are sufficient for ordering.
    parts = text.split()
    if len(parts) < 3:
        return None
    date_text, time_text, meridiem = parts[0], parts[1], parts[2]
    if "." in time_text:
        whole, fraction = time_text.rsplit(".", 1)
        time_text = f"{whole}.{fraction[:6]}"
    try:
        return datetime.strptime(f"{date_text} {time_text} {meridiem}", "%d-%b-%y %I.%M.%S.%f %p")
    except ValueError:
        return None


def as_float(value: object) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def load_mes(path: Path) -> dict:
    workbook = load_workbook(path, read_only=True, data_only=True, keep_links=False)
    sheet = workbook.worksheets[0]
    serial_params: dict[str, set[str]] = defaultdict(set)
    serial_tools: dict[str, Counter[str]] = defaultdict(Counter)
    serial_dates: dict[str, list[datetime]] = defaultdict(list)
    parameter = {
        name: {
            "rows": 0,
            "states": Counter(),
            "units": Counter(),
            "out_of_range": 0,
            "numeric": 0,
            "zeros": 0,
            "values": [],
            "low": Counter(),
            "high": Counter(),
        }
        for name in PARAMS
    }
    total_rows = 0
    serials: set[str] = set()
    unique_test_ids: set[str] = set()
    events = {}
    invalid_timestamps = 0
    min_date: datetime | None = None
    max_date: datetime | None = None

    for row in sheet.iter_rows(min_row=2, max_col=23, values_only=True):
        if not any(value not in (None, "") for value in row):
            continue
        total_rows += 1
        serial = str(row[4]).strip() if row[4] not in (None, "") else ""
        parameter_name = str(row[7]).strip() if row[7] not in (None, "") else ""
        if serial:
            serials.add(serial)
            if parameter_name:
                serial_params[serial].add(parameter_name)
            if row[1] not in (None, ""):
                serial_tools[serial][str(row[1]).strip()] += 1
            timestamp = parse_mes_timestamp(row[6])
            if timestamp:
                serial_dates[serial].append(timestamp)
                min_date = timestamp if min_date is None else min(min_date, timestamp)
                max_date = timestamp if max_date is None else max(max_date, timestamp)
        if row[2] not in (None, ""):
            event_id = str(row[2]).strip()
            unique_test_ids.add(event_id)
            event = events.setdefault(event_id, {"serial": serial, "type": str(row[0]), "tool": str(row[1]), "date": parse_mes_timestamp(row[6]), "values": {}, "states": {}, "counts": Counter()})
            if event["serial"] != serial or event["type"] != str(row[0]):
                raise ValueError("test ID has inconsistent serial or type")
            event["values"][parameter_name] = as_float(row[10])
            event["states"][parameter_name] = str(row[12])
            event["counts"][parameter_name] += 1
            if event["date"] is None:
                invalid_timestamps += 1
        if parameter_name not in parameter:
            continue
        entry = parameter[parameter_name]
        entry["rows"] += 1
        entry["states"][str(row[12])] += 1
        if row[13] not in (None, ""):
            entry["units"][str(row[13])] += 1
        if row[16] not in (None, ""):
            entry["low"][str(row[16])] += 1
        if row[17] not in (None, ""):
            entry["high"][str(row[17])] += 1
        value = as_float(row[10])
        low = as_float(row[16])
        high = as_float(row[17])
        if value is not None:
            entry["numeric"] += 1
            if value == 0:
                entry["zeros"] += 1
            if len(entry["values"]) < 20000:
                entry["values"].append(value)
            if low is not None and high is not None and (value < low or value > high):
                entry["out_of_range"] += 1
    workbook.close()

    complete_vectors = sum(set(PARAMS).issubset(values) for values in serial_params.values())
    return {
        "rows": total_rows,
        "unique_test_ids": len(unique_test_ids),
        "complete_16_parameter_tests": sum(set(PARAMS).issubset(e["values"]) and all(e["counts"][p] == 1 for p in PARAMS) for e in events.values()),
        "events": events,
        "invalid_timestamp_rows": invalid_timestamps,
        "complete_16_parameter_serials": complete_vectors,
        "parameter": parameter,
        "serials": serials,
        "serial_tools": serial_tools,
        "serial_dates": serial_dates,
        "date_min": min_date.isoformat() if min_date else None,
        "date_max": max_date.isoformat() if max_date else None,
    }


def load_repairs(path: Path) -> dict:
    workbook = load_workbook(path, read_only=True, data_only=True, keep_links=False)
    serials: set[str] = set()
    rows = 0
    models = Counter()
    countries = Counter()
    parts = Counter()
    actions = Counter()
    serial_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    visible_sheets = [sheet for sheet in workbook.worksheets if sheet.sheet_state == "visible"]
    # Filtered views in the same workbook are not independent observations.
    master = workbook["所有生产日期"] if "所有生产日期" in workbook.sheetnames else visible_sheets[0]
    for sheet in [master]:
        if sheet.sheet_state != "visible":
            continue
        for row in sheet.iter_rows(min_row=2, max_col=13, values_only=True):
            # The workbook ends with an Overall Result footer row.
            if row[0] == "Overall Result" or row[4] in (None, ""):
                continue
            if not any(value not in (None, "") for value in row):
                continue
            rows += 1
            serial = str(row[4]).strip()
            serials.add(serial)
            model = str(row[1]).strip() if row[1] not in (None, "") else ""
            country = str(row[2]).strip() if row[2] not in (None, "") else ""
            part = str(row[6]).strip() if row[6] not in (None, "") else ""
            action = str(row[7]).strip() if row[7] not in (None, "") else ""
            models[model] += 1
            countries[country] += 1
            parts[part] += 1
            actions[action] += 1
            serial_rows[serial].append(
                {
                    "model": model,
                    "country": country,
                    "part": part,
                    "action": action,
                    "manufacturing_month": str(row[8]),
                    "arrival_month": str(row[10]),
                }
            )
    workbook.close()
    return {
        "rows": rows,
        "serials": serials,
        "models": models,
        "countries": countries,
        "parts": parts,
        "actions": actions,
        "serial_rows": serial_rows,
    }


def write_csv(path: Path, rows: list[list[object]], header: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def reconstruct_response(events: dict, output_dir: Path, linkage_counts: list[int] | None = None) -> dict:
    """Device-disjoint chronological same-cycle regression with frozen test."""
    import numpy as np

    first = {}
    for event in events.values():
        if event["date"] is None or not event["serial"]:
            continue
        sn = event["serial"]
        if sn not in first or event["date"] < first[sn]["date"]:
            first[sn] = event
    inputs = ["WP2RunInHalfVolt", "WP2RunInFullVolt", "WP2RunInHalfCurrent", "WP2RunInFullCurrent"]
    target = "WP3RunInCurrent"
    eligible = []
    masked_counts = Counter()
    for e in first.values():
        obs = build_bosch_observation(e["values"], e["states"])
        for name, field in obs["fields"].items():
            if field["physical_model_mask"]:
                masked_counts[name] += 1
        if all(obs["fields"][p]["physical_model_mask"] and e["values"][p] > 0 and e["counts"][p] == 1 for p in inputs + [target]):
            eligible.append(e)
    eligible.sort(key=lambda e: (e["date"], e["serial"]))
    # Entire calendar days are assigned to one split; serials appear only once.
    days = sorted({e["date"].date() for e in eligible})
    cut1 = max(1, int(len(days) * .6))
    cut2 = max(cut1 + 1, int(len(days) * .8))
    day_groups = [set(days[:cut1]), set(days[cut1:cut2]), set(days[cut2:])]
    groups = [[e for e in eligible if e["date"].date() in ds] for ds in day_groups]
    if not all(groups):
        raise ValueError("need at least three nonempty chronological day groups")
    serial_groups = [{e["serial"] for e in g} for g in groups]
    assert not any(serial_groups[i] & serial_groups[j] for i in range(3) for j in range(i+1, 3))
    type_vocab = sorted({e["type"] for e in groups[0]})

    def matrix(group, include_response):
        out = []
        for e in group:
            v = e["values"]
            row = [float(e["type"] == t) for t in type_vocab] + [v[p] for p in inputs[:2]]
            if include_response:
                row += [v[p] for p in inputs[2:]]
            out.append(row)
        return np.asarray(out, dtype=float)

    ys = [np.asarray([e["values"][target] for e in g]) for g in groups]
    predictions = {"WP2_full_current_baseline": np.asarray([e["values"]["WP2RunInFullCurrent"] for e in groups[2]])}
    selection = {}
    fitted_models = {}
    for name, include in [("ridge_type_voltage", False), ("ridge_type_voltage_current", True)]:
        xs = [matrix(g, include) for g in groups]
        mu = xs[0].mean(axis=0)
        scale = xs[0].std(axis=0)
        scale[scale < 1e-12] = 1
        xs = [np.column_stack([np.ones(len(x)), (x-mu)/scale]) for x in xs]
        candidates = []
        for alpha in [.01, .1, 1., 10., 100.]:
            penalty = np.eye(xs[0].shape[1]) * alpha
            penalty[0,0] = 0
            coef = np.linalg.solve(xs[0].T @ xs[0] + penalty, xs[0].T @ ys[0])
            val_mae = float(np.abs(xs[1] @ coef - ys[1]).mean())
            candidates.append((val_mae, alpha, coef))
        best = min(candidates, key=lambda c: (c[0], c[1]))
        predictions[name] = xs[2] @ best[2]
        selection[name] = {"alpha":best[1], "validation_mae_A":best[0]}
        fitted_models[name] = {"type_vocabulary":type_vocab, "numeric_inputs":inputs if include else inputs[:2], "train_mean":mu.tolist(), "train_scale":scale.tolist(), "intercept_and_scaled_coefficients":best[2].tolist(), "alpha":best[1]}
    (output_dir / "fitted_models.json").write_text(json.dumps(fitted_models, indent=2), encoding="utf-8")
    metrics = {name: {"mae_A":float(np.abs(pred-ys[2]).mean()), "rmse_A":float(np.sqrt(((pred-ys[2])**2).mean()))} for name, pred in predictions.items()}
    difference = np.abs(predictions["ridge_type_voltage_current"]-ys[2]) - np.abs(predictions["ridge_type_voltage"]-ys[2])
    rng = np.random.default_rng(20261007)
    boots = np.asarray([rng.choice(difference, len(difference), replace=True).mean() for _ in range(1000)])
    write_csv(output_dir / "response_metrics.csv", [[k,v["mae_A"],v["rmse_A"]] for k,v in metrics.items()], ["model", "test_mae_A", "test_rmse_A"])

    # Keep the held-out association auditable by product and operating strata
    # without exporting serial numbers or individual source rows.
    def metric_row(label, subset):
        indices = [i for i, e in enumerate(groups[2]) if subset(e)]
        if not indices:
            return None
        truth = ys[2][indices]
        row = [label, len(indices)]
        for name in ["WP2_full_current_baseline", "ridge_type_voltage", "ridge_type_voltage_current"]:
            pred = predictions[name][indices]
            row.extend([float(np.abs(pred-truth).mean()), float(np.sqrt(((pred-truth)**2).mean()))])
        return row

    strata = [("all", lambda e: True)]
    for tool in sorted({e.get("tool", "") for e in groups[2]}):
        strata.append((f"tool:{tool}", lambda e, tool=tool: e.get("tool", "") == tool))
    strata.extend([
        ("WP2_full_voltage:<=200V", lambda e: e["values"]["WP2RunInFullVolt"] <= 200),
        ("WP2_full_voltage:>200V", lambda e: e["values"]["WP2RunInFullVolt"] > 200),
        ("WP2_full_current:<=2.5A", lambda e: e["values"]["WP2RunInFullCurrent"] <= 2.5),
        ("WP2_full_current:>2.5A", lambda e: e["values"]["WP2RunInFullCurrent"] > 2.5),
    ])
    stratified_rows = [row for label, subset in strata if (row := metric_row(label, subset)) is not None]
    write_csv(output_dir / "response_stratified.csv", stratified_rows,
              ["stratum", "n", "baseline_mae_A", "baseline_rmse_A", "type_voltage_mae_A", "type_voltage_rmse_A", "full_mae_A", "full_rmse_A"])

    def correlation(x, y):
        if len(x) < 2 or float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    pair_rows = []
    pair_groups = [("all", groups[2])]
    pair_groups.extend((f"tool:{tool}", [e for e in groups[2] if e.get("tool", "") == tool])
                       for tool in sorted({e.get("tool", "") for e in groups[2]}))
    for label, subset in pair_groups:
        x = np.asarray([e["values"]["WP2RunInFullCurrent"] for e in subset], dtype=float)
        y = np.asarray([e["values"]["WP3RunInCurrent"] for e in subset], dtype=float)
        delta = y - x
        pair_rows.append([label, len(subset), correlation(x, y), float(delta.mean()), float(np.median(delta)), float(np.quantile(delta, .1)), float(np.quantile(delta, .9))])
    write_csv(output_dir / "response_pair_summary.csv", pair_rows,
              ["stratum", "n", "pearson_wp2full_wp3_A", "mean_wp3_minus_wp2_A", "median_wp3_minus_wp2_A", "p10_delta_A", "p90_delta_A"])
    dates_by_serial = defaultdict(list)
    for event in events.values():
        if event["date"] is not None and event["serial"]:
            dates_by_serial[event["serial"]].append(event["date"])
    repeated_dates = [dates for dates in dates_by_serial.values() if len(dates) > 1]
    repeated_spans = [(max(dates)-min(dates)).total_seconds()/3600 for dates in repeated_dates]
    result = {
        "task": "offline same-cycle WP3 current reconstruction from WP2; not lifetime prediction",
        "independent_eligible_devices":len(eligible),
        "excluded_first_cycle_devices":len(first)-len(eligible),
        "split_counts":dict(zip(["train","validation","test"], map(len,groups))),
        "split_day_ranges":{key:[str(min(ds)),str(max(ds))] for key,ds in zip(["train","validation","test"],day_groups)},
        "seed":20261007,
        "bootstrap_replicates":1000,
        "test_serial_overlap":0,
        "target":target,
        "response_inputs":inputs,
        "selection":selection,
        "metrics":metrics,
        "paired_mae_current_minus_type_voltage_A":float(difference.mean()),
        "paired_mae_difference_95ci_A":np.quantile(boots,[.025,.975]).tolist(),
        "retested_devices":len(repeated_dates),
        "same_day_retested_devices":sum(min(dates).date() == max(dates).date() for dates in repeated_dates),
        "retest_span_median_hours":median(repeated_spans) if repeated_spans else None,
        "retest_span_max_hours":max(repeated_spans) if repeated_spans else None,
        "adapter_physical_mask_counts":dict(masked_counts),
        "test_unseen_types":sorted({e["type"] for e in groups[2]}-set(type_vocab)),
    }
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9,3.4), layout="constrained")
        labels=["WP2 current", "Type + voltage", "Type + voltage\n+ WP2 currents"]
        axes[0].bar(labels,[v["mae_A"] for v in metrics.values()],color=["#777777","#397f94","#b75159"])
        axes[0].set_ylabel("Held-out device MAE (A)")
        axes[0].tick_params(axis="x",labelsize=8)
        if linkage_counts is not None:
            axes[1].bar(["MES devices","Repair serials","Matched serials"],linkage_counts,color=["#397f94","#777777","#b75159"])
            axes[1].set_yscale("log")
            axes[1].set_ylabel("Object count (log scale)")
        else:
            axes[1].set_visible(False)
        axes[1].tick_params(axis="x",labelsize=8)
        fig.savefig(output_dir / "response_reconstruction.pdf")
        fig.savefig(output_dir / "response_reconstruction.png",dpi=160)
        plt.close(fig)
        result["figure_generated"] = True

        fig, ax = plt.subplots(figsize=(4.8, 3.6), layout="constrained")
        sample = np.arange(len(groups[2]))
        if len(sample) > 3000:
            sample = rng.choice(sample, 3000, replace=False)
        x = np.asarray([e["values"]["WP2RunInFullCurrent"] for e in groups[2]])
        ax.scatter(x[sample], ys[2][sample], s=7, alpha=.25, color="#397f94", edgecolors="none")
        lo = float(min(x.min(), ys[2].min())); hi = float(max(x.max(), ys[2].max()))
        ax.plot([lo, hi], [lo, hi], "--", color="#777777", linewidth=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("WP2 full-voltage current (A)")
        ax.set_ylabel("WP3 post-run current (A)")
        ax.set_title("Held-out same-cycle response")
        fig.savefig(output_dir / "response_relationship.pdf")
        fig.savefig(output_dir / "response_relationship.png", dpi=160)
        plt.close(fig)
        result["relationship_figure_generated"] = True
    except ImportError as error:
        result["figure_generated"] = False
        result["figure_error"] = str(error)
    return result


def run(mes_path: Path, repair_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    mes = load_mes(mes_path)
    repairs = load_repairs(repair_path)
    overlap = mes["serials"] & repairs["serials"]
    tool_for_repair = Counter()
    for serial in overlap:
        if mes["serial_tools"].get(serial):
            tool_for_repair[mes["serial_tools"][serial].most_common(1)[0][0]] += 1

    parameter_rows = []
    for name in PARAMS:
        item = mes["parameter"][name]
        semantic = SEMANTICS[name]
        parameter_rows.append(
            [
                name,
                semantic[0],
                semantic[1],
                semantic[2],
                item["rows"],
                json.dumps(dict(item["states"]), ensure_ascii=False, sort_keys=True),
                json.dumps(dict(item["units"]), ensure_ascii=False, sort_keys=True),
                item["numeric"],
                item["numeric"] / item["rows"] if item["rows"] else None,
                item["zeros"],
                item["out_of_range"],
                name in DIAGNOSTIC_ONLY,
            ]
        )
    write_csv(
        output_dir / "parameter_summary.csv",
        parameter_rows,
        ["parameter", "stage", "inferred_meaning", "source_unit", "rows", "state_counts", "units", "numeric_rows", "numeric_rate", "zero_rows", "raw_range_violations", "diagnostic_only"],
    )
    write_csv(
        output_dir / "repair_part_counts.csv",
        [[part, count] for part, count in repairs["parts"].most_common()],
        ["repair_part", "records"],
    )
    write_csv(
        output_dir / "repair_model_mapping.csv",
        [[model, count] for model, count in tool_for_repair.most_common()],
        ["mes_tool_for_matched_serial", "matched_serials"],
    )

    usable_parameters = [name for name in PARAMS if name not in DIAGNOSTIC_ONLY and name != "WP4ImpactEnergy"]
    usable_out_of_range = sum(mes["parameter"][name]["out_of_range"] for name in usable_parameters)
    result = {
        "source": {
            "mes_file": mes_path.name,
            "repair_file": repair_path.name,
            "raw_files_committed": False,
            "mes_sha256": hashlib.sha256(mes_path.read_bytes()).hexdigest(),
            "repair_sha256": hashlib.sha256(repair_path.read_bytes()).hexdigest(),
            "origin": "Bosch industrial data, according to the provider; no independent provenance certification",
        },
        "mes": {
            "rows": mes["rows"],
            "serials": len(mes["serials"]),
            "unique_test_ids": mes["unique_test_ids"],
            "complete_16_parameter_serials": mes["complete_16_parameter_serials"],
            "complete_16_parameter_tests": mes["complete_16_parameter_tests"],
            "invalid_timestamp_rows": mes["invalid_timestamp_rows"],
            "date_min": mes["date_min"],
            "date_max": mes["date_max"],
            "usable_parameters": usable_parameters,
            "usable_raw_range_violations": usable_out_of_range,
            "diagnostic_only_parameters": sorted(DIAGNOSTIC_ONLY),
            "unit_unverified_proxy_parameters": ["WP4ImpactEnergy"],
            "tool_counts": dict(Counter(e["tool"] for e in mes["events"].values() if e.get("tool"))),
            "parameter_quality": {
                name: {
                    "rows": item["rows"],
                    "numeric_rate": round(item["numeric"] / item["rows"], 6) if item["rows"] else None,
                    "zero_rate": round(item["zeros"] / item["rows"], 6) if item["rows"] else None,
                    "state_counts": dict(item["states"]),
                    "range_violation_rate": round(item["out_of_range"] / item["rows"], 6) if item["rows"] else None,
                }
                for name, item in mes["parameter"].items()
            },
        },
        "repair": {
            "rows": repairs["rows"],
            "unique_serials": len(repairs["serials"]),
            "top_parts": repairs["parts"].most_common(12),
            "top_models": repairs["models"].most_common(8),
            "top_actions": repairs["actions"].most_common(8),
        },
        "linkage": {
            "matched_serials": len(overlap),
            "mes_serials_without_repair_record": len(mes["serials"] - repairs["serials"]),
            "repair_serials_without_mes_record": len(repairs["serials"] - mes["serials"]),
            "repair_to_mes_match_rate": round(len(overlap) / len(repairs["serials"]), 6) if repairs["serials"] else None,
            "matched_mes_tools": tool_for_repair.most_common(),
            "august_2025_repair_serials": len({sn for sn, items in repairs["serial_rows"].items() if any(r["manufacturing_month"] == "08.2025" for r in items)}),
            "august_2025_matched_serials": len({sn for sn, items in repairs["serial_rows"].items() if any(r["manufacturing_month"] == "08.2025" for r in items)} & mes["serials"]),
            "interpretation": "descriptive linkage; repair coverage and observation windows are not established as complete",
        },
        "quality_flags": {
            "wp1_and_wp7_placeholder_channels": sorted(DIAGNOSTIC_ONLY),
            "raw_range_checks_include_status_and_dictionary_mismatch": True,
            "serials_exposed_in_outputs": False,
        },
    }
    result["response_reconstruction"] = reconstruct_response(mes["events"], output_dir, [len(mes["serials"]), len(repairs["serials"]), len(overlap)])
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = render_report(result)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return result


def render_report(result: dict) -> str:
    mes = result["mes"]
    repair = result["repair"]
    linkage = result["linkage"]
    part_lines = "\n".join(f"| {part} | {count} |" for part, count in repair["top_parts"])
    return f"""# Bosch GBH 2-24 数据审计与维修关联实验

本报告由 `run_bosch_experiment.py` 生成。原始 Excel 不进入 Git；报告只保存聚合结果。

## 数据规模

- MES 明细：{mes['rows']:,} 行，{mes['serials']:,} 个序列号，{mes['unique_test_ids']:,} 个测试结果标识。
- 16 参数完整序列号：{mes['complete_16_parameter_serials']:,}。
- 16 参数完整独立测试：{mes['complete_16_parameter_tests']:,}。
- MES 时间范围：{mes['date_min']} 至 {mes['date_max']}。
- 维修明细：{repair['rows']:,} 行，{repair['unique_serials']:,} 个序列号。
- MES 与维修直接匹配：{linkage['matched_serials']:,} 个，维修序列号匹配率 {linkage['repair_to_mes_match_rate']:.2%}。

## 参数语义和质量边界

WP2 表示半电压/全电压磨合运行测试，WP3 表示磨合后的运行电流和电压，WP4/5 分别对应冲击性能和离合器释放扭矩，WP6 对应空载电流和电压，WP7 的两个泄漏电流通道对应高压绝缘测试。工位对比不是长期老化轨迹，运行电流差不能直接命名为损伤增量。

`WP1DummyCurrent10`、`WP7Volt`、`WP7Current` 和 `WP7Similarity` 被标记为诊断字段，正式物理建模前需要 Bosch 测试字典。WP4 原始单位 m/ss 与能量单位不一致，保留为冲击性能代理量。当前的上下限检查保留作审计线索，不能直接解释为失效标签。

## 维修部件分布

| 维修部件 | 记录数 |
|---|---:|
{part_lines}

## 结论

这组数据适合验证统一数据契约、对象专用观测适配、阶段化性能特征和 MES 到维修事件的追溯链路。当前结果属于描述性工业案例；维修记录覆盖、未维修设备的观察窗口、字段字典和身份映射仍需确认，不能由本报告单独推出长期寿命或真实退化速率。

## 跨阶段电流重建

仅取每个序列号最早一次测试，输入 WP2 电压与电流，重建同一测试中的 WP3 电流。测试导出时点不能证明阶段时长；这是一项离线跨通道响应重建，不是服役期未来预测。完整结果见 summary.json 和 response_metrics.csv；训练标准化仅用训练设备，超参数仅用验证设备。维修标签不进入此任务。

设备数与划分：{json.dumps(result['response_reconstruction']['split_counts'], ensure_ascii=False)}。
模型指标：{json.dumps(result['response_reconstruction']['metrics'], ensure_ascii=False)}。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mes", type=Path, default=os.getenv("BOSCH_GBH2_MES"))
    parser.add_argument("--repair", type=Path, default=os.getenv("BOSCH_GBH2_REPAIR"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    if not args.mes or not args.repair:
        parser.error("provide --mes and --repair, or set BOSCH_GBH2_MES and BOSCH_GBH2_REPAIR")
    result = run(Path(args.mes), Path(args.repair), args.output)
    print(json.dumps({"output": str(args.output), "mes_rows": result["mes"]["rows"], "matched_serials": result["linkage"]["matched_serials"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
