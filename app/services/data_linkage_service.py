"""Data linkage between lifecycle quality rows and production trace rows.

This module keeps linkage and feature imputation outside the raw source tables.
Rows generated here are auditable: every mapped meter carries a source label and
confidence so demo/synthetic links can be excluded from formal experiments.
"""
import json
import re
from datetime import datetime
from typing import Iterable

import numpy as np
from sqlalchemy.orm import Session

from app.models.db_models import (
    BatchQualityProfile,
    FactoryTest,
    MeterIdentity,
    MeterQualityTrace,
    ProductionProcessProfile,
    QualityChain,
    QualityFeatureSnapshot,
)
from app.services.production_etl_service import (
    PRODUCTION_FEATURE_NAMES,
    get_production_features_for_barcode,
)
from app.services.reliability_sample_service import get_reliability_feature_matrix


SNAPSHOT_FEATURE_NAMES = [
    "crystal_ppm",
    "ct_ratio_err",
    "clock_accuracy_sd",
    "factory_fault",
    "initial_error_max",
    "final_error_max",
    "error_degradation_rate",
    "clock_drift_rate",
    "clock_daily_error_max",
    "error_variability_max",
    "error_consistency_max",
    "energy_balance_ratio",
    "energy_peak_ratio",
    "rs485_total_errors",
    "prod_ccd_retry",
    "prod_func_retry",
    "prod_reverify_retry",
    "prod_total_retry",
    "prod_smt_aoi_defect",
    "prod_wave_aoi_defect",
    "prod_repaired",
    "prod_repair_count",
    "prod_leak_current_max",
    "prod_quality_risk_score",
]

PROCESS_PROFILE_FEATURE_NAMES = [
    "process_smt_pressure_mm",
    "process_smt_precision_cpk",
    "process_reflow_temp_span_c",
    "process_reflow_conveyor_speed_cm_min",
    "process_online_current_meter_ua",
    "process_online_current_deviation_ua",
    "process_online_current_max_ua",
    "process_wave_preheat_span_c",
    "process_wave_solder_temp_c",
    "process_wave_peak_height_delta_mm",
]


QUALITY_DEFAULTS = {
    "crystal_ppm": 9.0,
    "ct_ratio_err": 0.5,
    "clock_accuracy_sd": 0.018,
    "factory_fault": 0.0,
    "initial_error_max": 0.0,
    "final_error_max": 0.0,
    "error_degradation_rate": 0.0,
    "clock_drift_rate": 0.0,
    "clock_daily_error_max": 0.22,
    "error_variability_max": 0.006,
    "error_consistency_max": 0.086,
    "energy_balance_ratio": 1.0,
    "energy_peak_ratio": 0.246,
    "rs485_total_errors": 0.0,
}


def normalize_barcode(value: str | None) -> str:
    """Normalize barcode variants such as M..., ATXM..., TXM..., CS_TXM....

    The current demo data stores the same physical meter as both
    ``M202604280001`` and ``ATXM202604280001``.  For matching, the stable part is
    the long numeric serial segment.
    """
    if value is None:
        return ""
    text = re.sub(r"[^A-Za-z0-9]", "", str(value).strip().upper())
    digit_runs = re.findall(r"\d{6,}", text)
    if digit_runs:
        return max(digit_runs, key=len)
    for prefix in ("CSTXM", "ATXM", "TXM", "CS", "M"):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def _to_float(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bool_float(value) -> float:
    return 1.0 if bool(value) else 0.0


def _json_load(value, default):
    if not value:
        return default
    try:
        parsed = json.loads(value)
        return parsed if parsed is not None else default
    except Exception:
        return default


def _process_profile_index(
    profiles: Iterable[ProductionProcessProfile],
) -> dict[str, ProductionProcessProfile]:
    index: dict[str, ProductionProcessProfile] = {}
    for profile in profiles:
        normalized_ids = _json_load(profile.normalized_meter_ids_json, [])
        if not normalized_ids:
            normalized_ids = [normalize_barcode(v) for v in _json_load(profile.meter_ids_json, [])]
        for normalized in normalized_ids:
            if normalized and normalized not in index:
                index[str(normalized)] = profile
    return index


def _process_profile_for_barcode(
    normalized_barcode: str | None,
    profiles_by_norm: dict[str, ProductionProcessProfile],
    profiles: list[ProductionProcessProfile],
) -> ProductionProcessProfile | None:
    if normalized_barcode:
        profile = profiles_by_norm.get(normalized_barcode)
        if profile is not None:
            return profile
    # 当前示范数据只有一个批次过程画像时，将其视为批次级工艺证据。
    return profiles[0] if len(profiles) == 1 else None


def _process_profile_features(
    profile: ProductionProcessProfile | None,
    normalized_barcode: str | None,
) -> list[float]:
    if profile is None:
        return [0.0 for _ in PROCESS_PROFILE_FEATURE_NAMES]

    current_map = _json_load(profile.online_current_by_meter_json, {})
    current_by_norm = {
        normalize_barcode(key): _to_float(value, 0.0)
        for key, value in current_map.items()
    } if isinstance(current_map, dict) else {}
    meter_current = current_by_norm.get(normalized_barcode or "")
    avg_current = _to_float(profile.online_current_avg_ua, 0.0)
    if meter_current is None:
        meter_current = avg_current
        current_deviation = 0.0
    else:
        current_deviation = abs(meter_current - avg_current)

    preheat_min = _to_float(profile.wave_preheat_temp_min_c, 0.0)
    preheat_max = _to_float(profile.wave_preheat_temp_max_c, 0.0)
    peak_1 = _to_float(profile.wave_peak_height_1_mm, 0.0)
    peak_2 = _to_float(profile.wave_peak_height_2_mm, 0.0)

    return [
        _to_float(profile.smt_pressure_mm, 0.0),
        _to_float(profile.smt_precision_cpk, 0.0),
        _to_float(profile.reflow_temp_span_c, 0.0),
        _to_float(profile.reflow_conveyor_speed_cm_min, 0.0),
        _to_float(meter_current, 0.0),
        _to_float(current_deviation, 0.0),
        _to_float(profile.online_current_max_ua, 0.0),
        max(preheat_max - preheat_min, 0.0),
        _to_float(profile.wave_solder_temp_c, 0.0),
        abs(peak_2 - peak_1) if peak_1 or peak_2 else 0.0,
    ]


def get_process_profile_features_for_barcode(barcode: str | None, db: Session) -> dict[str, float]:
    profiles = db.query(ProductionProcessProfile).all()
    profiles_by_norm = _process_profile_index(profiles)
    normalized = normalize_barcode(barcode)
    profile = _process_profile_for_barcode(normalized, profiles_by_norm, profiles)
    return dict(zip(PROCESS_PROFILE_FEATURE_NAMES, _process_profile_features(profile, normalized)))


def _upsert_identity(
    db: Session,
    *,
    meter_no: int | None,
    barcode: str,
    order_no: str | None,
    source: str,
    confidence: float,
    note: str,
) -> tuple[MeterIdentity, bool]:
    normalized = normalize_barcode(barcode)
    existing = None
    if meter_no is not None:
        existing = db.query(MeterIdentity).filter_by(meter_no=meter_no).first()
    if existing is None:
        existing = db.query(MeterIdentity).filter_by(barcode=barcode).first()
    if existing is None:
        existing = MeterIdentity(
            meter_no=meter_no,
            barcode=barcode,
            normalized_barcode=normalized,
            order_no=order_no,
            source=source,
            confidence=confidence,
            note=note,
        )
        db.add(existing)
        return existing, True

    # Preserve higher-confidence/manual links, but allow a demo meter_no bridge to
    # complete an otherwise barcode-only row.  The row source then stays explicit
    # as synthetic_demo so formal experiments can filter it out.
    should_refresh = confidence >= _to_float(existing.confidence, 0.0)
    should_attach_meter_no = existing.meter_no is None and meter_no is not None
    if should_refresh or should_attach_meter_no:
        existing.meter_no = meter_no if meter_no is not None else existing.meter_no
        existing.barcode = barcode
        existing.normalized_barcode = normalized
        existing.order_no = order_no or existing.order_no
        existing.source = source
        existing.confidence = confidence
        existing.note = note
        existing.updated_at = datetime.utcnow()
    return existing, False


def _trace_index(traces: Iterable[MeterQualityTrace]) -> dict[str, MeterQualityTrace]:
    index: dict[str, MeterQualityTrace] = {}
    for trace in traces:
        normalized = normalize_barcode(trace.barcode)
        if normalized and normalized not in index:
            index[normalized] = trace
    return index


def _factory_index(rows: Iterable[FactoryTest]) -> dict[str, FactoryTest]:
    index: dict[str, FactoryTest] = {}
    for row in rows:
        normalized = normalize_barcode(row.meter_id)
        if normalized and normalized not in index:
            index[normalized] = row
    return index


def build_meter_identity(db: Session, include_demo: bool = True) -> dict:
    """Build the barcode bridge table.

    Real links are inserted when the same normalized barcode appears in factory
    data and production trace data.  Aging-test meter numbers are sampled
    batch-level identifiers, so this function no longer creates default
    meter_no-to-barcode demo links.
    """
    traces = db.query(MeterQualityTrace).order_by(MeterQualityTrace.barcode).all()
    factories = db.query(FactoryTest).order_by(FactoryTest.meter_id).all()
    factories_by_norm = _factory_index(factories)

    created = updated = real_links = 0
    errors: list[str] = []

    # The identity table is a derived bridge. Rebuild generated rows to avoid
    # stale demo mappings when the matching strategy or imported data changes.
    db.query(MeterIdentity).filter(
        MeterIdentity.source.in_([
            "barcode_linked",
            "synthetic_demo",
            "sampled_batch_demo",
            "unlinked",
        ])
    ).delete(synchronize_session=False)
    db.flush()

    for trace in traces:
        normalized = normalize_barcode(trace.barcode)
        factory = factories_by_norm.get(normalized)
        if not factory:
            continue
        _, was_created = _upsert_identity(
            db,
            meter_no=None,
            barcode=trace.barcode,
            order_no=trace.order_no or None,
            source="barcode_linked",
            confidence=0.80,
            note="Matched factory_test.meter_id and meter_quality_trace.barcode by normalized serial.",
        )
        real_links += 1
        created += int(was_created)
        updated += int(not was_created)

    # SessionLocal disables autoflush, so make barcode-only rows visible before
    # adding low-confidence meter_no demo links that may reuse the same barcode.
    db.flush()

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        errors.append(str(exc))

    return {
        "identity_total": db.query(MeterIdentity).count(),
        "created": created,
        "updated": updated,
        "barcode_linked": real_links,
        "sampled_meter_to_barcode_links": 0,
        "note": (
            "Aging-test meter_no is treated as sampled batch evidence, not as a "
            "default one-to-one production barcode link."
        ),
        "errors": errors,
    }


def _identity_for_chain(db: Session, meter_no: int) -> MeterIdentity | None:
    return db.query(MeterIdentity).filter_by(meter_no=meter_no).first()


def _trace_for_identity(
    identity: MeterIdentity | None,
    traces_by_norm: dict[str, MeterQualityTrace],
) -> MeterQualityTrace | None:
    if identity is None:
        return None
    return traces_by_norm.get(identity.normalized_barcode or normalize_barcode(identity.barcode))


def _factory_for_trace(
    trace: MeterQualityTrace | None,
    factories_by_norm: dict[str, FactoryTest],
) -> FactoryTest | None:
    if trace is None:
        return None
    return factories_by_norm.get(normalize_barcode(trace.barcode))


def _quality_features(chain: QualityChain, factory: FactoryTest | None) -> dict:
    initial_error = _to_float(chain.initial_error_max, QUALITY_DEFAULTS["initial_error_max"])
    final_error = _to_float(chain.final_error_max, QUALITY_DEFAULTS["final_error_max"])
    degradation = chain.error_degradation_rate
    if degradation is None and (chain.initial_error_max is not None or chain.final_error_max is not None):
        degradation = max(final_error - initial_error, 0.0) / 10.0

    clock_drift = chain.clock_drift_rate
    if clock_drift is None and chain.final_clock_offset is not None:
        clock_drift = _to_float(chain.final_clock_offset) / (1000.0 / 24.0)

    factory_crystal_ppm = (
        factory.crystal_ppm
        if factory is not None and factory.crystal_ppm is not None
        else QUALITY_DEFAULTS["crystal_ppm"]
    )
    factory_ct_ratio_err = (
        factory.ct_ratio_err
        if factory is not None and factory.ct_ratio_err is not None
        else QUALITY_DEFAULTS["ct_ratio_err"]
    )
    factory_clock_accuracy_sd = (
        factory.clock_accuracy_sd
        if factory is not None and factory.clock_accuracy_sd is not None
        else QUALITY_DEFAULTS["clock_accuracy_sd"]
    )

    return {
        "crystal_ppm": _to_float(chain.crystal_ppm, factory_crystal_ppm),
        "ct_ratio_err": _to_float(chain.ct_ratio_err, factory_ct_ratio_err),
        "clock_accuracy_sd": _to_float(chain.clock_accuracy_sd, factory_clock_accuracy_sd),
        "factory_fault": _bool_float(chain.factory_fault or (factory.is_fault if factory else False)),
        "initial_error_max": initial_error,
        "final_error_max": final_error,
        "error_degradation_rate": _to_float(degradation, QUALITY_DEFAULTS["error_degradation_rate"]),
        "clock_drift_rate": _to_float(clock_drift, QUALITY_DEFAULTS["clock_drift_rate"]),
        "clock_daily_error_max": _to_float(chain.clock_daily_error_max, QUALITY_DEFAULTS["clock_daily_error_max"]),
        "error_variability_max": _to_float(chain.error_variability_max, QUALITY_DEFAULTS["error_variability_max"]),
        "error_consistency_max": _to_float(chain.error_consistency_max, QUALITY_DEFAULTS["error_consistency_max"]),
        "energy_balance_ratio": _to_float(chain.energy_balance_ratio, QUALITY_DEFAULTS["energy_balance_ratio"]),
        "energy_peak_ratio": _to_float(chain.energy_peak_ratio, QUALITY_DEFAULTS["energy_peak_ratio"]),
        "rs485_total_errors": _to_float(chain.rs485_total_errors, QUALITY_DEFAULTS["rs485_total_errors"]),
    }


def _infer_demo_batch_key(db: Session) -> tuple[str, str | None]:
    order_nos = [
        row[0]
        for row in db.query(MeterQualityTrace.order_no).distinct().all()
        if row[0]
    ]
    if len(order_nos) == 1:
        return order_nos[0], order_nos[0]
    if order_nos:
        return "MULTI_ORDER_DEMO", None
    return "UNSPECIFIED_BATCH", None


def build_batch_quality_profiles(db: Session) -> dict:
    """Aggregate sampled aging/quality-chain rows into batch-level profiles."""
    chains = db.query(QualityChain).order_by(QualityChain.meter_no).all()
    batch_key, order_no = _infer_demo_batch_key(db)

    db.query(BatchQualityProfile).filter(
        BatchQualityProfile.source.in_(["sampled_aging", "sampled_batch_demo"])
    ).delete(synchronize_session=False)

    if not chains:
        db.commit()
        return {"profile_total": db.query(BatchQualityProfile).count(), "created": 0}

    final_errors = [_to_float(c.final_error_max, 0.0) for c in chains]
    degradation_rates = [
        _to_float(
            c.error_degradation_rate,
            max(_to_float(c.final_error_max, 0.0) - _to_float(c.initial_error_max, 0.0), 0.0) / 10.0,
        )
        for c in chains
    ]
    clock_drifts = [
        _to_float(
            c.clock_drift_rate,
            _to_float(c.final_clock_offset, 0.0) / (1000.0 / 24.0) if c.final_clock_offset is not None else 0.0,
        )
        for c in chains
    ]
    labels = [int(c.quality_label or 0) for c in chains]
    sampled_meter_nos = [c.meter_no for c in chains]

    profile = BatchQualityProfile(
        batch_key=batch_key,
        order_no=order_no,
        source="sampled_aging",
        confidence=0.65 if order_no else 0.40,
        sample_count=len(chains),
        high_risk_count=sum(labels),
        high_risk_rate=sum(labels) / len(chains) if chains else 0.0,
        avg_final_error_max=float(np.mean(final_errors)) if final_errors else 0.0,
        max_final_error_max=max(final_errors) if final_errors else 0.0,
        avg_error_degradation_rate=float(np.mean(degradation_rates)) if degradation_rates else 0.0,
        max_clock_drift_rate=max(clock_drifts) if clock_drifts else 0.0,
        max_rs485_total_errors=max(int(c.rs485_total_errors or 0) for c in chains),
        sampled_meter_nos_json=json.dumps(sampled_meter_nos, ensure_ascii=False),
        note=(
            "Aging rows are sampled meters from the shipment/batch. This profile "
            "describes batch quality stability and should not be interpreted as "
            "a one-to-one barcode label."
        ),
    )
    db.add(profile)
    db.commit()
    return {
        "profile_total": db.query(BatchQualityProfile).count(),
        "created": 1,
        "batch_key": batch_key,
        "sample_count": len(chains),
        "high_risk_rate": profile.high_risk_rate,
    }


def _snapshot_values(snapshot: QualityFeatureSnapshot) -> list[float]:
    if snapshot.feature_json:
        try:
            values = json.loads(snapshot.feature_json)
            if isinstance(values, list) and len(values) == len(SNAPSHOT_FEATURE_NAMES):
                return [_to_float(v) for v in values]
        except json.JSONDecodeError:
            pass
    return [_to_float(getattr(snapshot, name, 0.0)) for name in SNAPSHOT_FEATURE_NAMES]


def features_from_snapshot(snapshot: QualityFeatureSnapshot) -> dict:
    return dict(zip(SNAPSHOT_FEATURE_NAMES, _snapshot_values(snapshot)))


def build_quality_feature_snapshots(
    db: Session,
    include_synthetic_demo: bool = False,
) -> dict:
    """Create reproducible fused feature rows for training and prediction."""
    identity_stats = build_meter_identity(db, include_demo=include_synthetic_demo)
    batch_stats = build_batch_quality_profiles(db)
    traces_by_norm = _trace_index(db.query(MeterQualityTrace).all())
    factories_by_norm = _factory_index(db.query(FactoryTest).all())
    chains = db.query(QualityChain).order_by(QualityChain.meter_no).all()

    created = updated = linked = unlinked = synthetic_demo = sampled_aging_unlinked = 0
    for chain in chains:
        identity = _identity_for_chain(db, chain.meter_no)
        trace = _trace_for_identity(identity, traces_by_norm)
        factory = _factory_for_trace(trace, factories_by_norm)
        quality = _quality_features(chain, factory)
        production = (
            get_production_features_for_barcode(trace.barcode, db)
            if trace is not None
            else {name: 0.0 for name in PRODUCTION_FEATURE_NAMES}
        )
        values = {**quality, **production}
        feature_row = [_to_float(values.get(name, 0.0)) for name in SNAPSHOT_FEATURE_NAMES]

        source = "sampled_aging_unlinked"
        confidence = 0.0
        barcode = normalized_barcode = None
        order_no = None
        batch_key = batch_stats.get("batch_key")
        if identity is not None:
            source = identity.source or "linked"
            confidence = _to_float(identity.confidence, 0.0)
            barcode = identity.barcode
            normalized_barcode = identity.normalized_barcode
            if trace is not None:
                order_no = trace.order_no
                batch_key = trace.order_no or batch_key
            if source == "synthetic_demo":
                synthetic_demo += 1
            else:
                linked += 1
        else:
            unlinked += 1
            sampled_aging_unlinked += 1

        snapshot = db.query(QualityFeatureSnapshot).filter_by(meter_no=chain.meter_no).first()
        if snapshot is None:
            snapshot = QualityFeatureSnapshot(meter_no=chain.meter_no)
            db.add(snapshot)
            created += 1
        else:
            updated += 1

        snapshot.barcode = barcode
        snapshot.normalized_barcode = normalized_barcode
        snapshot.order_no = order_no
        snapshot.batch_key = batch_key
        snapshot.source = source
        snapshot.confidence = confidence
        for name in SNAPSHOT_FEATURE_NAMES:
            setattr(snapshot, name, values.get(name, 0.0))
        snapshot.quality_label = int(chain.quality_label or 0)
        snapshot.quality_reason = chain.quality_reason
        snapshot.feature_json = json.dumps(feature_row, ensure_ascii=False)
        snapshot.updated_at = datetime.utcnow()

    db.commit()
    return {
        "identity": identity_stats,
        "batch_profile": batch_stats,
        "snapshot_total": db.query(QualityFeatureSnapshot).count(),
        "created": created,
        "updated": updated,
        "linked": linked,
        "synthetic_demo": synthetic_demo,
        "sampled_aging_unlinked": sampled_aging_unlinked,
        "unlinked": unlinked,
    }


def get_snapshot_feature_matrix(db: Session):
    snapshots = db.query(QualityFeatureSnapshot).order_by(QualityFeatureSnapshot.meter_no).all()
    process_profiles = db.query(ProductionProcessProfile).all()
    process_by_norm = _process_profile_index(process_profiles)
    _, reliability_features = get_reliability_feature_matrix(db, len(snapshots))
    X_rows, y, meta = [], [], []
    for snapshot in snapshots:
        normalized = snapshot.normalized_barcode or normalize_barcode(snapshot.barcode)
        process_profile = _process_profile_for_barcode(normalized, process_by_norm, process_profiles)
        X_rows.append(
            _snapshot_values(snapshot)
            + _process_profile_features(process_profile, normalized)
            + reliability_features
        )
        y.append(int(snapshot.quality_label or 0))
        meta.append({
            "meter_no": snapshot.meter_no,
            "barcode": snapshot.barcode,
            "batch_key": snapshot.batch_key,
            "process_profile_key": process_profile.batch_key if process_profile else None,
            "reliability_sample_features": bool(any(reliability_features)),
            "source": snapshot.source,
            "confidence": snapshot.confidence,
            "is_augmented": snapshot.source in {"synthetic_demo", "sampled_batch_demo"},
            "is_sampled_aging": snapshot.source == "sampled_aging_unlinked",
        })
    return np.array(X_rows, dtype=float), np.array(y, dtype=int), meta


def snapshot_for_meter(db: Session, meter_no: int) -> QualityFeatureSnapshot | None:
    return db.query(QualityFeatureSnapshot).filter_by(meter_no=meter_no).first()
