"""
文件轮询监控服务（FileWatcherService）

功能：
  1. 定时扫描监控目录，发现新的 Excel/CSV 文件
  2. 通过 MD5 去重，避免重复导入
  3. 根据文件名关键词自动识别数据类型
  4. 调用 production_etl_service 差量解析入库
  5. 记录采集日志（DataIngestionLog）
  6. 通过 SSE 通知前端有新数据到达

使用方式：
  # 在 FastAPI startup 中启动
  watcher = FileWatcherService()
  asyncio.create_task(watcher.start())
"""
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import config as cfg


# ─── 全局通知事件队列（供 SSE /notify/stream 消费）───────────────────────────
_notify_queues: list[asyncio.Queue] = []


def emit_notification(event: str, data: dict):
    """向所有前端订阅者推送通知"""
    msg = json.dumps({"event": event, "data": data}, ensure_ascii=False)
    for q in list(_notify_queues):
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass


async def notify_event_stream():
    """SSE 通知事件流生成器（/notify/stream 使用）"""
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    _notify_queues.append(q)
    try:
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=30)
                yield f"data: {msg}\n\n"
            except asyncio.TimeoutError:
                yield "data: {\"event\":\"heartbeat\"}\n\n"
    finally:
        if q in _notify_queues:
            _notify_queues.remove(q)


# ─── 文件名 → 数据类型映射规则 ─────────────────────────────────────────────
DATA_TYPE_RULES = [
    (["ccd", "CCD", "外观"],                                "ccd"),
    (["功能测试", "func", "功能"],                           "functest"),
    (["复校", "reverify", "复检"],                           "reverify"),
    (["耐压", "hipot", "绝缘"],                              "hipot"),
    (["贴片AOI", "SMT", "smt_aoi", "贴片"],                  "smt_aoi"),
    (["波峰", "波峰焊", "wave", "选择性"],                    "wave_aoi"),
    (["AOI", "aoi"],                                         "aoi"),
    (["维修", "repair", "返修"],                             "repair"),
    (["单板", "board", "单板测试"],                           "board_test"),
    (["晶振", "crystal"],                                     "crystal"),
    (["互感器", "ct", "CT"],                                  "ct"),
]


def infer_data_type(filename: str) -> str:
    """通过文件名关键词推断数据类型"""
    for keywords, dtype in DATA_TYPE_RULES:
        for kw in keywords:
            if kw in filename:
                return dtype
    return "unknown"


def md5_of_file(path: Path) -> str:
    """计算文件 MD5"""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


# ─── 差量入库：按条形码 upsert ──────────────────────────────────────────────

def _upsert_inspection_records(records: list[dict], process: str, db) -> int:
    """工序检测差量入库，按 (barcode, process) upsert"""
    from app.models.db_models import ProductionInspection
    new_count = 0
    for r in records:
        barcode = r.get("barcode")
        if not barcode:
            continue
        existing = db.query(ProductionInspection).filter_by(
            barcode=barcode, process=process
        ).order_by(ProductionInspection.id.desc()).first()
        if existing is None:
            obj = ProductionInspection(**{k: v for k, v in r.items()
                                         if hasattr(ProductionInspection, k)})
            db.add(obj)
            new_count += 1
        # 如果已有记录，本次追加新的测试次数（允许多次测试）
        # 这里简化：仅在没有任何记录时插入
    db.commit()
    return new_count


def _upsert_aoi_records(records: list[dict], aoi_type: str, db) -> int:
    """AOI 缺陷差量入库"""
    from app.models.db_models import AoiDefect
    new_count = 0
    for r in records:
        barcode = r.get("barcode")
        if not barcode:
            continue
        existing = db.query(AoiDefect).filter_by(
            barcode=barcode, aoi_type=aoi_type
        ).first()
        if existing is None:
            obj = AoiDefect(**{k: v for k, v in r.items()
                               if hasattr(AoiDefect, k)})
            db.add(obj)
            new_count += 1
    db.commit()
    return new_count


def _upsert_repair_records(records: list[dict], db) -> int:
    """维修记录差量入库"""
    from app.models.db_models import RepairRecord
    new_count = 0
    for r in records:
        barcode = r.get("barcode")
        if not barcode:
            continue
        existing = db.query(RepairRecord).filter_by(barcode=barcode).first()
        if existing is None:
            obj = RepairRecord(**{k: v for k, v in r.items()
                                  if hasattr(RepairRecord, k)})
            db.add(obj)
            new_count += 1
    db.commit()
    return new_count


def parse_and_upsert(file_path: Path, data_type: str, db) -> tuple[int, int]:
    """
    解析文件并差量入库

    Returns
    -------
    (record_count, new_count)
    """
    from app.services.production_etl_service import (
        parse_inspection, parse_aoi_defects, parse_repair_records,
        parse_board_test,
    )

    record_count = 0
    new_count = 0

    try:
        if data_type in ("ccd", "functest", "reverify", "hipot"):
            process_map = {
                "ccd": "CCD", "functest": "功能测试",
                "reverify": "复校", "hipot": "耐压",
            }
            process = process_map[data_type]
            records = parse_inspection(str(file_path), process)
            record_count = len(records)
            new_count = _upsert_inspection_records(records, process, db)

        elif data_type in ("smt_aoi", "wave_aoi", "aoi"):
            aoi_type = "SMT" if data_type == "smt_aoi" else ("波峰焊" if data_type == "wave_aoi" else "SMT")
            records = parse_aoi_defects(str(file_path), aoi_type, {})
            record_count = len(records)
            new_count = _upsert_aoi_records(records, aoi_type, db)

        elif data_type == "repair":
            records = parse_repair_records(str(file_path))
            record_count = len(records)
            new_count = _upsert_repair_records(records, db)

        elif data_type == "board_test":
            records = parse_board_test(str(file_path), failed_only=True)
            record_count = len(records)
            # board_test 直接用 production_etl_service 的写入逻辑（简化）
            from app.models.db_models import BoardTestDetail
            existing_barcodes = {r[0] for r in db.query(BoardTestDetail.barcode).all()}
            new_records = [r for r in records if r.get("barcode") not in existing_barcodes]
            db.bulk_save_objects([BoardTestDetail(**{k: v for k, v in r.items()
                                                    if hasattr(BoardTestDetail, k)})
                                  for r in new_records])
            db.commit()
            new_count = len(new_records)

    except Exception as e:
        raise RuntimeError(f"解析 {data_type} 文件失败: {e}")

    return record_count, new_count


# ─── 自动打标（用当前激活模型预测，入 QualityLabelReview 队列）─────────────

def auto_label_new_records(new_count: int, data_type: str, log_id: int, db) -> int:
    """
    对新入库的生产数据，用当前模型预测风险，写入审核队列
    （因生产数据无 meter_no，仅记录 barcode + 预测结果）

    Returns
    -------
    入队数量
    """
    if new_count == 0:
        return 0

    from app.models.db_models import MeterQualityTrace, QualityLabelReview
    from app.services.predict_service import load_active_model, FEATURE_NAMES
    import numpy as np

    model, feat_names, version, scaler, model_type = load_active_model(db)
    if model is None:
        return 0  # 无模型，跳过自动打标

    # 取最近 new_count 条未在审核队列中的追溯汇总
    existing_barcodes = {
        r[0] for r in db.query(QualityLabelReview.barcode).filter(
            QualityLabelReview.review_status == "pending"
        ).all()
    }

    # 从 MeterQualityTrace 取最近更新的记录
    recent_traces = db.query(MeterQualityTrace).order_by(
        MeterQualityTrace.id.desc()
    ).limit(new_count * 2).all()

    added = 0
    for trace in recent_traces:
        if trace.barcode in existing_barcodes:
            continue
        if added >= new_count:
            break

        # 构造特征向量（基于 MeterQualityTrace 字段）
        feat_vec = np.array([[
            0.0, 0.0, 0.0, 0.0,  # 来料特征（无）
            0.0, 0.0, 0.0, 0.0,  # 老化特征（无）
            0.0, 0.0, 0.0, 1.0, 0.246,  # 其他
            0.0,  # rs485
            float(trace.ccd_retry_count or 0),
            float(trace.func_retry_count or 0),
            float(trace.reverify_retry_count or 0),
            float((trace.ccd_retry_count or 0) + (trace.func_retry_count or 0) + (trace.reverify_retry_count or 0)),
            float(trace.smt_aoi_defect or False),
            float(trace.wave_aoi_defect or False),
            float(trace.repaired or False),
            float(trace.repair_count or 0),
            0.0,  # leak_current_max（未存）
            float(trace.quality_risk_score or 0.0),
        ]])

        try:
            x_input = scaler.transform(feat_vec) if scaler is not None else feat_vec
            risk_prob = float(model.predict_proba(x_input)[0][1])
            auto_label = 1 if risk_prob >= 0.5 else 0
            confidence = abs(risk_prob - 0.5) * 2
        except Exception:
            risk_prob, auto_label, confidence = 0.0, 0, 0.0

        review = QualityLabelReview(
            barcode=trace.barcode,
            meter_no=None,
            source="file_watch",
            ingestion_log_id=log_id,
            auto_label=auto_label,
            auto_confidence=round(confidence, 4),
            auto_risk_prob=round(risk_prob, 4),
            feature_snapshot=json.dumps([round(float(v), 4) for v in feat_vec[0]]),
        )
        db.add(review)
        added += 1

    db.commit()
    return added


# ─── 主监控服务 ────────────────────────────────────────────────────────────

class FileWatcherService:
    """
    生产线文件监控服务

    启动后以 asyncio 后台任务运行，每隔 WATCH_INTERVAL_SEC 秒扫描一次
    """

    def __init__(self):
        self.watch_dir = cfg.WATCH_DIR
        self.interval = cfg.WATCH_INTERVAL_SEC
        self._running = False
        self._seen_hashes: set = set()
        self._db_factory = None  # 由 main.py 注入 SessionLocal

    def set_db_factory(self, db_factory):
        """注入数据库 Session 工厂"""
        self._db_factory = db_factory

    def _load_seen_hashes(self):
        """从数据库加载已处理文件的 MD5 集合（防重启后重复导入）"""
        if self._db_factory is None:
            return
        db = self._db_factory()
        try:
            from app.models.db_models import DataIngestionLog
            hashes = {r[0] for r in db.query(DataIngestionLog.file_hash).all() if r[0]}
            self._seen_hashes = hashes
        except Exception:
            pass
        finally:
            db.close()

    async def start(self):
        """启动监控循环"""
        self._running = True
        self._load_seen_hashes()
        print(f"📡 文件监控服务已启动，监控目录：{self.watch_dir}，间隔：{self.interval}s")
        emit_notification("watcher_started", {
            "watch_dir": str(self.watch_dir),
            "interval": self.interval,
            "message": f"文件监控已启动，监控目录：{self.watch_dir}",
        })
        while self._running:
            try:
                await self._scan_once()
            except Exception as e:
                print(f"⚠️  文件监控扫描异常：{e}")
            await asyncio.sleep(self.interval)

    def stop(self):
        self._running = False

    async def scan_now(self) -> dict:
        """手动触发一次立即扫描（由 API 接口调用）"""
        return await self._scan_once()

    async def _scan_once(self) -> dict:
        """
        执行一次目录扫描

        Returns
        -------
        扫描摘要 dict
        """
        if not self.watch_dir.exists():
            return {"scanned": 0, "new_files": 0}

        found_files = list(self.watch_dir.glob("**/*.xlsx")) + \
                      list(self.watch_dir.glob("**/*.xls")) + \
                      list(self.watch_dir.glob("**/*.csv"))

        new_files = []
        for file_path in found_files:
            if "processed" in str(file_path):
                continue  # 跳过 processed 子目录
            file_hash = md5_of_file(file_path)
            if not file_hash or file_hash in self._seen_hashes:
                continue
            new_files.append((file_path, file_hash))

        results = []
        for file_path, file_hash in new_files:
            result = await self._process_file(file_path, file_hash)
            results.append(result)

        summary = {
            "scanned": len(found_files),
            "new_files": len(new_files),
            "processed": results,
            "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        if new_files:
            emit_notification("scan_summary", {
                "new_files": len(new_files),
                "message": f"发现 {len(new_files)} 个新文件",
                **summary,
            })

        return summary

    async def _process_file(self, file_path: Path, file_hash: str) -> dict:
        """处理单个新文件"""
        if self._db_factory is None:
            return {"file": str(file_path), "error": "DB未初始化"}

        data_type = infer_data_type(file_path.name)
        db = self._db_factory()
        log_id = None

        try:
            # 1. 先写入采集日志（pending 状态）
            from app.models.db_models import DataIngestionLog
            log_entry = DataIngestionLog(
                file_name=file_path.name,
                file_path=str(file_path),
                file_hash=file_hash,
                data_type=data_type,
                status="pending",
            )
            db.add(log_entry)
            db.commit()
            db.refresh(log_entry)
            log_id = log_entry.id

            # 2. 解析 + 差量入库
            record_count, new_count = parse_and_upsert(file_path, data_type, db)

            # 3. 更新日志
            log_entry.record_count = record_count
            log_entry.new_count = new_count
            log_entry.status = "reviewed" if new_count == 0 else "pending"
            db.commit()

            # 4. 自动打标，进入审核队列
            labeled = 0
            if new_count > 0:
                labeled = auto_label_new_records(new_count, data_type, log_id, db)

            # 5. 记录已处理
            self._seen_hashes.add(file_hash)

            # 6. 推送通知
            emit_notification("new_data", {
                "file_name": file_path.name,
                "data_type": data_type,
                "record_count": record_count,
                "new_count": new_count,
                "labeled_count": labeled,
                "log_id": log_id,
                "message": f"新文件 {file_path.name}：{new_count} 条新记录入库，{labeled} 条进入审核队列",
            })

            result = {
                "file": file_path.name,
                "data_type": data_type,
                "record_count": record_count,
                "new_count": new_count,
                "labeled": labeled,
            }
            print(f"✅ 处理文件 {file_path.name}：{new_count} 新记录，{labeled} 入审核队列")
            return result

        except Exception as e:
            error_msg = str(e)
            if log_id is not None:
                try:
                    from app.models.db_models import DataIngestionLog
                    log_entry = db.query(DataIngestionLog).filter_by(id=log_id).first()
                    if log_entry:
                        log_entry.status = "error"
                        log_entry.error_msg = error_msg
                        db.commit()
                except Exception:
                    pass
            self._seen_hashes.add(file_hash)  # 防止无限重试
            emit_notification("error", {
                "file_name": file_path.name,
                "message": f"处理失败: {error_msg}",
            })
            print(f"❌ 处理文件 {file_path.name} 失败: {error_msg}")
            return {"file": file_path.name, "error": error_msg}
        finally:
            db.close()


# 全局单例
_watcher_instance: Optional[FileWatcherService] = None


def get_watcher() -> FileWatcherService:
    """获取全局监控服务实例"""
    global _watcher_instance
    if _watcher_instance is None:
        _watcher_instance = FileWatcherService()
    return _watcher_instance
