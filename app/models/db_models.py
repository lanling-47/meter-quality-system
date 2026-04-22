"""SQLAlchemy 数据模型定义"""
import json
from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from database import Base


class CrystalData(Base):
    """晶振来料检测数据"""
    __tablename__ = "crystal_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_no = Column(String(50), nullable=False, index=True, comment="批次号")
    sample_no = Column(Integer, nullable=False, comment="样品编号 1-10")
    model = Column(String(50), comment="型号 HC-49SMD")
    center_freq_hz = Column(Float, comment="中心频率 Hz")
    load_cap_pf = Column(Float, comment="负载电容 pF")
    freq_error_ppm = Column(Float, comment="常温频率误差 ppm")
    is_qualified = Column(Boolean, default=True, comment="是否合格")
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class CTData(Base):
    """互感器来料检测数据"""
    __tablename__ = "ct_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_no = Column(String(50), nullable=False, index=True, comment="批次号")
    sample_no = Column(Integer, nullable=False, comment="样品编号 1-10")
    model = Column(String(100), comment="型号 HDCT3 5(60)A2mA")
    # 比差（多电流点，存为JSON列表）
    ratio_errors_json = Column(Text, comment="比差多点数据 JSON")
    # 角差（多电流点）
    angle_errors_json = Column(Text, comment="角差多点数据 JSON")
    # 关键单值
    ratio_error_max = Column(Float, comment="最大比差 %")
    ratio_error_range = Column(Float, comment="比差极差（线性度指标）%")
    angle_error_max = Column(Float, comment="最大角差 min")
    dc_resistance = Column(Float, comment="抗直流特性偏差")
    is_qualified = Column(Boolean, default=True, comment="是否合格")
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        d['ratio_errors'] = json.loads(self.ratio_errors_json) if self.ratio_errors_json else []
        d['angle_errors'] = json.loads(self.angle_errors_json) if self.angle_errors_json else []
        return d


class FactoryTest(Base):
    """电表出厂检测数据"""
    __tablename__ = "factory_test"

    id = Column(Integer, primary_key=True, autoincrement=True)
    meter_id = Column(String(50), nullable=False, unique=True, index=True, comment="电表唯一ID（22位）")
    crystal_batch = Column(String(50), comment="使用晶振批次")
    ct_batch = Column(String(50), comment="使用互感器批次")
    crystal_ppm = Column(Float, comment="晶振频率误差 ppm")
    ct_ratio_err = Column(Float, comment="互感器比差 %")
    fault_type = Column(String(50), comment="故障类型（正常/频差偏大/比差偏大）")
    is_fault = Column(Boolean, default=False, comment="是否有故障")
    meter_constant_err = Column(Float, comment="仪表常数误差 %")
    clock_accuracy_sd = Column(Float, comment="时钟准确度 s/d")
    rs485_ok = Column(Boolean, default=True, comment="RS485通信是否正常")
    ir_ok = Column(Boolean, default=True, comment="红外通信是否正常")
    module_ok = Column(Boolean, default=True, comment="通信模块是否正常")
    startup_ok = Column(Boolean, default=True, comment="起动试验")
    creep_ok = Column(Boolean, default=True, comment="潜动试验")
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class AgingTest(Base):
    """1000h加速老化退化试验数据（含全量Sheet特征）"""
    __tablename__ = "aging_test"

    id = Column(Integer, primary_key=True, autoincrement=True)
    meter_no = Column(Integer, nullable=False, index=True, comment="电表编号 196-232")
    test_date_label = Column(String(20), nullable=False, comment="测试时间点标签 如 6.18、6.29")
    test_hours = Column(Float, comment="近似测试时长 h")
    test_condition = Column(String(200), comment="试验条件")
    basic_errors_json = Column(Text, comment="基本误差多工况数据 JSON")
    error_a_imin_10 = Column(Float, comment="A Imin 1.0 基本误差 %")
    error_a_imin_05l = Column(Float, comment="A Imin 0.5L 基本误差 %")
    error_max = Column(Float, comment="所有工况最大基本误差 %")
    clock_offset_s = Column(Integer, comment="与北京时间差值 s")
    clock_daily_error_sd = Column(Float, comment="日误差 s/d（日计时Sheet）")
    error_variability = Column(Float, comment="误差变差最大值 %")
    error_consistency = Column(Float, comment="误差一致性偏差最大值 %")
    energy_total = Column(Float, comment="组合有功总电能（kWh/1000）")
    energy_forward = Column(Float, comment="正向有功总电能")
    energy_balance_ratio = Column(Float, comment="正反向电量比")
    energy_peak_ratio = Column(Float, comment="尖电量占比")
    rs485_errors = Column(Integer, default=0, comment="RS485通信异常次数")
    quality_label = Column(Integer, default=0, comment="0=正常 1=高风险")
    quality_reason = Column(String(200), comment="标注原因")
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        d['basic_errors'] = json.loads(self.basic_errors_json) if self.basic_errors_json else {}
        return d


class QualityChain(Base):
    """质量链路关联表（每块电表汇总特征）"""
    __tablename__ = "quality_chain"

    id = Column(Integer, primary_key=True, autoincrement=True)
    meter_no = Column(Integer, nullable=False, unique=True, index=True, comment="电表编号")
    # 来料特征
    crystal_batch = Column(String(50))
    ct_batch = Column(String(50))
    crystal_ppm = Column(Float, comment="晶振频率误差 ppm")
    ct_ratio_err = Column(Float, comment="互感器比差 %")
    # 出厂检测特征
    clock_accuracy_sd = Column(Float, comment="出厂时钟准确度 s/d")
    factory_fault = Column(Boolean, default=False, comment="出厂时是否故障")
    factory_fault_type = Column(String(50))
    # 老化试验基础特征
    initial_error_max = Column(Float, comment="试验初始最大基本误差 %")
    final_error_max = Column(Float, comment="试验末期最大基本误差 %")
    error_degradation_rate = Column(Float, comment="误差退化速率 %/100h")
    initial_clock_offset = Column(Integer, comment="初始时钟偏差 s")
    final_clock_offset = Column(Integer, comment="末期时钟偏差 s")
    clock_drift_rate = Column(Float, comment="时钟漂移速率 s/d")
    rs485_total_errors = Column(Integer, default=0, comment="RS485累计异常次数")
    # ── 新增特征（来自未使用的Sheets）──────────────────────────────────
    clock_daily_error_max = Column(Float, comment="日计时最大日误差 s/d（日计时Sheet末期值）")
    error_variability_max = Column(Float, comment="误差变差最大值 %（末期，误差变差Sheet）")
    error_consistency_max = Column(Float, comment="误差一致性偏差最大绝对值 %（末期）")
    cover_open_count = Column(Integer, default=0, comment="开表盖次数（开表盖Sheet）")
    energy_balance_ratio = Column(Float, comment="正反向电量比（末期，正反向电量Sheet）")
    energy_peak_ratio = Column(Float, comment="尖电量占比（末期）")
    # 数据增强标记
    is_augmented = Column(Boolean, default=False, comment="是否为增强数据")
    # 目标标签
    quality_label = Column(Integer, default=0, comment="0=正常 1=高风险")
    quality_reason = Column(String(500))
    # 预测结果
    predicted_risk = Column(Float, comment="模型预测风险概率 0-1")
    predicted_label = Column(Integer, comment="模型预测类别")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class ProductionInspection(Base):
    """生产过程工序检测记录（CCD/功能测试/耐压/复校）"""
    __tablename__ = "production_inspection"

    id = Column(Integer, primary_key=True, autoincrement=True)
    barcode = Column(String(30), nullable=False, index=True, comment="电表条形码 ATXM")
    order_no = Column(String(30), index=True, comment="订单号 DDH")
    station_no = Column(String(20), comment="表位号 BWH")
    process = Column(String(30), nullable=False, comment="工序名称 DQGX")
    test_count = Column(Integer, default=1, comment="测试次数 JYCS")
    result = Column(String(10), comment="结论（合格/不合格）JYJL")
    test_time = Column(DateTime, comment="测试时间 JYRQ")
    device_no = Column(String(30), comment="设备编号 JYZZ")
    remark = Column(String(500), comment="不合格项备注 BZ")
    leak_current = Column(Float, comment="漏电流 LDL（仅耐压工序）")
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class AoiDefect(Base):
    """AOI不合格/缺陷记录（贴片AOI + 波峰焊AOI）"""
    __tablename__ = "aoi_defect"

    id = Column(Integer, primary_key=True, autoincrement=True)
    barcode = Column(String(30), nullable=False, index=True, comment="电表条形码 TXM")
    order_no = Column(String(30), index=True, comment="订单号 DDH")
    device_no = Column(String(30), comment="设备编号 SBBH")
    defect_point = Column(String(50), comment="故障点 GZD")
    defect_desc = Column(String(100), comment="故障描述 GZMS")
    process_no = Column(String(20), comment="当前工序 DQGX")
    detect_time = Column(DateTime, comment="检测时间 SJ")
    aoi_type = Column(String(20), comment="AOI类型：smt贴片/wave波峰焊")
    quality_risk = Column(String(200), comment="质量隐患描述（来自故障字典）")
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class RepairRecord(Base):
    """维修记录（含故障现象/原因/工序）"""
    __tablename__ = "repair_record"

    id = Column(Integer, primary_key=True, autoincrement=True)
    barcode = Column(String(30), nullable=False, index=True, comment="电表条形码 TXM")
    order_no = Column(String(30), index=True, comment="订单号 PCBM")
    fault_code = Column(String(20), comment="故障现象代码 GZXXDM")
    fault_name = Column(String(100), comment="故障现象名称 GZXXMC")
    cause_code = Column(String(20), comment="故障原因代码 GZYYDM")
    cause_name = Column(String(100), comment="故障原因名称 GZYYMC")
    fault_location = Column(String(100), comment="故障地点 GZDD")
    fault_process = Column(String(20), comment="故障工序 GZGX")
    fault_time = Column(DateTime, comment="故障记录时间 GZJLSJ")
    repair_time = Column(DateTime, comment="维修记录时间 WXJLSJ")
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class BoardTestDetail(Base):
    """单板详细测试记录（按测试项目展开）"""
    __tablename__ = "board_test_detail"

    id = Column(Integer, primary_key=True, autoincrement=True)
    barcode = Column(String(30), nullable=False, index=True, comment="电表条形码 CS_TXM")
    order_no = Column(String(30), index=True, comment="订单号 DDH")
    scheme_name = Column(String(50), comment="方案名称 CS_XMM")
    seq_no = Column(Integer, comment="顺序号 CS_XH")
    test_item = Column(String(100), comment="测试项目 CS_XM")
    result = Column(String(20), comment="结论 CS_JG")
    test_data = Column(String(200), comment="测试数据 CS_DATA")
    test_time = Column(DateTime, comment="测试时间 CS_DATE")
    device_no = Column(String(30), comment="设备编号 GZBH")
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class MeterQualityTrace(Base):
    """电表全工序质量追溯汇总（每块电表一条记录）"""
    __tablename__ = "meter_quality_trace"

    id = Column(Integer, primary_key=True, autoincrement=True)
    barcode = Column(String(30), nullable=False, unique=True, index=True, comment="电表条形码")
    order_no = Column(String(30), index=True, comment="订单号")
    # 工序通过情况
    ccd_pass = Column(Boolean, comment="CCD检测通过")
    ccd_retry_count = Column(Integer, default=0, comment="CCD重测次数")
    hipot_pass = Column(Boolean, comment="耐压测试通过")
    leak_current_max = Column(Float, comment="最大漏电流")
    func_pass = Column(Boolean, comment="功能测试通过")
    func_retry_count = Column(Integer, default=0, comment="功能测试重测次数")
    func_fail_items = Column(Text, comment="功能测试不合格项 JSON列表")
    reverify_pass = Column(Boolean, comment="复校通过")
    reverify_retry_count = Column(Integer, default=0, comment="复校重测次数")
    reverify_fail_items = Column(Text, comment="复校不合格项 JSON列表")
    # AOI情况
    smt_aoi_defect = Column(Boolean, default=False, comment="贴片AOI发现缺陷")
    smt_aoi_defect_types = Column(Text, comment="贴片AOI缺陷类型 JSON")
    wave_aoi_defect = Column(Boolean, default=False, comment="波峰焊AOI发现缺陷")
    wave_aoi_defect_types = Column(Text, comment="波峰焊AOI缺陷类型 JSON")
    # 维修情况
    repaired = Column(Boolean, default=False, comment="是否经过维修")
    repair_count = Column(Integer, default=0, comment="维修次数")
    fault_processes = Column(Text, comment="故障工序列表 JSON")
    fault_names = Column(Text, comment="故障现象列表 JSON")
    # 最终结果
    final_pass = Column(Boolean, comment="最终是否合格（所有工序均通过）")
    total_retry_count = Column(Integer, default=0, comment="总重测次数")
    quality_risk_score = Column(Float, comment="综合质量风险评分 0-1")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class ModelVersion(Base):
    """AI模型版本管理"""
    __tablename__ = "model_version"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version_tag = Column(String(50), nullable=False, unique=True, comment="版本标签 如 v1.0_20260315")
    model_type = Column(String(50), comment="模型类型 xgboost/lightgbm")
    model_path = Column(String(200), comment="模型文件路径")
    n_samples = Column(Integer, comment="训练样本数")
    auc_score = Column(Float, comment="ROC-AUC 分数")
    f1_score = Column(Float, comment="F1 分数（高风险类）")
    feature_names_json = Column(Text, comment="特征名列表 JSON")
    is_active = Column(Boolean, default=False, comment="是否为当前激活模型")
    train_log = Column(Text, comment="训练日志")
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class DataIngestionLog(Base):
    """
    数据采集日志
    记录每次文件轮询发现的新文件及解析结果
    """
    __tablename__ = "data_ingestion_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_name = Column(String(255), nullable=False, comment="文件名")
    file_path = Column(String(500), comment="文件完整路径")
    file_hash = Column(String(64), unique=True, index=True, comment="文件 MD5，用于去重")
    data_type = Column(String(50), comment="数据类型 ccd/functest/reverify/hipot/aoi/repair/board_test")
    record_count = Column(Integer, default=0, comment="解析到的总记录数")
    new_count = Column(Integer, default=0, comment="本次新增记录数（去重后）")
    status = Column(String(20), default="pending", comment="pending/reviewed/trained/error")
    error_msg = Column(Text, comment="解析错误信息（如有）")
    parsed_at = Column(DateTime, default=datetime.utcnow, comment="解析完成时间")
    reviewed_at = Column(DateTime, nullable=True, comment="审核时间")
    reviewed_by = Column(String(50), nullable=True, comment="审核人")
    triggered_train = Column(Boolean, default=False, comment="是否已触发重训练")

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        if d.get("parsed_at"):
            d["parsed_at"] = d["parsed_at"].strftime("%Y-%m-%d %H:%M:%S")
        if d.get("reviewed_at"):
            d["reviewed_at"] = d["reviewed_at"].strftime("%Y-%m-%d %H:%M:%S")
        return d


class QualityLabelReview(Base):
    """
    质量标注人工审核队列
    新数据自动打标后，进入此队列等待人工确认，
    确认后纳入训练集
    """
    __tablename__ = "quality_label_review"

    id = Column(Integer, primary_key=True, autoincrement=True)
    barcode = Column(String(50), index=True, comment="条形码（生产数据主键）")
    meter_no = Column(Integer, nullable=True, comment="电表编号（关联老化数据，可空）")
    source = Column(String(50), comment="数据来源：file_watch/manual/import")
    ingestion_log_id = Column(Integer, ForeignKey("data_ingestion_log.id"), nullable=True)

    # 自动打标结果（来自当前激活模型）
    auto_label = Column(Integer, comment="自动标注结果 0=正常 1=高风险")
    auto_confidence = Column(Float, comment="模型预测置信度")
    auto_risk_prob = Column(Float, comment="模型预测风险概率")
    feature_snapshot = Column(Text, comment="打标时使用的特征快照 JSON")

    # 人工审核结果
    manual_label = Column(Integer, nullable=True, comment="人工确认标注 0/1，null=未审核")
    review_status = Column(String(20), default="pending", comment="pending/confirmed/rejected/skipped")
    review_note = Column(Text, nullable=True, comment="审核备注（如发现异常说明原因）")
    reviewed_by = Column(String(50), nullable=True, comment="审核人")

    # 是否已加入训练集
    added_to_train = Column(Boolean, default=False, comment="是否已加入训练集")

    created_at = Column(DateTime, default=datetime.utcnow)
    reviewed_at = Column(DateTime, nullable=True)

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        if d.get("created_at"):
            d["created_at"] = d["created_at"].strftime("%Y-%m-%d %H:%M:%S")
        if d.get("reviewed_at"):
            d["reviewed_at"] = d["reviewed_at"].strftime("%Y-%m-%d %H:%M:%S")
        return d

