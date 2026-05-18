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


class ProductionProcessProfile(Base):
    """数字化电路板生产过程画像，来源于出厂检测工作簿下方的工艺参数区块。"""
    __tablename__ = "production_process_profile"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_key = Column(String(120), nullable=False, unique=True, index=True, comment="批次/文件画像标识")
    source_file = Column(String(255), nullable=True, comment="来源 Excel 文件名")
    meter_ids_json = Column(Text, nullable=True, comment="该工艺画像覆盖的电表 ID 列表 JSON")
    normalized_meter_ids_json = Column(Text, nullable=True, comment="标准化电表 ID 列表 JSON")

    smt_pressure_mm = Column(Float, nullable=True, comment="SMT 贴片压力/贴装位移 mm")
    smt_precision_tol_mm = Column(Float, nullable=True, comment="SMT 贴片精度公差 mm")
    smt_precision_cpk = Column(Float, nullable=True, comment="SMT 贴片精度 CPK 下限")

    reflow_temp_min_c = Column(Float, nullable=True, comment="回流焊温区最低温度 ℃")
    reflow_temp_max_c = Column(Float, nullable=True, comment="回流焊温区最高温度 ℃")
    reflow_temp_span_c = Column(Float, nullable=True, comment="回流焊温区跨度 ℃")
    reflow_conveyor_speed_cm_min = Column(Float, nullable=True, comment="回流焊传送带速度 cm/min")
    reflow_profile_json = Column(Text, nullable=True, comment="回流焊温度曲线 JSON")

    online_current_min_ua = Column(Float, nullable=True, comment="在线测试 PCB 电流最小值 uA")
    online_current_max_ua = Column(Float, nullable=True, comment="在线测试 PCB 电流最大值 uA")
    online_current_avg_ua = Column(Float, nullable=True, comment="在线测试 PCB 电流均值 uA")
    online_current_by_meter_json = Column(Text, nullable=True, comment="单表在线测试电流映射 JSON")

    wave_flux_spray_qty = Column(Float, nullable=True, comment="波峰焊助焊剂喷涂量")
    wave_preheat_temp_min_c = Column(Float, nullable=True, comment="波峰焊预热最低温度 ℃")
    wave_preheat_temp_max_c = Column(Float, nullable=True, comment="波峰焊预热最高温度 ℃")
    wave_preheat_temp_avg_c = Column(Float, nullable=True, comment="波峰焊预热平均温度 ℃")
    wave_solder_temp_c = Column(Float, nullable=True, comment="波峰焊锡炉温度 ℃")
    wave_peak_height_1_mm = Column(Float, nullable=True, comment="波峰 1 高度")
    wave_peak_height_2_mm = Column(Float, nullable=True, comment="波峰 2 高度")
    wave_profile_json = Column(Text, nullable=True, comment="波峰焊参数 JSON")

    confidence = Column(Float, default=0.75, comment="画像可信度")
    raw_json = Column(Text, nullable=True, comment="原始解析结果 JSON")
    note = Column(String(500), nullable=True, comment="解析说明")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        for key in (
            "meter_ids_json",
            "normalized_meter_ids_json",
            "reflow_profile_json",
            "online_current_by_meter_json",
            "wave_profile_json",
            "raw_json",
        ):
            out_key = key[:-5] if key.endswith("_json") else key
            try:
                d[out_key] = json.loads(d[key]) if d.get(key) else []
            except Exception:
                d[out_key] = d.get(key)
        return d


class ReliabilitySampleProductionRecord(Base):
    """X202201001 可靠性样机生产修调后多次测试明细。"""
    __tablename__ = "reliability_sample_production_record"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sample_id = Column(String(50), nullable=False, index=True, comment="电子批号")
    source_file = Column(String(255), nullable=True)
    test_item = Column(String(50), index=True, comment="测试项目")
    status = Column(String(50), nullable=True, comment="状态/方向")
    phase_sequence = Column(String(50), nullable=True, comment="相序")
    phase_element = Column(String(50), nullable=True, comment="分合元")
    current_point = Column(String(50), nullable=True, comment="电流点")
    power_factor = Column(String(50), nullable=True, comment="功率因数")
    error_value = Column(Float, nullable=True, comment="误差值")
    actual_value = Column(String(100), nullable=True, comment="实际值")
    operator = Column(String(50), nullable=True, comment="操作者")
    operation_time = Column(DateTime, nullable=True, index=True, comment="操作时间")
    operation_result = Column(String(20), nullable=True, comment="操作结果")
    raw_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class ReliabilitySampleAfterRecord(Base):
    """X202201001 可靠性样机试验后误差分析明细。"""
    __tablename__ = "reliability_sample_after_record"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sample_id = Column(String(50), nullable=False, index=True, comment="电子批号")
    source_file = Column(String(255), nullable=True)
    test_item = Column(String(50), index=True, comment="测试项目")
    current_point = Column(String(50), nullable=True, comment="电流点/检验点")
    power_factor = Column(String(50), nullable=True, comment="功率因数")
    pre_values_json = Column(Text, nullable=True, comment="车间试验前多次测试值 JSON")
    pre_avg = Column(Float, nullable=True, comment="试验前平均值")
    pre_min = Column(Float, nullable=True, comment="试验前最小值")
    pre_max = Column(Float, nullable=True, comment="试验前最大值")
    after_value = Column(Float, nullable=True, comment="试验后测试值")
    after_min_deviation = Column(Float, nullable=True, comment="试验后相对试验前最小偏差")
    after_max_deviation = Column(Float, nullable=True, comment="试验后相对试验前最大偏差")
    after_avg_deviation = Column(Float, nullable=True, comment="试验后相对试验前平均偏差")
    raw_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class ReliabilitySampleProfile(Base):
    """按电子批号聚合的可靠性样机生产-试验后质量画像。"""
    __tablename__ = "reliability_sample_profile"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sample_id = Column(String(50), nullable=False, unique=True, index=True, comment="电子批号")
    batch_key = Column(String(100), default="X202201001", index=True)
    source = Column(String(50), default="production_after_reliability")
    production_record_count = Column(Integer, default=0)
    production_error_count = Column(Integer, default=0)
    production_error_abs_max = Column(Float, default=0.0)
    production_error_abs_avg = Column(Float, default=0.0)
    production_error_abs_std = Column(Float, default=0.0)
    production_bad_count = Column(Integer, default=0)
    production_test_date_count = Column(Integer, default=0)
    after_error_count = Column(Integer, default=0)
    after_error_abs_max = Column(Float, default=0.0)
    after_error_abs_avg = Column(Float, default=0.0)
    after_deviation_abs_max = Column(Float, default=0.0)
    after_deviation_abs_avg = Column(Float, default=0.0)
    timing_after_abs = Column(Float, default=0.0)
    reliability_risk_score = Column(Float, default=0.0)
    quality_label = Column(Integer, default=0)
    note = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

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


class MeterIdentity(Base):
    """电表身份映射表：连接老化/质量链路编号与生产条形码。"""
    __tablename__ = "meter_identity"

    id = Column(Integer, primary_key=True, autoincrement=True)
    meter_no = Column(Integer, unique=True, nullable=True, index=True, comment="老化/质量链路电表编号")
    barcode = Column(String(50), unique=True, nullable=False, index=True, comment="生产条形码")
    normalized_barcode = Column(String(50), nullable=False, index=True, comment="标准化条码")
    order_no = Column(String(50), nullable=True, index=True, comment="订单号")
    batch_no = Column(String(50), nullable=True, index=True, comment="批次号")
    product_model = Column(String(100), nullable=True, comment="产品型号")
    source = Column(String(50), default="unknown", comment="映射来源")
    confidence = Column(Float, default=0.0, comment="映射置信度 0-1")
    note = Column(String(500), nullable=True, comment="映射说明")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class QualityFeatureSnapshot(Base):
    """融合后的质量风险训练快照：固定特征口径，便于论文实验复现。"""
    __tablename__ = "quality_feature_snapshot"

    id = Column(Integer, primary_key=True, autoincrement=True)
    meter_no = Column(Integer, nullable=False, unique=True, index=True)
    barcode = Column(String(50), nullable=True, index=True)
    normalized_barcode = Column(String(50), nullable=True, index=True)
    order_no = Column(String(50), nullable=True, index=True)
    batch_key = Column(String(100), nullable=True, index=True)
    source = Column(String(50), default="linked", comment="linked/synthetic_demo/unlinked")
    confidence = Column(Float, default=0.0)

    crystal_ppm = Column(Float)
    ct_ratio_err = Column(Float)
    clock_accuracy_sd = Column(Float)
    factory_fault = Column(Boolean, default=False)
    initial_error_max = Column(Float)
    final_error_max = Column(Float)
    error_degradation_rate = Column(Float)
    clock_drift_rate = Column(Float)
    clock_daily_error_max = Column(Float)
    error_variability_max = Column(Float)
    error_consistency_max = Column(Float)
    energy_balance_ratio = Column(Float)
    energy_peak_ratio = Column(Float)
    rs485_total_errors = Column(Integer, default=0)

    prod_ccd_retry = Column(Float, default=0.0)
    prod_func_retry = Column(Float, default=0.0)
    prod_reverify_retry = Column(Float, default=0.0)
    prod_total_retry = Column(Float, default=0.0)
    prod_smt_aoi_defect = Column(Float, default=0.0)
    prod_wave_aoi_defect = Column(Float, default=0.0)
    prod_repaired = Column(Float, default=0.0)
    prod_repair_count = Column(Float, default=0.0)
    prod_leak_current_max = Column(Float, default=0.0)
    prod_quality_risk_score = Column(Float, default=0.0)

    quality_label = Column(Integer, default=0)
    quality_reason = Column(String(500))
    feature_json = Column(Text, nullable=True, comment="按基础快照特征顺序保存的24维特征数组")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class BatchQualityProfile(Base):
    """批次抽检质量画像：老化实验是批次/抽检粒度，不默认对应到每块生产表。"""
    __tablename__ = "batch_quality_profile"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_key = Column(String(100), nullable=False, unique=True, index=True, comment="批次标识，优先 order_no/batch_no")
    order_no = Column(String(50), nullable=True, index=True)
    batch_no = Column(String(50), nullable=True, index=True)
    source = Column(String(50), default="sampled_aging", comment="sampled_aging/sampled_batch_demo")
    confidence = Column(Float, default=0.0)

    sample_count = Column(Integer, default=0)
    high_risk_count = Column(Integer, default=0)
    high_risk_rate = Column(Float, default=0.0)
    avg_final_error_max = Column(Float, default=0.0)
    max_final_error_max = Column(Float, default=0.0)
    avg_error_degradation_rate = Column(Float, default=0.0)
    max_clock_drift_rate = Column(Float, default=0.0)
    max_rs485_total_errors = Column(Integer, default=0)
    sampled_meter_nos_json = Column(Text, nullable=True)
    note = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        d["sampled_meter_nos"] = json.loads(self.sampled_meter_nos_json) if self.sampled_meter_nos_json else []
        return d


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

