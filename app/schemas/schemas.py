"""Pydantic 请求/响应模型"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from datetime import datetime


class CrystalDataIn(BaseModel):
    batch_no: str
    sample_no: int
    model: Optional[str] = "HC-49SMD"
    center_freq_hz: Optional[float] = 3579500.0
    load_cap_pf: Optional[float] = 20.0
    freq_error_ppm: float
    is_qualified: bool = True


class CTDataIn(BaseModel):
    batch_no: str
    sample_no: int
    model: Optional[str] = "HDCT3"
    ratio_error_max: float
    ratio_error_range: Optional[float] = None
    angle_error_max: Optional[float] = None
    dc_resistance: Optional[float] = None
    is_qualified: bool = True


class FactoryTestIn(BaseModel):
    meter_id: str
    crystal_batch: str
    ct_batch: str
    crystal_ppm: float
    ct_ratio_err: float
    fault_type: Optional[str] = None
    meter_constant_err: Optional[float] = None
    clock_accuracy_sd: Optional[float] = None
    rs485_ok: bool = True


class PredictRequest(BaseModel):
    crystal_ppm: float
    ct_ratio_err: float
    clock_accuracy_sd: Optional[float] = None
    factory_fault: bool = False
    initial_error_max: Optional[float] = None


class PredictResponse(BaseModel):
    risk_level: str          # 低/中/高
    risk_prob: float
    risk_label: int
    shap_values: Optional[Dict[str, float]] = None
    root_cause_text: Optional[str] = None
    model_version: Optional[str] = None


class DegradationResponse(BaseModel):
    meter_no: int
    time_points: List[str]
    error_series: List[float]
    clock_series: List[int]
    degradation_rate: float
    rul_hours: Optional[float]
    rul_label: str
    trend_type: str


class ImportResult(BaseModel):
    success: bool
    message: str
    records_imported: int
    errors: List[str] = []


class DashboardStats(BaseModel):
    total_meters: int
    high_risk_count: int
    normal_count: int
    avg_crystal_ppm: float
    avg_ct_ratio: float
    latest_import: Optional[str]
