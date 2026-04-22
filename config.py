"""全局配置（支持通过环境变量覆盖路径，用于 Docker 部署）"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ── 路径配置（Docker 中通过环境变量覆盖，使用持久化存储目录）────────────────
DATA_DIR = BASE_DIR / "data" / "raw"

# 数据库路径：优先读取环境变量 DB_PATH，默认放在项目根目录
DB_PATH = Path(os.environ.get("DB_PATH", str(BASE_DIR / "meter_quality.db")))

# ML 模型目录：优先读取环境变量 ML_MODELS_DIR
ML_MODELS_DIR = Path(os.environ.get("ML_MODELS_DIR", str(BASE_DIR / "ml_models")))

# 确保目录存在
ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# 数据文件名
CRYSTAL_FILE = "晶振和互感器10个样品的检验数据26.3.11.xlsx"
FACTORY_FILE = "样品出厂检测和数字化电路板生产过程数据26.xlsx"
AGING_FILE = "单相表-20版正面布局的耐久性（可靠性）试验数据26.3.6.xlsx"

# 生产过程数据目录（相对项目上级目录）
PRODUCTION_DATA_DIR = BASE_DIR.parent / "电表生产数据"

# ── 文件监控配置（生产线实时采集）──────────────────────────────────────────
# 轮询监控目录：生产线 MES 系统定时导出 Excel/CSV 文件到此目录
WATCH_DIR = Path(os.environ.get("WATCH_DIR", str(BASE_DIR.parent / "watch_dir")))
# 轮询间隔（秒）：默认 5 分钟
WATCH_INTERVAL_SEC = int(os.environ.get("WATCH_INTERVAL_SEC", 300))
# 新增多少条数据后，前端显示"建议重训练"提示
AUTO_RETRAIN_THRESHOLD = int(os.environ.get("AUTO_RETRAIN_THRESHOLD", 50))
# 默认重训练使用的模型类型
RETRAIN_MODEL_TYPE = os.environ.get("RETRAIN_MODEL_TYPE", "tabnet")

# 确保监控目录存在
WATCH_DIR.mkdir(parents=True, exist_ok=True)

# 数据库 URL
DATABASE_URL = f"sqlite:///{DB_PATH}"

# ── 质量阈值 ─────────────────────────────────────────────────────────────────
# 加速老化试验（85℃/85%RH）条件下，参照 GB/T 17215.321 1级表：基本误差限 ±1%
ERROR_THRESHOLD = 0.8        # 基本误差高风险阈值 %（老化末期 >0.8% 预警）
CLOCK_THRESHOLD = 10         # 时钟偏差阈值 s（1000h末）
CRYSTAL_PPM_LIMIT = 20       # 晶振频差限值 ppm
CT_RATIO_LIMIT = 1.0         # 互感器比差限值 %
CLOCK_DAILY_THRESHOLD = 0.3  # 日误差高风险阈值 s/d
CONSISTENCY_THRESHOLD = 0.12 # 误差一致性偏差阈值 %

# ── ML 模型参数 ───────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.3

# 老化试验时间点 → 累计小时数映射
AGING_TIME_MAP = {
    "6.18": 0, "6.19": 0,
    "6.29": 264,
    "7.8": 480,
    "7.16": 672,
    "7.25": 864,
    "8.5": 1128, "8.6": 1128,
}
