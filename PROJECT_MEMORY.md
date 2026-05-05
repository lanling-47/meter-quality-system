# 智能电表全周期质量管控系统 — 项目工作记忆文档

> **用途**：新 Agent / 新工作空间接手时的完整上下文说明
> **最后更新**：2026-04-23
> **GitHub**：https://github.com/lanling-47/meter-quality-system（分支：`new`）
> **本地路径**：`/Users/hujie07/Downloads/abiye/meter_quality_system`
> **运行命令**：`cd meter_quality_system && uvicorn main:app --port 8090 --reload`
> **访问地址**：http://localhost:8090

---

## 一、项目背景与目标

这是一个浙大专业硕士的产学研合作项目，与某电表公司合作。

**核心目标**：利用电表全生命周期数据（来料检测 + 加速老化试验 + 生产过程），在出厂前预测哪些电表有质量风险，并提供根因追溯、退化分析、用电异常检测等功能。

**部署场景**：在电表公司内部网络服务器上线，生产线 MES 系统定时导出 Excel 文件到指定目录，系统自动采集并分析。

---

## 二、目录结构

```
/Users/hujie07/Downloads/abiye/
├── meter_quality_system/          ← 主项目代码
│   ├── main.py                    ← FastAPI 入口，注册所有路由
│   ├── config.py                  ← 全局配置（路径/阈值/文件监控参数）
│   ├── database.py                ← SQLAlchemy SessionLocal + Base
│   ├── requirements.txt
│   ├── meter_quality.db           ← SQLite 数据库（20MB）
│   ├── ml_models/                 ← 训练好的模型 pkl 文件
│   ├── app/
│   │   ├── models/db_models.py    ← 13张数据库表的 ORM 定义
│   │   ├── routers/               ← 11个路由文件
│   │   └── services/              ← 核心算法服务（12个文件）
│   ├── templates/                 ← Jinja2 HTML 模板（12个页面）
│   ├── static/                    ← Bootstrap CSS 等静态文件
│   ├── data/raw/                  ← 原始 Excel 数据文件（3个历史数据）
│   ├── 智能电表全周期质量管控系统_技术汇报.md   ← 完整技术汇报文档
│   └── 组会汇报_2026.04.20.md    ← 简洁组会汇报
│
├── 电表生产数据/                  ← 真实生产线 Excel 数据（已解析入库）
│   ├── CCD数据.xlsx               (1.3MB)
│   ├── 功能测试数据.xlsx           (1.3MB)
│   ├── 复校数据.xlsx               (1.3MB)
│   ├── 耐压测试数据.xlsx           (1.45MB)
│   ├── 单板测试数据.xlsx           (45MB)
│   ├── 维修数据.xlsx               (21KB)
│   ├── 贴片AOI不合格数据.xls       (31KB)
│   ├── 贴片AOI的故障.xls           (23KB)
│   └── 选择性波峰焊AOI.xls         (32KB)
│
├── watch_dir/                     ← 文件监控目录（生产线实时采集）
├── 晶振和互感器10个样品的检验数据26.3.11.xlsx
├── 单相表-20版正面布局的耐久性（可靠性）试验数据26.3.6.xlsx
└── 样品出厂检测和数字化电路板生产过程数据26.xlsx
```

---

## 三、数据库现状（`meter_quality.db`）

### 各表记录数

| 表名 | 记录数 | 说明 |
|------|--------|------|
| `crystal_data` | 20 | 晶振来料（2批次×10样品） |
| `ct_data` | 10 | 互感器来料（1批次×10样品） |
| `factory_test` | 0 | 出厂检测（⚠️ 未入库） |
| `aging_test` | 180 | 1000h加速老化（~37只 × 7时间点） |
| `quality_chain` | **231** | 训练集核心表（31原始 + 200 SMOTE扩增） |
| `production_inspection` | **100,030** | 生产工序检测（CCD/功能测试/复校/耐压各~25000条） |
| `meter_quality_trace` | **24,210** | 生产质量追溯汇总 |
| `aoi_defect` | 199 | AOI缺陷记录 |
| `repair_record` | 165 | 维修记录 |
| `board_test_detail` | 780 | 单板测试失败记录 |
| `model_version` | 10 | AI模型版本记录 |
| `data_ingestion_log` | 0 | 实时采集日志（监控服务未运行） |
| `quality_label_review` | 0 | 人工审核队列（无待审数据） |

### ⚠️ 关键数据问题：两表未打通

**最大问题**：`quality_chain`（训练集，按 `meter_no` 索引，196~232号）和 `meter_quality_trace`（生产数据，按 `barcode` 条形码索引）**没有建立关联**。

- `quality_chain` 没有生产过程特征列（`prod_ccd_retry` 等列**在数据库中不存在**，只在 `predict_service.py` 的特征名定义中存在）
- 导致训练时生产特征全部为默认值 0
- `meter_quality_trace` 有 24,210 条真实生产数据但完全没被用于训练

**解决方向**：建立 `barcode ↔ meter_no` 映射，或修改训练流程直接使用 `meter_quality_trace` 作为训练基础。

### 生产数据 Excel 关键字段（已确认）

```
CCD数据.xlsx:
  列: ATXM(条形码), DDH（订单号）, BWH（表位号）, JYCS（测试次数）, DQGX（当前工序）, JYJL（结论）, JYRQ（时间）, JYZZ（设备编号）

功能测试数据.xlsx:
  列: ATXM(条形码), DDH（订单号）, BWH（表位号）, JYCS（测试次数）, DQGX（当前工序）, JYJL（结论）, JYRQ（时间）, BZ（不合格项备注）

维修数据.xlsx:
  列: TXM（条形码）, GZXXDM（故障现象代码）, GZXXMC（故障现象名称）, GZYYDM（故障原因代码）, GZYYMC（故障原因名称）, GZDD（故障地点）, GZJLSJ（故障记录时间）, GZGX（故障工序）

贴片AOI不合格数据.xls:
  列: SBBH（设备编号）, TXM（条形码）, GZD（故障点）, GZMS（故障描述）, DQGX（当前工序）, SJ（时间）
```

---

## 四、算法架构

### 4.1 特征工程（24维，但生产特征当前未实装）

```python
# 文件：app/services/predict_service.py  FEATURE_NAMES 变量
来料特征（4维）:
  crystal_ppm, ct_ratio_err, clock_accuracy_sd, factory_fault

老化特征（10维）:
  initial_error_max, final_error_max, error_degradation_rate, clock_drift_rate,
  clock_daily_error_max, error_variability_max, error_consistency_max,
  energy_balance_ratio, energy_peak_ratio, rs485_total_errors

生产过程特征（10维）— ⚠️ 当前填充为0：
  prod_ccd_retry, prod_func_retry, prod_reverify_retry, prod_total_retry,
  prod_smt_aoi_defect, prod_wave_aoi_defect, prod_repaired, prod_repair_count,
  prod_leak_current_max, prod_quality_risk_score
```

### 4.2 模型层次

```
第1层：规则引擎（基线，无需训练）
第2层：TabNet（AAAI 2021）— 当前激活模型
        ↳ 文件: app/services/tabnet_service.py
        ↳ 参数: n_d=16, n_a=16, n_steps=3, mask_type=sparsemax, max_epochs=200
        ↳ 当前 AUC: 0.999（⚠️ 因生产特征为0，数据过拟合，不可信）
第3层：DQN 在线学习层
        ↳ 文件: app/services/drl_agent.py
        ↳ 纯 NumPy 实现，无需 PyTorch
第4层：SHAP 根因追溯
        ↳ 文件: app/services/shap_service.py
```

### 4.3 消融实验框架

```
文件: app/services/ablation_service.py
设计: 4特征组 × 5算法 = 20组实验，5折CV，Mean ± Std
特征组: A(4维来料) / B(14维来料+老化) / C(24维全特征) / D(10维仅生产)
算法: 规则引擎 / RandomForest / XGBoost / LightGBM / TabNet
⚠️ 当前还未运行过消融实验（因生产特征为0，暂时无意义）
```

---

## 五、系统页面（9个）

| URL | 功能 | 文件 |
|-----|------|------|
| `/dashboard` | 总览仪表盘 KPI | routers/dashboard.py |
| `/import` | 数据导入 + 模型训练（SSE进度） | routers/data_import.py |
| `/predict` | 批量预测 + 单表单预测 | routers/risk_predict.py |
| `/rootcause` | SHAP 根因追溯 | routers/root_cause.py |
| `/degradation` | 退化曲线分析 | routers/degradation.py |
| `/anomaly` | 多维用电异常检测 | routers/anomaly_detect.py |
| `/production` | 生产追溯 + 维修分析 | routers/production.py |
| `/ablation` | 消融实验（4×5矩阵，SSE流式） | routers/ablation.py |
| `/data-review` | 数据采集审核 + 训练触发 | routers/data_review.py |

---

## 六、已完成的工作

### 阶段1：基础系统
- ✅ FastAPI + SQLAlchemy + SQLite 基础架构
- ✅ ETL 解析3个历史 Excel 文件入库（晶振/互感器/老化）
- ✅ 14维特征 + LightGBM 训练
- ✅ SHAP 根因追溯
- ✅ 退化曲线、用电异常检测、仪表盘

### 阶段2：生产数据融合
- ✅ 解析 `电表生产数据/` 下10个Excel文件入库
- ✅ `production_inspection`（10万条）、`meter_quality_trace`（2.4万条）
- ✅ 生产追溯页面（/production）
- ✅ 维修分析页面（/repair-analysis）
- ⚠️ 生产特征未打通到 `quality_chain` 训练集（最大遗留问题）

### 阶段3：TabNet + 消融实验
- ✅ 安装 pytorch-tabnet，更新 requirements.txt
- ✅ tabnet_service.py（训练封装、5折CV、注意力提取）
- ✅ ablation_service.py（4×5消融框架）
- ✅ ablation.html（4图可视化 + 学术表格）
- ✅ predict_service.py 改造（TabNet默认 + LightGBM降级）
- ✅ load_active_model 返回值扩展（model, feat_names, version, scaler, model_type）

### 阶段4：实时采集 + 在线更新
- ✅ file_watcher_service.py（5分钟轮询监控目录）
- ✅ notify.py（SSE通知 + 手动上传接口）
- ✅ retrain.py（增量训练 + 全量重训练路由）
- ✅ data_review.py + data_review.html（审核队列 + 训练触发）
- ✅ config.py 扩展（WATCH_DIR、AUTO_RETRAIN_THRESHOLD等）
- ✅ 新增2张DB表（data_ingestion_log、quality_label_review）

### 文档
- ✅ 技术汇报文档（`智能电表全周期质量管控系统_技术汇报.md`）
- ✅ 组会汇报（`组会汇报_2026.04.20.md`）

---

## 七、当前最优先的未解决问题

### 🔴 P0：生产过程特征未打通到训练集

**现状**：
- `quality_chain` 没有生产过程列（prod_xxx），24维特征在训练时有10维全为0
- `meter_quality_trace` 有24,210条真实数据但完全未被用于训练
- 当前 AUC=0.999 是因为样本量极少（231条，200条是SMOTE扩增），存在严重过拟合

**解决方案**：两种路线

**方案A（推荐，如果两表可关联）**：
```python
# 查找 quality_chain.meter_no 是否对应某个条形码规律
# 例如 meter_no=196 → barcode 前缀 A472...
# 建立映射后 JOIN 两表，给 quality_chain 补填生产过程特征
```

**方案B（如果无法关联，以生产数据为主）**：
```python
# 将 meter_quality_trace 作为训练基础（2.4万条）
# 用自动打标策略（多次返修/AOI缺陷/重测次数阈值）生成质量标签
# 构建新的特征矩阵（纯生产过程特征，放弃老化试验特征）
```

### 🔴 P0：factory_test 表为空

`factory_test` 表有 0 条记录，但 `样品出厂检测和数字化电路板生产过程数据26.xlsx` 文件存在，需要检查 ETL 是否解析失败。

### 🟡 P1：quality_chain 原始样本只有31条

真实（非SMOTE）样本极少，无法验证模型泛化性。等生产数据打通后，`meter_quality_trace` 的2.4万条将成为主要训练集。

### 🟡 P1：消融实验尚未运行

在生产特征打通之前，运行消融实验没有意义（生产特征全为0，Group-C与Group-B结果相同）。

---

## 八、关键代码文件速查

| 问题/需求 | 关键文件 | 关键函数/类 |
|---------|---------|------------|
| 训练新模型 | `app/services/predict_service.py` | `train_model(db, model_type="tabnet")` |
| TabNet 训练细节 | `app/services/tabnet_service.py` | `train_tabnet(X, y, feature_names)` |
| 特征名定义 | `app/services/predict_service.py` | `FEATURE_NAMES`（第24-55行） |
| 从DB构建特征矩阵 | `app/services/predict_service.py` | `_build_feature_matrix(db)` |
| 生产ETL解析 | `app/services/production_etl_service.py` | `parse_inspection()`, `parse_aoi_defects()` |
| 历史ETL解析 | `app/services/etl_service.py` | `auto_import_on_startup()` |
| 根因分析 | `app/services/shap_service.py` | `explain_meter(meter_no, db)` |
| 退化分析 | `app/services/degradation_service.py` | `get_meter_degradation(meter_no, db)` |
| 异常检测 | `app/services/anomaly_service.py` | `get_power_data(db)` |
| 文件监控 | `app/services/file_watcher_service.py` | `FileWatcherService`, `get_watcher()` |
| DQN在线学习 | `app/services/drl_agent.py` | `DQNAgent`, `learn_from_feedback()` |
| 消融实验 | `app/services/ablation_service.py` | `run_ablation_experiment(db, emit_fn)` |
| 数据库表定义 | `app/models/db_models.py` | 全部13张表 |
| 路由注册 | `main.py` | `app.include_router(...)` |
| 全局配置 | `config.py` | 路径/阈值/监控参数 |

---

## 九、技术栈

| 层次 | 技术 | 版本要求 |
|------|------|---------|
| Web 框架 | FastAPI + uvicorn | ≥0.100 |
| ORM | SQLAlchemy | ≥2.0 |
| 数据库 | SQLite | — |
| 模板 | Jinja2 | — |
| 机器学习 | scikit-learn | ≥1.3 |
| 深度学习 | pytorch-tabnet | ≥4.1 |
| 梯度提升 | LightGBM + XGBoost | — |
| 过采样 | imbalanced-learn | — |
| 可解释 AI | SHAP | — |
| 前端 | Bootstrap 5 + ECharts 5 | CDN |
| 容器 | Docker + Docker Compose | — |
| Python 环境 | /opt/miniconda3/bin/python3 | 3.13 |

---

## 十、下一步工作建议（优先级排序）

### 🔴 最高优先（影响论文核心结论）

1. **打通生产特征到训练集**
   - 检查 `quality_chain.meter_no`（196~232）与 `meter_quality_trace.barcode` 的对应关系
   - 若能关联：JOIN补填生产特征到 `quality_chain`
   - 若不能关联：以 `meter_quality_trace` 为训练基础，设计自动打标规则

2. **修复 factory_test 空表**
   - 检查 `app/services/etl_service.py` 中对 `样品出厂检测和数字化电路板生产过程数据26.xlsx` 的解析逻辑

3. **运行消融实验，获得真实对比结果**
   - 生产特征打通后，访问 `/ablation` 页面运行实验
   - 保存结果截图用于论文 Table

### 🟡 中优先（系统完整性）

4. **用真实大样本重新训练模型**
   - 当前231条（200条SMOTE）→ 目标用2.4万条 `meter_quality_trace`
   - 设计合理的质量标签生成策略（多次返修 / 多次重测 / AOI缺陷阈值）

5. **SHAP 适配 TabNet**
   - 当前 `shap_service.py` 仍用 `shap.TreeExplainer`（LightGBM专用）
   - TabNet 模型需改用 `shap.DeepExplainer` 或 TabNet 自带的 `explain()` 方法

6. **前端：训练结果展示补充**
   - TabNet 的注意力权重矩阵可视化
   - 消融实验完成后的图表自动截图功能

### 🟢 低优先（部署完善）

7. 将 SQLite 迁移至 PostgreSQL（支持并发）
8. 添加用户认证（JWT）
9. 编写用户操作手册

---

## 十一、论文写作进度提示

| 章节 | 状态 | 关键依赖 |
|------|------|---------|
| 第一章 引言 | 可写 | 已有开题报告基础 |
| 第二章 相关工作 | 可写 | 已有文献 PDF |
| **第三章 方法** | 可写框架，待补实验数字 | 需消融实验结果 |
| **第四章 实验** | ⚠️ 阻塞 | 需要生产特征打通 + 消融实验运行 |
| 第五章 系统实现 | 可写 | 系统已完成 |
| 第六章 总结 | 最后写 | — |

---

*文档生成时间：2026-04-23 | 工作空间：/Users/hujie07/Downloads/abiye*
