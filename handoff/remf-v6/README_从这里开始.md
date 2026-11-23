# 电表质量追溯系统交接

2026 年 11 月 23 日 · 衡序 REMF V6 展示版

## 第一次上手

1. 将整包解压到可写的本地目录。安装 Windows 64 位 Python 3.11（安装器勾选 Python Launcher 或加入 PATH）。Python 安装器由接收方准备。
2. 双击根目录的 `01-install.cmd`。它在本目录创建 `.venv`，使用随包的 Windows 安装包离线安装 Web、CPU 推理和测试依赖。看到 `Ready:` 后进行下一步。安装了多个 Python 时，可在 PowerShell 执行 `powershell -ExecutionPolicy Bypass -File .\env\install.ps1 -Python "C:\实际路径\python.exe"`。
3. 双击 `02-start.cmd`，保持窗口打开，浏览器访问 `http://127.0.0.1:8092`。首次启动会把随包数据导入本地 SQLite。检查页面显示模型已就绪、10,000 台电表、1,000 台异常。端口已被占用时，在 PowerShell 输入 `$env:REMF_PORT=8093` 后运行 `.\02-start.cmd` 并访问对应端口。按 Ctrl+C 停止服务。
4. 打开“检测与根因分析”，搜索 `SIM6-U00037`，核对 200 次通信中失败 18 次，点“运行模型推理”。首位候选应为 RS485 连接焊接异常，约 86.2%。切换“检测前风险试算”，加载同一编号，修改“RS485 来料裕量代理”从 0.622 到 0.800，重新计算，通信风险约从 2.18% 变为 2.11%。
5. 要复测，在 PowerShell 运行 `.\03-test.cmd`。结果写入 `system/remf_acceptance/`，测试在隔离库运行。通过的标志为 23 个集成用例通过、补充检查 JSON 包含 B01 至 B06 六项通过。

## 接手时按任务找文件

| 想做什么 | 入口 |
| --- | --- |
| 修改页面、交互或字段 | `system/remf_app/static/` 为完整前端。后端入口 `system/remf_main.py`，接口在 `web.py`，字段和检测限值在 `schema.py`，Excel/CSV 解析在 `ingest.py`，SQLite 存储在 `store.py`。这几个 Python 模块均在 `system/remf_app/`。 |
| 调试模型调用 | `inference.py` 启动独立 Python 工作进程，`gpu_worker.py` 校验并加载权重、标准化输入、集成和校准概率。两者位于 `system/remf_app/`。 |
| 找部署算法和权重 | `system/remf_app/model_bundle/` 内的 `models.py` 为 REMF，`risk_model.py` 为检测前 MLP。每个模型各有 17、43、101 三个种子的 .pt 权重，并配套预处理、校准或哈希文件。发布时按整套管理。 |
| 核对样本、接口和训练输入 | `data/` 有原版 V6 Excel、中文总表、制造与检测明细、复测维修记录、数据字典和质量报告。首页按模拟来源建库。标有“仅审计”的数据仅用于仿真复核。 |
| 读取实验结果或重训 | `research/` 有原 GPU 训练代码、传统基线和消融、第三方源码及许可证、检查点、逐台结果及风险试算实验记录。见同目录《训练与指标.md》。 |
| 查测试或演示 | `tests_history/` 为原始证据和界面截图，`system/tests/` 为可重跑脚本，`system/remf_acceptance/` 为交接副本的复测结果。`docs/` 内有员工操作指南、功能测试报告和演示说明。 |

## 企业试录与状态保存

默认状态库为 `system/.remf_state/workbench.sqlite3`。交接包从 V6 合成样本重新建库，旧展示机的临时试算、手工新增记录仅在历史手册和截图中出现。备份时先停止本包服务，再复制完整状态目录。恢复时用独立目录启动并核对编号与历史。

要在空库试录，先创建两个独立的空目录，在 PowerShell 设置 `$env:REMF_SOURCE='C:\meter_trial\empty_source'` 及 `$env:REMF_STATE_DIR='C:\meter_trial\state'`，再从同一窗口执行 `.\02-start.cmd`。此时由页面导入实际表单或手工录入。企业试录与模拟演示使用各自的状态目录。

本包的安装和启停以本页为准。历史 Word 手册中的双环境、后台启动及 PID 操作对应原开发机部署，界面操作步骤仍适用。当前交接范围为端口 8092 对应的电表 REMF 应用。历史仓库中的其他产品服务另行维护。
