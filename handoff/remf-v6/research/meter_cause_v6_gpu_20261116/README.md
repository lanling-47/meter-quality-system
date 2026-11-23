# v6 电表根因分析 GPU 实验

本目录是 2026 年 11 月 16 日使用新 v6 半合成样本重新训练、验证和测试的独立实验。只做首次检测异常后的候选根因辅助分析，不训练检测前故障预测。

## 阅读顺序

1. `电表根因模型训练与测试报告.docx`：图文测试报告，黑白表格。
2. `模型训练与测试报告.md`：同一结果的可检索文本。
3. `三种子集成测试指标.csv`、`test_metrics_by_seed.csv`：集成及各随机种子结果。
4. `测试逐台候选原因.csv`、`选定模型全部故障候选.csv`：逐台候选排名与分值。
5. `验收结果.json`、`接口验收.json`：独立数值复核及推理接口验证。

## 数据和边界

源样本位于 `D:/biye/outputs/meter_samples_v6_20261116`；完整源数据另见同级 `meter_samples_v6_20261116_完整交付.zip`。模型包不重复携带原始 Excel 和审计真值。

10000 台中有 1000 台首次检测故障。单标签监督仅用 292 台单因模拟证据病例，训练 187、验证 54、测试 51。训练集无标签前置观测用于拟合标准化，符合限制的首轮维修响应仅在训练中作辅助监督。单因、待确认、多因及未确认全部是合成记录，真实企业确认标签为零。

`data.py`只读取协议白名单字段。未来检查、复测、维修结论不能进入推理输入。`仅审计`文件中的隐含真值不参与训练。v6 留出集已有前期探索，不能称全新的外部盲测。

## 环境和复现

本机 Conda Python：`D:/Users/jay/anaconda3/envs/yolov9/python.exe`，PyTorch 2.5.1+cu121，GPU RTX 4060。科学计算依赖独立放在 `D:/biye/outputs/meter_v6_gpu_dependencies`，不改动原 Conda 科学计算包。实际加载版本及路径见 `environment.json`。`requirements-freeze.txt`同时记录既有环境包，可能含原环境依赖，实际加载路径以 environment 为准。

在本机 PowerShell 执行：

```powershell
./run_training.ps1
$env:PYTHONPATH='D:/biye/outputs/meter_v6_gpu_dependencies'
& 'D:/Users/jay/anaconda3/envs/yolov9/python.exe' ./verify_results.py
& 'D:/Users/jay/anaconda3/envs/yolov9/python.exe' ./check_inference.py
& 'D:/Users/jay/anaconda3/envs/yolov9/python.exe' ./report.py
```

重训练会覆盖本目录已有输出；复现时建议复制实验目录并修改 protocol.json 的源数据路径，再运行。新机器可创建 Python 3.9 Conda 环境，安装 PyTorch 2.5.1 CUDA 12.1 wheel 以及 requirements-scientific.txt，修改 PowerShell 中解释器和依赖目录路径。GPU 训练强制使用 CUDA，不会静默回退 CPU；传统基线使用 CPU。

训练有 24 次神经模型运行、4 个传统基线。每次神经运行只看验证集选轮次；固定 selection_frozen.json 后运行 evaluate.py。不得依据测试分数调整本轮协议并继续称其为独立测试。

## 独立 CSV 推理

```powershell
& 'D:/Users/jay/anaconda3/envs/yolov9/python.exe' ./predict.py ./interface_check/input.csv ./prediction.csv
```

输入字段见 preprocessing.json 中 feature_columns 和 observation_columns，共 46 项，保留原始单位。首次检测通过产品不强制归因；缺字段、空值和非有限值会拒绝。`--model remf_v6`可指定研究模型。当前接口未部署到质量系统，输出不是企业真实因果概率。

## 开源实现

TabM 官方实现来自 https://github.com/yandex-research/tabm ，数值嵌入实现来自 https://github.com/yandex-research/rtdl-num-embeddings 。本次使用的源码及 Apache 2.0 许可证保存在 vendor 目录，交付 manifest 提供文件哈希。REMF 是本项目候选方法名称，不表示已证明首创或因果识别。

## 运行修正记录

训练前环境排查发现部分既有环境 DLL 加载受系统策略限制，最终使用可正常运行的 yolov9 GPU 环境与独立科学计算依赖，未关闭安全策略。首次启动训练发现 Windows NumPy 整数默认为 int32，已显式将分类目标转为 torch.long；该次未完成任何模型，修正后从头按同一协议完成训练。未根据测试结果改样本或调参。
