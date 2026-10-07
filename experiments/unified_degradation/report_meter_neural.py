"""Render the completed, frozen meter experiment without retraining."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    root = Path(__file__).parent / 'results/meter_neural_v1'
    metrics = pd.read_csv(root / 'metrics.csv').set_index('method')
    rows = pd.read_csv(root / 'predictions.csv')
    provenance = json.loads((root / 'provenance.json').read_text(encoding='utf-8'))
    history = json.loads((root / 'training_history.json').read_text(encoding='utf-8'))
    methods = ['persistence', 'population_increment', 'absolute_mlp_ensemble',
               'random_forest', 'ridge', 'meter_residual_ensemble']
    labels = ['保持值', '群体增量', '普通MLP集成', '随机森林', '岭回归', '改进残差网络集成']
    english = ['Persistence', 'Population increment', 'Absolute MLP', 'Random forest', 'Ridge', 'Residual neural net']
    table = '\n'.join(f'| {label} | {metrics.loc[method, "mae"]:.6f} | {metrics.loc[method, "rmse"]:.6f} |'
                      for method, label in zip(methods, labels))
    neural = metrics.loc['meter_residual_ensemble', 'mae']
    rf = metrics.loc['random_forest', 'mae']
    mlp = metrics.loc['absolute_mlp_ensemble', 'mae']
    rows['absolute_error'] = (rows.prediction - rows.target).abs()
    horizon = rows.groupby(['method', 'round']).absolute_error.mean().unstack()
    horizon.to_csv(root / 'horizon_metrics.csv')
    rows.groupby(['method', 'meter']).absolute_error.mean().to_csv(root / 'per_meter_metrics.csv')
    htable = '\n'.join(f'| {label} | ' + ' | '.join(f'{horizon.loc[method, h]:.6f}' for h in (4, 5, 6)) + ' |'
                       for method, label in zip(methods, labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].barh(english, [metrics.loc[m, 'mae'] for m in methods])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Test MAE (percentage points)')
    axes[0].set_title('9 held-out meters; 3 future rounds')
    for seed, records in history.items():
        axes[1].plot([r['epoch'] for r in records], [r['train_mae'] for r in records], label=seed)
    axes[1].set(xlabel='Epoch', ylabel='Training MAE (percentage points)', title='Final refits: fixed CV-selected budget')
    axes[1].legend(title='Seed')
    fig.savefig(root / 'comparison.png', dpi=160)
    fig.savefig(root / 'comparison.pdf')
    plt.close(fig)
    report = f'''# 电表深度学习模型迭代 v1

## 本轮结果

在历史固定划分的9只测试电表上，改进网络的MAE相对随机森林下降{100*(rf-neural)/rf:.2f}%，
相对普通MLP下降{100*(mlp-neural)/mlp:.2f}%。这不是稳定优越性的证明。
岭回归MAE更低；普通MLP的RMSE也低于改进网络，说明不同误差指标的排名不同。

| 方法 | MAE（百分点） | RMSE（百分点） |
|---|---:|---:|
{table}

按电表配对bootstrap的“新网络MAE减随机森林MAE”95%区间为
{provenance['paired_neural_minus_rf_ci95']}，跨过零。测试表仅9只，该区间也不包含训练样本选择的不确定性。

## 方法与实际选择

预测 = 第3轮观测 + 训练电表未来增量的逐通道中位数 + 神经网络个体增量修正。
输入为第3轮的10项误差，以及第1到2轮、第2到3轮的误差变化，共30维。
输出为第4、5、6轮各10项误差增量，共30维；没有使用测试表未来值构造特征。

采用两层Tanh编码器、Smooth-L1损失、增量收缩、AdamW权重衰减和梯度裁剪。
备选结构包含共同偏移、功率因数、电流档位三个共享因子以及自由通道残差支路。
四折交叉验证选择结果：{provenance['selected']['config']}。
最终选择的是自由输出残差网络，而不是三因子约束版；不能宣称结构先验已得到实证支持。
该网络属于多输出MLP的任务适配与训练改进，尚不足以单独证明新的算法原创性。

最终训练轮数固定为{provenance['final_epochs']}，由内层四折最佳轮数的中位数确定。
5个预先指定随机种子等权集成，未选择测试集表现最好的种子。
四折最佳轮数为{provenance['selected']['epochs']}，波动较大；这更提示小样本验证不稳定，不能简单归因于训练未收敛。
测试结果没有用于进一步调参；普通MLP也采用5种子集成和内层早停，RF与岭回归均按相同内层划分选参数。

## 数据与评估边界

直接读取原始Excel“基本误差”页，29只电表有完整6轮轨迹。
维持先导实验种子20260927及20只训练、9只测试的划分。训练集内做4折选择，所有归一化、增量中心、早停只使用对应训练/验证分区。
各模型都只读测试电表前3轮，预测后3轮。270个评分单元不等于270个独立样本。
无合成训练或测试观测，原始Excel、数据库及论文未被改写。

原表日期为：{provenance['dates']}。
因累计受应力时长尚未核实，本轮使用检测轮次，不沿用数据库未经核实的小时轴。
旧测试集历史上已被查看，本轮属于探索性结果；同批次留表验证不等于严格按日历时间进行的前瞻性部署验证。
本方法仍需训练表的完整轨迹，不能用于声称所有电表均只观测到第3轮时即可无标签训练。

## 分预测轮次

| 方法 | 第4轮MAE | 第5轮MAE | 第6轮MAE |
|---|---:|---:|---:|
{htable}

## 文件及复现

- `run_meter_neural.py`：读取、选择、训练、预测与配对bootstrap。
- `test_meter_neural.py`：历史输入隔离、训练统计量隔离、保存/复载一致性三项测试。
- `selection.json`：全部神经网络候选内层验证结果，保留未胜出的结构版本。
- `provenance.json`：原始文件SHA256、划分、所选参数及实验限制。
- `meter_residual_seed*.pt`：五个模型及训练归一化参数。
- `predictions.csv`、`metrics.csv`、`per_meter_metrics.csv`、`horizon_metrics.csv`：完整预测与评估。
- `comparison.png/pdf`、`training_history.json`：比较图与最终训练曲线。

在meter-quality-system目录执行：

```powershell
& 'D:/biye/meter-quality-system/.conda/python.exe' -u experiments/unified_degradation/run_meter_neural.py
& 'D:/biye/meter-quality-system/.conda/python.exe' -m unittest experiments.unified_degradation.test_meter_neural -v
& 'D:/biye/meter-quality-system/.conda/python.exe' experiments/unified_degradation/report_meter_neural.py
```

## 对论文的影响

该分支面向真实电表未来有符号误差，与原CP-UDM-Net公开器件相对退化阶段任务不同，不能直接替换后者的结果表。
可作为电表主线的预测模块候选，但尚未接入在线数字孪生更新，也不预测真实RUL。
下一步应优先收集独立批次或新的检测轮次做冻结验证；再比较稳健状态更新、缺测与校准偏移条件下的表现。
目前没有证据支持继续单纯加深网络，也不能以这次小幅超过RF宣称普遍超过传统机器学习。
'''
    (root / 'report.md').write_text(report, encoding='utf-8')
    print(root / 'report.md')


if __name__ == '__main__':
    main()
