# -*- coding: utf-8 -*-
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
ROOT=Path(__file__).resolve().parent
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,
                     'font.size':10,'figure.dpi':160,'axes.spines.top':False,'axes.spines.right':False})
NAMES={'mlp_manufacturing':'仅制造 MLP','mlp_observations':'仅检测 MLP','mlp_combined':'联合 MLP',
 'tabm_combined':'联合 TabM','grouped_no_response':'分组融合 无响应监督','remf_v6':'REMF v6 完整版',
 'remf_no_groups':'REMF 去分组','remf_no_mask':'REMF 去响应掩码','logistic_combined':'逻辑回归',
 'rf_combined':'随机森林','hist_combined':'直方图梯度提升','majority_prior':'训练频率先验'}
SHORT=['CT与补偿失配','采样焊点异常','残留泄漏','RS485连接异常','LCD连接异常','供电裕量不足','RS485器件异常','LCD屏体或驱动异常']
def js(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in r)+' |' for r in rows)
def pct(x):return f'{100*x:.2f}%'

def main():
    selection=js('selection_frozen.json');ev=js('evaluation.json');env=js('environment.json');meta=js('preprocessing.json')
    byseed=pd.read_csv(ROOT/'test_metrics_by_seed.csv');ens=pd.read_csv(ROOT/'三种子集成测试指标.csv').set_index('model')
    mean=byseed.groupby('model').mean(numeric_only=True);std=byseed.groupby('model').std(numeric_only=True).fillna(0)
    order=list(NAMES); selected=selection['selected_model']; neural=selection['selected_neural']
    mainrows=[]
    for n in order:
        mainrows.append([NAMES[n],f'{100*mean.loc[n,"macro_f1_8"]:.2f} ± {100*std.loc[n,"macro_f1_8"]:.2f}',
             pct(mean.loc[n,'accuracy']),pct(mean.loc[n,'top2']),pct(mean.loc[n,'top3']),pct(ens.loc[n,'macro_f1_8'])])
    # Means across seeds, not the test-selected best run.
    fig,ax=plt.subplots(figsize=(9,5.6));pos=np.arange(len(order))
    ax.barh(pos,[100*mean.loc[n,'macro_f1_8'] for n in order],xerr=[100*std.loc[n,'macro_f1_8'] for n in order],
            color=['#222222' if n==neural else '#999999' for n in order],capsize=3)
    ax.set_yticks(pos,[NAMES[n] for n in order]);ax.invert_yaxis();ax.set_xlim(0,100)
    ax.set_xlabel('八类宏F1 / %');ax.set_title('固定测试集51例  神经模型为3种子均值与样本标准差')
    ax.grid(axis='x',alpha=.15);fig.tight_layout();fig.savefig(FIG/'模型对比.png');plt.close(fig)
    causes=meta['cause_codes'];counts=pd.read_csv(ROOT/'标签分布.csv').set_index('split').reindex(columns=causes)
    fig,ax=plt.subplots(figsize=(8.5,4));pos=np.arange(8);width=.25
    for k,(s,c) in enumerate([('train','#333333'),('validation','#888888'),('test','#cccccc')]):ax.bar(pos+(k-1)*width,counts.loc[s],width,label=s,color=c)
    ax.set_xticks(pos,SHORT,rotation=25,ha='right');ax.set_ylabel('台数');ax.legend(frameon=False,ncol=3);ax.set_title('单因模拟证据292例  稀有原因测试数量有限');fig.tight_layout();fig.savefig(FIG/'标签分布.png');plt.close(fig)
    predictions=pd.read_csv(ROOT/'测试逐台候选原因.csv');known=predictions[(predictions.model==selected)&(predictions.confirmation_status=='单因证据支持')]
    y=known.label.map({c:i for i,c in enumerate(causes)}).to_numpy(int);p=known[['p_'+c for c in causes]].to_numpy()
    cm=confusion_matrix(y,p.argmax(1),labels=np.arange(8));fig,ax=plt.subplots(figsize=(8,5.8))
    ax.imshow(cm,cmap='Greys',vmin=0,vmax=max(1,cm.max()))
    for i in range(8):
        for j in range(8):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black',fontsize=10)
    ax.set_xticks(range(8),SHORT,rotation=35,ha='right');ax.set_yticks(range(8),SHORT)
    ax.set_xlabel('候选第一位');ax.set_ylabel('模拟单因证据标签');ax.set_title('验证集选定模型 '+NAMES[selected]);fig.tight_layout();fig.savefig(FIG/'混淆矩阵.png');plt.close(fig)
    history=js('training_history.json');fig,axes=plt.subplots(1,2,figsize=(9,3.7))
    for n,style in [('mlp_combined','-'),('tabm_combined','--'),('remf_v6',':')]:
        run=next(r for r in history if r['model']==n and r['seed']==17);h=pd.DataFrame(run['history'])
        axes[0].plot(h.epoch,h.ce,label=NAMES[n],ls=style);axes[1].plot(h.epoch,h.validation_macro_f1*100,label=NAMES[n],ls=style)
    axes[0].set_ylabel('训练分类交叉熵');axes[1].set_ylabel('验证八类宏F1 / %')
    for ax in axes:ax.set_xlabel('轮次');ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(FIG/'训练曲线.png');plt.close(fig)
    from metrics import temperature
    cp=temperature(p,selection['calibration'][selected]['temperature']);conf=cp.max(1);corr=cp.argmax(1)==y
    fig,ax=plt.subplots(figsize=(6,4.1));ax.plot([0,1],[0,1],color='#999999',ls='--')
    for lo in np.arange(0,1,.2):
        m=(conf>=lo)&(conf<lo+.2)
        if m.any():ax.scatter(conf[m].mean(),corr[m].mean(),s=30+5*m.sum(),c='black');ax.annotate('n='+str(m.sum()),(conf[m].mean(),corr[m].mean()),xytext=(5,4),textcoords='offset points',fontsize=8)
    ax.set_xlim(0,1.05);ax.set_ylim(0,1.05);ax.set_xlabel('平均校准模型置信度');ax.set_ylabel('实际正确比例');ax.set_title('校准可靠性仅51例  不代表企业概率');fig.tight_layout();fig.savefig(FIG/'置信度校准.png');plt.close(fig)
    ci=ev['bootstrap_by_batch'][selected];sel=ens.loc[selected];nn=ens.loc[neural]
    perclass=js('per_class_reports.json')[selected]
    classrows=[[SHORT[i],int(perclass[c]['support']),pct(perclass[c]['precision']),pct(perclass[c]['recall']),pct(perclass[c]['f1-score'])] for i,c in enumerate(causes)]
    rs=pd.read_csv(ROOT/'维修响应辅助任务.csv');rrows=[]
    for r in rs.itertuples():rrows.append([NAMES[r.model],r.domain,r.n,f'{r.recorded_response_mae:.4f}',f'{r.training_action_mean_mae:.4f}'])
    ablation=[[NAMES[n],pct(mean.loc[n,'macro_f1_8']),f'{100*std.loc[n,"macro_f1_8"]:.2f}',pct(ens.loc[n,'macro_f1_8'])] for n in ['remf_v6','grouped_no_response','remf_no_groups','remf_no_mask']]
    paired=ev['paired_neural_vs_rf'];cal=selection['calibration'][selected]
    retained=int(sel.selective_retained);retacc=pct(sel.selective_accuracy) if retained else '未保留病例'
    recommendation='本轮没有稳定证明神经模型优于随机森林。' if paired['p025']<=0<=paired['p975'] else '配对区间未跨零，但仍是小规模合成留出的探索性结果，不能外推企业效果。'
    missing=pd.read_csv(ROOT/'缺失输入压力测试.csv')
    trainseconds=float(pd.read_csv(ROOT/'validation_metrics.csv').seconds.sum())
    text=f'''# 电表故障根因模型 GPU 训练与测试报告

实验日期 2026年11月16日

## 1 本轮结论

使用冻结v6样本完成24次GPU神经网络训练和4个CPU传统基线训练。本轮仅研究首次出厂检测异常后的根因辅助分析，不训练检测前预测，也不将数字孪生、博世电锤验证作为已完成内容。

按验证集三种子平均八类宏F1选定的模型是{NAMES[selected]}；选定的神经模型是{NAMES[neural]}。测试单因证据仅51例，选定模型集成宏F1为{pct(sel.macro_f1_8)}，Top2为{pct(sel.top2)}，Top3为{pct(sel.top3)}。这些是模拟证据标签上的结果，不是真实企业根因识别准确率。

选定神经模型的集成宏F1为{pct(nn.macro_f1_8)}。与随机森林的配对批次bootstrap差值区间为[{100*paired['p025']:.2f}, {100*paired['p975']:.2f}]个百分点。{recommendation} 完整版REMF是否更好，必须同时结合消融和逐类表现判断，不能按名称预设胜出。

## 2 数据与决策时点

10000台独立电表包含9000台首次通过与1000台首次故障。根因单标签监督仅使用292台单因模拟证据病例，训练187、验证54、测试51。多因素36例、改善待确认570例、未确认102例均不强制赋予单一原因。正常样本不加入故障原因分类，训练6400台正常与故障的无标签前置观测仅用于拟合输入均值与标准差。

{table(['原因','训练','验证','测试'],[[SHORT[i],int(counts.loc['train',c]),int(counts.loc['validation',c]),int(counts.loc['test',c])] for i,c in enumerate(causes)])}

![单因证据分布](figures/标签分布.png)

输入为34项制造及元件观测和12项首次检测观测。推理时点固定在首次检测后、检查复测维修之前。维修、复测、诊断检查及确认结论不作为推理输入；仅训练集首轮维修响应允许用于辅助监督。训练可用无伴随干预的首轮维修{meta['aux_first_repairs']['train']}例。后续维修受首轮处理影响，本轮不用于辅助标签。

沿用原始train/validation/test，虚拟批次、背景电表、CT及晶振来源不交叉。真实配对整机制造及维修确认记录为0；29台电表背景与有限元件抽检扩展为半合成数据，并不代表10000台实测。v6测试此前已做简单模型探索性审计，本轮新增实验冻结不等于建立了全新的外部盲测。

## 3 本机运行环境与冻结协议

实际解释器为 `{env['executable']}`，Python {env['python'].split()[0]}，PyTorch {env['torch']}，CUDA运行时{env['cuda_runtime']}，GPU为{env['gpu']}。所有24次神经训练均强制cuda:0，GPU不可用会直接报错，没有静默退回CPU。传统逻辑回归与树模型使用CPU。

神经训练最大300轮、AdamW学习率0.001、权重衰减0.01，45轮未改进早停。使用固定种子17、43、101；训练类别权重为样本频率平方根倒数。每次训练仅按验证宏F1选最佳轮次，同分按验证NLL；模型家族按三种子验证宏F1均值选定。测试预测在selection_frozen.json写出后才执行。本轮训练与基线拟合累计计时约{trainseconds/60:.1f}分钟，峰值CUDA张量显存{env['peak_cuda_allocated_bytes']/1024**2:.1f}MiB，不含桌面与驱动开销。

GPU环境复用本机Conda yolov9，因旧科学计算库加载失败，另在包外目标目录安装科学计算依赖，以PYTHONPATH加载。没有改动旧项目CPU环境、已有质量系统或v6数据。实际包版本见environment.json与依赖清单。

## 4 方法及本轮候选创新

基线包括仅制造MLP、仅检测MLP、联合MLP、官方TabM、逻辑回归、随机森林、直方图梯度提升与训练频率先验。

REMF v6将制造字段分成热工艺与焊膏、CT、补偿、清洁、装配、来料裕量六组；按候选原因学习组表示，与首次检测表示融合进行8类排序。仅训练时，原因对应的响应分支拟合“维修前两次复测均值减首轮维修后三次均值”，按训练集尺度归一化，并对计量、通信、显示域均衡计算损失，辅助损失权重0.2。

响应掩码表达预先约定的工程关系：CT、采样焊点和残留作用于计量域，通信连接与收发器作用于通信域，LCD连接与驱动作用于显示域，供电可涉及三域。它不是模型发现的因果图；组权重也不等于因果贡献。记录维修动作的选择存在混杂，前后变化不是无偏因果效应。

候选创新是检验分组先验与实际记录维修响应能否辅助有限标签根因诊断。本轮不宣称首创、不宣称已实现严格概念瓶颈或数字孪生，也不把复杂模型默认视为更优。

## 5 根因分类与候选排序结果

{table(['模型','八类宏F1 均值±标准差','Top1均值','Top2均值','Top3均值','集成宏F1'],mainrows)}

神经模型均值与标准差基于3个独立训练种子；传统基线只有1次训练，标准差0不表示没有统计不确定性。集成先平均3个种子的类别概率，再计算指标；它与单次模型指标均值不同，不能混写。

选定模型Top1正确{round(sel.accuracy*51)}/51台，即{pct(sel.accuracy)}；Top3命中51/51只表示正确原因进入三个候选，并非八类根因识别率100%。其中LCD屏体或驱动异常仅1例且Top1未识别，供电与LCD连接也较弱，宏F1因此明显低于总体准确率。简单检测特征可以区分部分故障表征，但不等于已经定位真实制造原因。

本轮仅制造MLP集成宏F1为{pct(ens.loc['mlp_manufacturing','macro_f1_8'])}，联合MLP为{pct(ens.loc['mlp_combined','macro_f1_8'])}，均低于仅检测模型。可能涉及标签稀缺、制造观测信号弱、组间分布变化和小样本过拟合；本实验尚不能分清各因素贡献。测试中直方图梯度提升宏F1为{pct(ens.loc['hist_combined','macro_f1_8'])}，高于验证选定模型，说明验证排序存在不稳定；没有据此更换已冻结的选模结果，也不宣称深度学习优于全部传统方法。

![模型比较](figures/模型对比.png)

### 5.1 验证选定模型逐类表现

{table(['原因','测试例数','精确率','召回率','F1'],classrows)}

![混淆矩阵](figures/混淆矩阵.png)

测试单因病例来自{ev['test_single_batches']}个虚拟批次。选定模型集成宏F1按批次重抽样2000次的2.5%至97.5%范围为[{pct(ci['p025'])}, {pct(ci['p975'])}]。重抽样固定8类，部分重抽样没有稀有类，因此区间仅供探索性参考。模拟标签选择机制、来源规模和稀有类别不足限制了外推。

三类计量子集仅{ev['test_metering3_n']}例。附表的macro_f1_metering3是在已知属于前三类时，将8类输出限制并重新排序得到的闭集指标；并非另外训练的三分类器，也不等于端到端八类分类效果。它与上一轮专门三分类随机森林0.719不能直接作优劣比较。

## 6 消融实验及维修响应

{table(['设置','宏F1均值','种子标准差 百分点','集成宏F1'],ablation)}

去响应监督、去分组、去掩码均使用同一数据划分、相同种子和选轮次规则。消融不一致或小幅提升只能作趋势分析，不能据此声称稳定创新增益。没有根据测试结果修改本轮网络或样本重新训练。

完整版REMF集成宏F1为{pct(ens.loc['remf_v6','macro_f1_8'])}，去响应监督版本为{pct(ens.loc['grouped_no_response','macro_f1_8'])}。本轮响应监督没有改善这项分类指标；不能将辅助响应预测误差的改善直接写成根因分类创新已得到验证。当前更适合将REMF作为待验证候选方法，下一轮用新的验证数据检验，而不是增加网络复杂度追求本测试集分数。

![训练曲线](figures/训练曲线.png)

首轮维修响应在全部可用故障测试病例上评估，不限单因标签。按实际维修目标的训练均值是简单参照；计量MAE单位为误差百分点，通信为每200次失败次数，显示为缺段数。

{table(['模型','故障域','例数','响应MAE','训练动作均值MAE'],rrows)}

响应预测只评价已记录动作，不能支持对未尝试维修的成功率承诺。如果分支比训练动作均值还差，应保留为研究辅助项，不部署维修效果数值预测。

## 7 置信度 多因与未确认病例

温度校准仅用验证NLL选择，选定模型温度为{cal['temperature']:.3f}。校准前测试NLL={sel.nll:.4f}，校准后={sel.calibrated_nll:.4f}；ECE从{sel.ece:.4f}变为{sel.calibrated_ece:.4f}。51例上的校准估计不稳定，模型置信度不是企业真实概率。

![置信度](figures/置信度校准.png)

仅在验证集至少保留10例且经验正确率不低于80%时选择最大覆盖阈值；若不存在满足条件的阈值则全部提示证据不足。冻结阈值{cal['threshold']:.6f}，验证保留{cal['validation_retained']}例；测试51例中保留{retained}例，覆盖率{pct(sel.selective_coverage)}，保留部分正确率为{retacc}。验证达标不保证测试或企业达标。

测试共有180台故障，所有模型均输出Top1至Top3。对其中{ev['multi_n']}台多因证据病例，选定模型Top3覆盖全部记录原因的比例为{pct(ev['multi_top3_all_covered']) if ev['multi_n'] else '无可评价病例'}。这里只评估候选覆盖，没有训练独立多标签模型。待确认或未确认病例没有根因真值，不能计算根因准确率，也不能将低置信度拒识率称为未知识别准确率。

输入缺失压力测试随机遮盖10%和30%的46维输入，并用训练均值填补，不重训练。它只是固定测试挑战，实际推理接口默认拒绝缺字段和空值，未假定现场缺失随机发生。

{table(['随机缺失比例','八类宏F1','Top3'],[[pct(r.missing_fraction),pct(r.macro_f1_8),pct(r.top3)] for r in missing.itertuples()])}

## 8 系统接口与复现交付

提供checkpoints下全部28个权重文件、preprocessing.json、固定选模与校准参数、逐台候选原因、逐种子指标、训练记录及验收结果。predict.py按字段白名单仅提取34项制造字段与12项首次观测；首次检测通过的电表不强制归因，缺字段和非有限值直接报错。

本轮只提供独立CSV推理入口，没有替换现有电表系统的运行模型。接入系统前需要更新八类字典、46维输入、校准与拒识状态，以及多因和未知病例的展示。此前v5五分类模型不能直接接收本版数据。

复现入口为run_training.ps1。所有神经权重重载已逐一检查验证预测一致；最终验收独立重算逐台概率对应的宏F1与准确率，核对数据/协议/代码/权重哈希及训练设备。详见验收结果.json。源样本请使用已交付的v6完整数据包，不在本包重复放入原始Excel。

## 9 论文表述与下一步

当前可以写：在具有参数重叠、偶发故障、维修不完全有效和不确定标签的半合成电表数据上，完成了八类候选原因诊断、传统基线与深度模型对比、分组及响应监督消融、候选排序和不确定性分析。

当前不能写：已验证真实企业根因识别率、已证明维修因果效应、已完成电表与博世电锤通用验证、数字孪生已上线，或深度学习显著优于传统方法。

下一步优先补充真实元件—工艺—检测—拆检/维修配对证据，并增加稀有原因的确认病例。跨产品通用性需要在电锤上用一致的数据结构与实验流程另行验证；数字孪生可承载产品履历与维修反馈，但不能替代真实根因证据。论文最终实验应重新冻结方案，准备新的来源或外部留出数据。
'''
    (ROOT/'模型训练与测试报告.md').write_text(text,encoding='utf-8')
    # Structured content for the bundled-runtime Word renderer.
    doc={'title':'电表故障根因模型训练与测试报告','subtitle':'v6 半合成样本  本机 RTX 4060  2026年11月16日',
         'selection':NAMES[selected],'neural':NAMES[neural],'summary':text.split('## 1 本轮结论\n\n')[1].split('\n\n## 2')[0],
         'main_headers':['模型','宏F1 均值±标准差','Top2','Top3'],
         'main_rows':[[r[0],r[1],r[3],r[4]] for r in mainrows],
         'counts':[[SHORT[i],int(counts.loc['train',c]),int(counts.loc['validation',c]),int(counts.loc['test',c])] for i,c in enumerate(causes)],
         'class_rows':classrows,'ablation':ablation,'response_rows':rrows,
         'source_markdown':'模型训练与测试报告.md'}
    (ROOT/'report_content.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
    print('REPORT FIGURES AND MARKDOWN WRITTEN')

if __name__=='__main__':main()
