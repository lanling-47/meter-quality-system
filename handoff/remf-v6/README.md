# 电表质量追溯系统交接

本目录包含 REMF V6 系统源码、两套模型的训练及推理代码、模型权重、模拟样本、Windows 离线安装包、员工手册和测试记录。

首次获取请安装 Git LFS，在终端运行：

```powershell
git lfs install
git clone --branch handoff/remf-v6-20261123 --single-branch https://github.com/lanling-47/meter-quality-system.git
cd meter-quality-system/handoff/remf-v6
```

使用 Windows 64 位 Python 3.11。运行 `01-install.cmd` 安装，运行 `02-start.cmd` 启动，浏览器打开 http://127.0.0.1:8092 。运行 `03-test.cmd` 可重新验证功能。

[详细交接与操作说明](README_从这里开始.md) · [测试记录](测试记录.md) · [模型训练与指标](research/训练与指标.md)
