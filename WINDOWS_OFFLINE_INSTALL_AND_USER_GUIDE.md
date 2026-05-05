# Windows 离线安装与使用简明指南

## 1. 适用场景

本指南用于厂方 Windows 断网电脑部署和使用智能电表质量管控系统。

支持系统：

```text
Windows 10 / Windows 11
```

访问地址：

```text
http://localhost:8090
```

默认账号：

```text
admin
```

默认密码：

```text
admin123
```

---

## 2. 部署包内容

解压后的目录应包含：

```text
start.bat                          一键启动脚本
offline_install\python-3.11.9-amd64.exe  Python 安装包
offline_packages_windows\          离线依赖包
meter_quality.db                    本地数据库
ml_models\                          模型目录
data\raw\                           原始数据目录
app\ / templates\ / static\         系统程序文件
```

注意：

```text
offline_packages_windows 目录不能删除。
meter_quality.db 不能随意删除。
ml_models 目录不能随意删除。
```

---

## 3. 安装步骤

### 第一步：解压系统包

将压缩包解压到固定目录，例如：

```text
D:\meter-quality-system\
```

建议不要放在桌面或下载目录。

### 第二步：安装 Python

如果电脑没有 Python，进入：

```text
offline_install\
```

双击安装：

```text
python-3.11.9-amd64.exe
```

安装时必须勾选：

```text
Add python.exe to PATH
```

如果已经安装 Python 3.10 或以上版本，可跳过此步骤。

### 第三步：启动系统

进入系统目录，双击：

```text
start.bat
```

首次启动会自动：

```text
创建虚拟环境
离线安装依赖
启动系统服务
```

首次启动可能需要几分钟。

看到下面信息说明启动成功：

```text
Uvicorn running on http://0.0.0.0:8090
Application startup complete
```

注意：

```text
不要关闭 start.bat 窗口。
关闭窗口后系统会停止运行。
```

### 第四步：打开系统

打开 Edge 或 Chrome，访问：

```text
http://localhost:8090
```

登录：

```text
账号：admin
密码：admin123
```

---

## 4. 日常使用流程

推荐使用顺序：

```text
1. 双击 start.bat 启动系统
2. 浏览器打开 http://localhost:8090
3. 登录系统
4. 进入“部署向导”检查系统状态
5. 下载 Excel 模板
6. 导入检测数据
7. 查看质量看板
8. 使用出厂故障预测
9. 查看生产追溯和老化分析
10. 定期备份数据
```

---

## 5. 主要功能入口

| 功能 | 地址 | 用途 |
|---|---|---|
| 登录 | `/login` | 登录系统 |
| 部署向导 | `/setup` | 查看路径、模板、数据状态 |
| 数据导入 | `/import` | 上传 Excel、手动录入、批量导入 |
| 质量看板 | `/dashboard` | 查看质量统计和风险概况 |
| 出厂故障预测 | `/factory-fault` | 预测电表出厂故障风险 |
| 生产追溯 | `/production` | 查询电表生产过程记录 |
| 老化分析 | `/degradation` | 查看老化试验趋势 |

示例：

```text
http://localhost:8090/dashboard
```

---

## 6. 数据导入方式

系统支持三种导入方式。

### 方式一：上传 Excel

```text
进入 /import
选择 Excel 文件
点击上传导入
```

### 方式二：批量导入 data/raw

将 Excel 文件放入：

```text
data\raw\
```

然后在页面点击批量导入。

### 方式三：手动录入

在 `/import` 页面选择手动录入，适合少量补录。

---

## 7. 出厂故障预测

进入：

```text
/factory-fault
```

支持两种方式：

```text
1. 手动输入参数预测
2. 输入电表编号查询并预测
```

输出结果包括：

```text
故障概率
风险等级
可能故障类型
关键影响因素
```

如果还没有训练模型，系统会自动使用规则引擎，仍可正常预测。

---

## 8. 数据存放位置

假设系统解压到：

```text
D:\meter-quality-system\
```

主要数据位置：

```text
数据库：D:\meter-quality-system\meter_quality.db
模型：D:\meter-quality-system\ml_models\
原始数据：D:\meter-quality-system\data\raw\
监听目录：D:\watch_dir\
```

系统使用相对路径，复制到其他电脑后会自动适配。

---

## 9. 数据备份

请定期备份以下内容：

```text
meter_quality.db
ml_models\
data\raw\
```

推荐备份到：

```text
D:\meter-quality-backup\日期\
```

恢复时：

```text
1. 关闭 start.bat 窗口
2. 将备份文件复制回系统目录
3. 重新双击 start.bat
```

---

## 10. 常见问题

### 10.1 双击 start.bat 提示找不到 Python

处理：

```text
安装 offline_install\python-3.11.9-amd64.exe
安装时勾选 Add python.exe to PATH
安装后重新双击 start.bat
```

### 10.2 浏览器打不开 localhost:8090

先检查：

```text
start.bat 窗口是否还在运行
```

如果窗口关闭，请重新双击 `start.bat`。

如果提示端口被占用，执行：

```cmd
netstat -ano | findstr :8090
taskkill /F /PID 进程ID
```

然后重新启动。

### 10.3 离线安装依赖失败

检查目录是否存在：

```text
offline_packages_windows\
```

该目录中应有很多 `.whl` 文件。如果没有，说明部署包不完整。

### 10.4 页面能打开但没有数据

说明还没有导入数据。

处理：

```text
进入 /setup 下载模板
填写 Excel
进入 /import 导入数据
回到 /dashboard 查看统计
```

### 10.5 关闭窗口后系统不能访问

正常现象。

```text
start.bat 窗口关闭后，系统服务会停止。
需要重新双击 start.bat。
```

---

## 11. 现场验收清单

| 检查项 | 结果 |
|---|---|
| 可以安装 Python |  |
| 双击 start.bat 可以启动 |  |
| 浏览器可以访问 localhost:8090 |  |
| admin/admin123 可以登录 |  |
| 可以下载 Excel 模板 |  |
| 可以导入 Excel 数据 |  |
| 质量看板有统计结果 |  |
| 出厂故障预测可以使用 |  |
| 生产追溯可以查询 |  |
| 老化分析可以查看 |  |
| 重启后数据不丢失 |  |

---

## 12. 一句话说明

厂方断网 Windows 电脑部署时：

```text
解压系统包 → 安装 Python → 双击 start.bat → 浏览器打开 localhost:8090 → 登录使用
```

系统完全本地运行，不需要互联网。
