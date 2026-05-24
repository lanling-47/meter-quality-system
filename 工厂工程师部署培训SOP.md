# 工厂工程师部署培训 SOP

## 一、培训目标

让工厂工程师掌握以下能力：

1. 能独立把智能电表质量管控系统部署到工厂电脑或服务器。
2. 能启动、停止、重启系统。
3. 能导入 Excel 数据或进行手动录入。
4. 能备份数据库和模型文件。
5. 能处理常见问题，例如端口占用、Python 未安装、页面打不开、数据不显示。

建议培训方式：

> 先演示一遍，再让工程师照着 SOP 自己部署一遍，最后按验收清单逐项确认。

## 二、推荐部署方式

如果工厂只是单机使用、现场录入、演示和小规模预测，推荐：

```text
Windows 离线部署
```

如果工厂有 IT 服务器、Docker 运维能力、希望长期稳定运行，推荐：

```text
Docker 部署
```

本系统优先建议先教工厂工程师 Windows 离线部署，因为门槛最低，不依赖互联网，不需要他们懂后端开发。

## 三、部署前准备

### 1. 硬件要求

最低可用配置：

```text
CPU：普通双核/四核均可
内存：8GB 起步，推荐 16GB
硬盘：建议 SATA SSD，至少 20GB 可用空间
系统：Windows 10 / Windows 11
浏览器：Edge 或 Chrome
```

不建议使用：

```text
4GB 内存 + 机械硬盘
```

原因是系统启动、Excel 导入、浏览器访问和模型预测会比较卡。

### 2. 软件包准备

交付给工厂工程师的部署包建议包含：

```text
meter-quality-system\
├── start.bat
├── main.py
├── requirements.txt
├── app\
├── templates\
├── static\
├── data\
├── ml_models\
├── meter_quality.db
├── offline_install\
│   └── python-3.11.9-amd64.exe
└── offline_packages_windows\
    └── *.whl
```

必须强调：

```text
offline_packages_windows 不能删除
meter_quality.db 不能删除
ml_models 不能删除
```

## 四、现场教学流程

### 第一步：讲清楚系统是什么

可以这样给工厂工程师解释：

> 这套系统是本地运行的智能电表质量管控软件，不需要外网。它通过浏览器访问，后台服务运行在本机 8090 端口。所有数据保存在本机 SQLite 数据库 `meter_quality.db` 中，模型文件保存在 `ml_models` 目录中。

需要让工程师记住三个关键词：

```text
启动脚本：start.bat
访问地址：http://localhost:8090
数据库文件：meter_quality.db
```

### 第二步：解压部署包

建议路径：

```text
D:\meter-quality-system\
```

不建议放在：

```text
桌面
下载目录
微信临时文件夹
中文层级过深的路径
```

原因：

- 路径过深容易导致依赖安装或文件读取异常。
- 桌面和下载目录容易被误删。
- 微信文件夹容易清理或移动。

### 第三步：安装 Python

如果工厂电脑没有 Python：

```text
进入 offline_install\
双击 python-3.11.9-amd64.exe
```

安装时必须勾选：

```text
Add python.exe to PATH
```

安装完成后关闭安装窗口。

如果电脑已经安装 Python 3.10 或 3.11，可以跳过。

### 第四步：启动系统

进入系统目录：

```text
D:\meter-quality-system\
```

双击：

```text
start.bat
```

第一次启动会自动完成：

```text
创建 venv 虚拟环境
离线安装 Python 依赖
启动 FastAPI 服务
```

首次启动可能需要几分钟。

看到类似内容说明启动成功：

```text
Application startup complete
Uvicorn running on http://0.0.0.0:8090
```

提醒工程师：

```text
不要关闭这个黑色命令窗口。
关闭窗口后系统服务会停止。
```

### 第五步：浏览器访问系统

打开 Edge 或 Chrome，输入：

```text
http://localhost:8090
```

默认登录：

```text
账号：admin
密码：admin123
```

如果是同一局域网其他电脑访问，需要输入部署电脑 IP：

```text
http://部署电脑IP:8090
```

例如：

```text
http://192.168.1.50:8090
```

如果其他电脑访问不了，需要检查：

- 部署电脑防火墙是否允许 8090 端口。
- 部署电脑和访问电脑是否在同一局域网。
- `start.bat` 窗口是否仍在运行。

## 五、教工程师日常怎么用

建议给工程师固定一套操作顺序：

```text
1. 双击 start.bat 启动系统
2. 浏览器打开 http://localhost:8090
3. 登录 admin/admin123
4. 进入“部署向导”检查状态
5. 进入“数据管理”或“数据导入”
6. 上传 Excel 或逐条录入数据
7. 查看“质量看板”
8. 使用“出厂故障预测”
9. 查看“生产追溯”“异常检测”“根因追溯”
10. 下班前备份 meter_quality.db
```

## 六、数据导入培训

### 1. Excel 上传

操作路径：

```text
数据导入页面 -> 选择 Excel -> 点击上传
```

告诉工程师：

- 不要随意改模板表头。
- 不要合并关键数据单元格。
- 日期、编号、条码要保持原格式。
- 上传后要看页面是否提示成功。

### 2. data/raw 批量导入

适合管理员批量初始化。

把 Excel 放到：

```text
data\raw\
```

然后在系统中点击批量导入。

### 3. 手动录入

适合少量补录：

```text
数据管理 -> 新增记录 -> 保存
```

重点告诉工程师：

- 条码、表号、批次号要尽量填准确。
- 不确定的数据不要乱填，可以留空或备注。
- 手动录入后要回到列表确认是否保存成功。

## 七、数据备份培训

必须重点教。

系统最重要的文件是：

```text
meter_quality.db
ml_models\
data\raw\
```

建议每天或每周备份到：

```text
D:\meter-quality-backup\2026-05-19\
```

备份步骤：

```text
1. 关闭 start.bat 窗口
2. 复制 meter_quality.db
3. 复制 ml_models 文件夹
4. 复制 data/raw 文件夹
5. 重新双击 start.bat 启动
```

不要在系统运行中直接覆盖数据库。

## 八、常见问题处理

### 1. 页面打不开

先检查：

```text
start.bat 窗口是否还在
浏览器地址是否是 http://localhost:8090
```

如果窗口关闭：

```text
重新双击 start.bat
```

### 2. 提示端口 8090 被占用

Windows 命令行执行：

```cmd
netstat -ano | findstr :8090
```

找到 PID 后执行：

```cmd
taskkill /F /PID 进程ID
```

再重新双击 `start.bat`。

### 3. 提示找不到 Python

处理：

```text
重新安装 offline_install\python-3.11.9-amd64.exe
安装时勾选 Add python.exe to PATH
安装完成后重新打开 start.bat
```

### 4. 依赖安装失败

检查：

```text
offline_packages_windows\ 是否存在
requirements.txt 是否存在
```

如果 `offline_packages_windows` 没有 `.whl` 文件，说明部署包不完整，需要重新拷贝。

### 5. 登录后没有数据

原因通常是还没有导入数据。

处理：

```text
进入 数据导入 / 数据管理
上传 Excel 或手动录入
再回到 dashboard 查看
```

### 6. 模型预测不可用

系统没有训练模型时，会使用规则引擎。

如果要使用 AI 模型：

```text
先导入数据
进入模型训练或预测页面
点击训练
等待训练完成
```

训练时间取决于电脑性能。

## 九、现场验收清单

让工程师逐项打勾：

| 检查项 | 是否通过 |
|---|---|
| Python 已安装 |  |
| `start.bat` 能正常启动 |  |
| 浏览器能打开 `http://localhost:8090` |  |
| admin/admin123 能登录 |  |
| 部署向导页面能打开 |  |
| 数据管理页面能打开 |  |
| 能上传或录入一条数据 |  |
| 质量看板能显示统计 |  |
| 出厂故障预测能输出结果 |  |
| 异常检测页面能显示图表 |  |
| 关闭并重启后数据不丢失 |  |
| 已完成数据库备份演示 |  |

## 十、给工程师的极简版口诀

可以让他们记住：

```text
解压到 D 盘
装 Python
双击 start.bat
浏览器打开 localhost:8090
数据先导入
数据库常备份
窗口别乱关
```

## 十一、如果使用 Docker 部署

如果工厂工程师懂 Docker，可按以下方式部署：

```bash
docker-compose up -d
```

查看状态：

```bash
docker-compose ps
```

查看日志：

```bash
docker-compose logs -f
```

停止：

```bash
docker-compose stop
```

重启：

```bash
docker-compose restart
```

注意：

```text
不要执行 docker-compose down -v
```

因为这会删除数据卷，可能导致数据库和模型丢失。

Docker 方式更适合有 IT 运维人员的工厂服务器，普通现场工程师建议优先使用 Windows 离线部署。

## 十二、培训建议安排

建议安排 60-90 分钟：

| 时间 | 内容 |
|---|---|
| 10 分钟 | 讲系统用途、访问方式、数据保存位置 |
| 20 分钟 | 演示 Windows 离线部署 |
| 20 分钟 | 演示数据导入、手动录入、预测和看板 |
| 15 分钟 | 演示备份、重启、端口占用处理 |
| 15 分钟 | 让工程师独立操作一遍 |
| 10 分钟 | 按验收清单确认 |

最终目标不是让他们理解代码，而是让他们会：

```text
会启动
会导入
会录入
会备份
会处理常见故障
```

