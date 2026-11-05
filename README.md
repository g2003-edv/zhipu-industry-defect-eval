# 工业零件瑕疵检测评测系统

对比 YOLOv8 与智谱 GLM-4V 在工业零件表面瑕疵识别任务上的表现，并在原始画面与光照扰动
（亮度 ±30、对比度 ×0.7 / ×1.3）下统计准确率、精确率、召回率、F1 与单张推理耗时，
自动生成 CSV 与两张对比图。

## 结论与可比性（重要）

**本项目中的 YOLOv8n 是未微调的 COCO 预训练检测器，不是瑕疵检测模型，两者指标不可直接比较。**

- COCO 权重没有"瑕疵 / 划痕 / 破损"类别，`yolo_detect()` 只能按"是否检出任意目标 + 检测框
  置信度均值"给出判定，它的召回率反映的是 COCO 类别是否出现在矿泉水瓶照片里，而不是缺陷
  检出能力。因此它在这里只作为一个**负向基线**（说明"直接拿通用检测器当质检器"不可行）。
- GLM-4V 是通用多模态模型，使用零样本提示，未在本数据集上微调。
- 要得出"多模态模型优于检测器"的结论，需要：用同一批数据训练/微调一个真正的瑕疵检测器
  （如 YOLOv8-cls / 分割模型）作为正基线，并在更大规模、独立采集的测试集上复测。
- 现有结果来自 44 张同一只矿泉水瓶的实拍，样本量小、场景单一，仅用于流程验证。

## 项目目录

```
.
├── data/                # 数据集（不随仓库分发，见 data/README.md）
│   ├── defect/          # 带瑕疵图片
│   └── normal/          # 无瑕疵图片
├── eval_main.py         # 评测主程序
├── eval_result.csv      # 历史评测结果（修复前的判定逻辑生成，见下文）
├── drift_curve.png      # 光照扰动下 F1 衰减曲线
├── metric_bar.png       # 原始场景双模型指标对比
├── requirements.txt     # 依赖（含版本区间）
├── Dockerfile
├── .dockerignore
└── README.md
```

## 评测方法

1. 固定 5 个场景：原始原图、亮度 -30、亮度 +30、对比度 0.7、对比度 1.3。
2. 每个场景下遍历 `data/defect` 与 `data/normal`，逐张推理并累计 TP/TN/FP/FN。
3. GLM-4V 要求输出 JSON `{"defect": true|false}`：
   - JSON 解析成功 → 按布尔值判定；
   - JSON 解析失败 → 文本兜底，**先匹配否定表达**（"没有瑕疵""无瑕疵""未见异常"等），
     再匹配肯定表达；仍无法判断则记为 `error`；
   - 调用异常（网络、鉴权、响应结构异常）同样记为 `error`。
4. `error` 不计入 TP/TN/FP/FN，指标只在有效样本上计算，CSV 输出 `错误数` 与 `有效样本数`。
5. 指标定义：准确率 `(TP+TN)/有效样本`、精确率 `TP/(TP+FP)`、召回率 `TP/(TP+FN)`、
   F1 为二者调和平均；耗时用 `time.perf_counter()` 统计，取单张均值。

## 历史结果（存档）

原始运行（44 张实拍：29 张瑕疵、15 张正常）的 F1 分数：

| 数据场景 | YOLOv8n（未微调 COCO 基线） | GLM-4V（零样本） |
| --- | --- | --- |
| 原始原图 | 0.176 | 0.857 |
| 亮度 -30 | 0.121 | 0.857 |
| 亮度 +30 | 0.065 | 0.893 |
| 对比度 0.7 | 0.171 | 0.893 |
| 对比度 1.3 | 0.242 | 0.873 |

> 注意：`eval_result.csv` 是**修复前**的判定逻辑产出的存档数据。旧逻辑用
> `"有瑕疵" in ans` 判断，"没有瑕疵"会被误判为有瑕疵，且异常被静默记为 `False`。
> 本次修复没有重跑评测（不造新数据），因此上表数值仅供存档参考，不能作为修复后代码的
> 性能结论。重新运行会覆盖该文件并写入 `错误数`、`有效样本数`、`baseline_note` 列。

## 环境与版本

- Python 3.10+
- 依赖见 `requirements.txt`，关键约束：

| 包 | 版本区间 | 说明 |
| --- | --- | --- |
| torch | `>=2.0,<2.4` | 与 torchvision 0.15~0.18 成对安装 |
| torchvision | `>=0.15,<0.19` | 与 torch 版本匹配 |
| ultralytics | `>=8.1,<8.3` | 与上述 torch 组合兼容 |
| numpy | `>=1.24,<2.0` | 避免 numpy 2.x 与旧版 opencv/ultralytics 的 ABI 冲突 |

**关于 `torch.load` 的安全边界**：本项目不再全局替换 `torch.load`、不再强制
`weights_only=False`。旧版 `ultralytics 8.1` 的权重加载路径在 `torch>=2.6`（默认
`weights_only=True`）下会报错，正确做法是锁定兼容版本，而不是关闭 pickle 安全校验。
如必须使用新版 torch，请升级到已适配 `weights_only` 的 ultralytics 版本，或先转换权重，
不要用猴补丁绕过。

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ZHIPU_API_KEY` | 无（**必填**） | 智谱 API Key。只从环境变量读取，缺失时启动即报错退出（不再交互输入，避免 Docker 挂起） |
| `GLM_MODEL` | `glm-4v-plus` | 云端模型名 |
| `YOLO_WEIGHTS` | `./yolov8n.pt` | 权重路径，缺失时 ultralytics 会在运行时下载 |
| `DATA_DIR` | `./data` | 数据根目录 |
| `RESULT_CSV` | `./eval_result.csv` | 结果 CSV 路径 |
| `DRIFT_CURVE_PNG` | `./drift_curve.png` | 衰减曲线输出路径 |
| `METRIC_BAR_PNG` | `./metric_bar.png` | 指标柱状图输出路径 |
| `YOLO_CONF` | `0.3` | YOLO 推理置信度阈值 |
| `YOLO_MEAN_CONF` | `0.7` | 判定为"有瑕疵"的平均置信度阈值 |

运行时打印设备信息：`DEVICE` 由 `torch.cuda.is_available()` 决定，并作为 `device`
参数传给 YOLO 推理（旧实现用 `os.path.exists("cuda")`，结果永远是 CPU）。

## 运行

```bash
export ZHIPU_API_KEY="your-key"        # Windows PowerShell: $env:ZHIPU_API_KEY="your-key"
python eval_main.py
```

启动后自动遍历数据、跑完 5 个场景 × 2 个模型，输出 `eval_result.csv`、`drift_curve.png`、
`metric_bar.png`。日志会打印每个组合的 F1、错误数与单张耗时。

## Docker

镜像不包含权重和数据集，两者在运行时挂载（或由 ultralytics 在运行时下载权重）：

```bash
docker build -t defect-eval .

mkdir -p output
docker run --rm \
  -e ZHIPU_API_KEY="$ZHIPU_API_KEY" \
  -e RESULT_CSV=/app/output/eval_result.csv \
  -e DRIFT_CURVE_PNG=/app/output/drift_curve.png \
  -e METRIC_BAR_PNG=/app/output/metric_bar.png \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/yolov8n.pt:/app/yolov8n.pt:ro" \
  -v "$(pwd)/output:/app/output" \
  defect-eval
```

- 容器以非 root 用户（uid 10001）运行，`/app/output` 需对当前用户可写。
- 镜像构建阶段不再执行 `YOLO('yolov8n.pt')`，避免构建时联网下载权重；未挂载权重时需要
  容器能访问外网。
- 图表中的中文标签需要镜像内存在中文字体（如 `fonts-noto-cjk`），否则中文显示为方块，
  不影响数值结果与 CSV。
- `.dockerignore` 已排除 `.git`、`*.pt`、`data/`、`*.png`、`*.csv`，镜像不会带入私料。

## 产物与复现

- `eval_result.csv`：每行一个（模型, 场景），列为准确率、精确率、召回率、F1、单张耗时、
  错误数、有效样本数、`baseline_note`（说明该模型作为基线的可比性限制）。
- `drift_curve.png` / `metric_bar.png`：由同一份结果 DataFrame 绘制。
- 重新生成：准备好 `data/`（见 `data/README.md`），配置 `ZHIPU_API_KEY` 后运行
  `python eval_main.py`，三个产物会被覆盖。历史文件记录了修复前的逻辑，复现前请留意版本差异。

## 数据来源与许可

- 原始数据为作者自采的同一只矿泉水瓶实拍（29 张瑕疵 / 15 张正常，共 44 张），已从版本库移除，
  不再分发；原因与准备方式见 `data/README.md`。
- 代码以仓库根目录的 `LICENSE` 为准；YOLOv8 权重遵循 Ultralytics 的许可（AGPL-3.0 或商业许可），
  GLM-4V 的使用遵循智谱开放平台的服务条款，商用前请自行确认。
- 仓库中的历史指标、图表是上述数据的产物，不代表任何第三方数据集的性能。

## 硬件基准

- `eval_result.csv` 中的单张耗时是原始运行的实测值（YOLOv8n 约 0.03~0.07 s/张，
  GLM-4V 约 2.7~12.7 s/张，随网络与服务负载波动），**测试硬件的具体型号未随运行记录保存**，
  绝对耗时不可跨机器比较，只能用于同一次运行内的相对比较。
- 复测时请记录 CPU/GPU 型号、torch 版本、图片分辨率，并在报告中注明，否则耗时数据不可复现。

## 已知限制

- 样本量小（44 张、单一物体、单一背景），指标置信区间很宽，不支持统计显著性结论。
- YOLOv8n 基线未针对瑕疵任务训练，召回率低是预期结果，不代表 YOLO 系列不适合质检。
- GLM-4V 的判定受提示词与温度影响，`temperature=0.1` 仍非完全确定；生产使用应固定提示词版本。
- 评测只覆盖亮度与对比度扰动，未覆盖模糊、噪声、遮挡、角度变化。
