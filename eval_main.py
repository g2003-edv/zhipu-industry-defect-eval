"""工业零件瑕疵检测评测：YOLOv8 基线 vs 智谱 GLM-4V。

对比两个模型在原始画面与光照扰动（亮度、对比度）下的瑕疵判定表现。
GLM-4V 要求结构化 JSON 输出，无法解析的回复单独计入 error，不计入 TN；
YOLOv8n 为未微调的 COCO 预训练检测器，仅作为基线，指标不可直接比较。

运行前需设置环境变量 ZHIPU_API_KEY（见 README）。
"""

import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")  # 无图形环境（Docker / 服务器）也能生成图片
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from zhipuai import ZhipuAI  # noqa: E402


# ---------------------------------------------------------------------------
# 配置（均可通过环境变量覆盖）
# ---------------------------------------------------------------------------
def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[警告] 环境变量 {name}={raw!r} 不是有效数字，改用默认值 {default}")
        return default


DATA_DIR = os.getenv("DATA_DIR", "./data")
RESULT_CSV = os.getenv("RESULT_CSV", "./eval_result.csv")
DRIFT_CURVE_PNG = os.getenv("DRIFT_CURVE_PNG", "./drift_curve.png")
METRIC_BAR_PNG = os.getenv("METRIC_BAR_PNG", "./metric_bar.png")
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "./yolov8n.pt")
GLM_MODEL = os.getenv("GLM_MODEL", "glm-4v-plus")
YOLO_CONF = env_float("YOLO_CONF", 0.3)
YOLO_MEAN_CONF = env_float("YOLO_MEAN_CONF", 0.7)

# 显卡可用性只能以 torch 为准（原实现用 os.path.exists("cuda")，永远为 CPU）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 说明各模型作为「基线」的可比性，随结果一起落盘，避免结论被误读
BASELINE_NOTES = {
    "YOLOv8n": (
        "未微调的 COCO 预训练检测器（无瑕疵类别），仅按检测框置信度均值判定，"
        "指标不可与 GLM-4V 直接比较"
    ),
    "GLM-4V智谱多模态": "通用多模态模型零样本提示，未在本数据集上微调",
}

DEFECT_PROMPT = (
    "你是工业质检员。判断图中零件表面是否存在划痕、破损、变形等工业缺陷。"
    "只输出一个 JSON 对象，不要输出任何解释："
    '存在瑕疵输出 {"defect": true}，不存在瑕疵输出 {"defect": false}。'
)

_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}")
# 否定表达优先匹配，避免“没有瑕疵”被“有瑕疵”误判为真
_NEGATIVE_PATTERNS = (
    "不存在瑕疵",
    "没有瑕疵",
    "未发现瑕疵",
    "未见瑕疵",
    "无瑕疵",
    "不存在缺陷",
    "没有缺陷",
    "未发现缺陷",
    "无缺陷",
    "未见异常",
    "无异常",
)
_POSITIVE_PATTERNS = (
    "存在瑕疵",
    "有瑕疵",
    "存在缺陷",
    "有缺陷",
    "存在划痕",
    "有划痕",
    "不合格",
    "不正常",
)

_client = None
_yolo_model = None
_glm_error_logged = 0


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
def get_client():
    """懒加载智谱客户端；只读环境变量，缺失时报错（不再交互式 getpass）。"""
    global _client
    if _client is None:
        api_key = os.getenv("ZHIPU_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "未设置环境变量 ZHIPU_API_KEY，请先配置智谱 API Key（见 README）"
            )
        _client = ZhipuAI(api_key=api_key)
    return _client


def get_yolo_model():
    """懒加载 YOLO，避免 import 阶段触发权重下载。"""
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(YOLO_WEIGHTS)
    return _yolo_model


# ---------------------------------------------------------------------------
# 图像扰动
# ---------------------------------------------------------------------------
def shift_brightness(img, delta):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # 转int避免负数溢出，再裁剪回0-255
    v = v.astype(np.int16) + delta
    v = np.clip(v, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


def shift_contrast(img, alpha):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


# ---------------------------------------------------------------------------
# GLM-4V 判定与解析
# ---------------------------------------------------------------------------
def _log_glm_error(exc: Exception) -> None:
    """错误按条计数；终端只打印前几条，避免 200+ 行刷屏。"""
    global _glm_error_logged
    if _glm_error_logged < 5:
        _glm_error_logged += 1
        print(f"[警告] GLM-4V 失败（计入 error）：{type(exc).__name__}: {exc}")


def _parse_text_fallback(text: str) -> Optional[bool]:
    """纯文本兜底：先排除否定表达，再匹配肯定表达；仍无法判断则返回 None。"""
    for phrase in _NEGATIVE_PATTERNS:
        if phrase in text:
            return False
    for phrase in _POSITIVE_PATTERNS:
        if phrase in text:
            return True
    return None


def parse_defect_answer(text) -> Optional[bool]:
    """解析模型回复。

    返回 True（有瑕疵）/ False（无瑕疵）/ None（无法解析，调用方计入 error）。
    优先解析 JSON 的 defect 字段，失败时回退到带否定词排除的文本匹配。
    """
    if not isinstance(text, str) or not text.strip():
        return None

    cleaned = re.sub(r"```(?:json|JSON)?", "", text.strip()).replace("```", "").strip()

    for candidate in _JSON_OBJECT_RE.finditer(cleaned):
        try:
            payload = json.loads(candidate.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or "defect" not in payload:
            continue
        value = payload["defect"]
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "有", "是"}:
                return True
            if lowered in {"false", "0", "no", "n", "无", "否"}:
                return False

    return _parse_text_fallback(cleaned)


def glm4v_detect(img_cv) -> Optional[bool]:
    """GLM-4V 瑕疵判定：True=有瑕疵，False=无瑕疵，None=调用或解析失败。"""
    try:
        ok, buf = cv2.imencode(".jpg", img_cv)
        if not ok:
            print("[警告] 图像编码失败，计入 error")
            return None
        img_b64 = base64.b64encode(buf).decode()
        res = get_client().chat.completions.create(
            model=GLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                        {"type": "text", "text": DEFECT_PROMPT},
                    ],
                }
            ],
            temperature=0.1,
        )
        answer = res.choices[0].message.content
    except Exception as exc:  # noqa: BLE001 - 单张图片失败不应中断整轮评测
        _log_glm_error(exc)
        return None

    result = parse_defect_answer(answer)
    if result is None:
        _log_glm_error(
            ValueError(f"无法解析模型回复（计入 error）：{str(answer)[:80]!r}")
        )
    return result


# ---------------------------------------------------------------------------
# YOLO 基线
# ---------------------------------------------------------------------------
def yolo_detect(img_cv) -> bool:
    """未微调基线：取所有检测框置信度均值，超过阈值即判为有瑕疵。"""
    res = get_yolo_model()(img_cv, conf=YOLO_CONF, device=DEVICE, verbose=False)
    boxes = res[0].boxes if len(res) else None
    confs = [] if boxes is None else [float(box.conf.item()) for box in boxes]
    if not confs:
        return False
    return float(np.mean(confs)) > YOLO_MEAN_CONF


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------
@dataclass
class Counts:
    """单个（模型, 场景）组合的混淆计数；error 不计入混淆矩阵。"""

    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    error: int = 0
    elapsed: float = 0.0
    count: int = 0


def compute_metrics(counts: Counts):
    valid = counts.tp + counts.tn + counts.fp + counts.fn
    acc = (counts.tp + counts.tn) / valid if valid else 0.0
    prec = counts.tp / (counts.tp + counts.fp) if (counts.tp + counts.fp) else 0.0
    recall = counts.tp / (counts.tp + counts.fn) if (counts.tp + counts.fn) else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0
    avg_time = counts.elapsed / counts.count if counts.count else 0.0
    return acc, prec, recall, f1, avg_time, valid


def _run_files(files, label: bool, drift_func, model_func, counts: Counts) -> None:
    for path in files:
        img = cv2.imread(path)
        if img is None:
            counts.error += 1
            print(f"[警告] 图片无法读取，计入 error：{path}")
            continue
        if drift_func:
            img = drift_func(img)
        t1 = time.perf_counter()
        pred = model_func(img)
        counts.elapsed += time.perf_counter() - t1
        counts.count += 1
        if pred is None:
            counts.error += 1
        elif label:
            counts.tp += int(bool(pred))
            counts.fn += int(not pred)
        else:
            counts.fp += int(bool(pred))
            counts.tn += int(not pred)


def _list_images(folder: str):
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def run_benchmark():
    defect_files = _list_images(os.path.join(DATA_DIR, "defect"))
    normal_files = _list_images(os.path.join(DATA_DIR, "normal"))
    print(f"瑕疵样本：{len(defect_files)} 张，正常样本：{len(normal_files)} 张")

    drift_scenes = [
        ("原始原图", None),
        ("亮度-30", lambda x: shift_brightness(x, -30)),
        ("亮度+30", lambda x: shift_brightness(x, 30)),
        ("对比度0.7", lambda x: shift_contrast(x, 0.7)),
        ("对比度1.3", lambda x: shift_contrast(x, 1.3)),
    ]
    models = [("YOLOv8n", yolo_detect), ("GLM-4V智谱多模态", glm4v_detect)]
    all_result = []

    for scene_name, drift_func in drift_scenes:
        for model_name, model_func in models:
            counts = Counts()
            _run_files(defect_files, True, drift_func, model_func, counts)
            _run_files(normal_files, False, drift_func, model_func, counts)

            acc, prec, recall, f1, avg_time, valid = compute_metrics(counts)
            all_result.append(
                {
                    "模型": model_name,
                    "数据场景": scene_name,
                    "准确率": round(acc, 3),
                    "精确率": round(prec, 3),
                    "召回率": round(recall, 3),
                    "F1分数": round(f1, 3),
                    "单张推理耗时(s)": round(avg_time, 4),
                    "错误数": counts.error,
                    "有效样本数": valid,
                    "baseline_note": BASELINE_NOTES[model_name],
                }
            )
            print(
                f"【{model_name} | {scene_name}】 F1={f1:.3f}, "
                f"错误数={counts.error}, 有效样本={valid}, 单张耗时={avg_time:.4f}s"
            )

    df = pd.DataFrame(all_result)
    df.to_csv(RESULT_CSV, encoding="utf-8-sig", index=False)
    print(f"评测指标表格已保存：{RESULT_CSV}")
    return df


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
def plot_drift_curve(df):
    if df.empty:
        print("[警告] 结果为空，跳过衰减曲线绘制")
        return
    plt.figure(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e"]
    markers = ["o", "s"]
    model_list = list(df["模型"].unique())
    for idx, m in enumerate(model_list):
        data = df[df["模型"] == m]
        x = list(range(len(data)))
        y = data["F1分数"].values
        plt.plot(
            x,
            y,
            label=m,
            color=colors[idx % len(colors)],
            marker=markers[idx % len(markers)],
            linewidth=2,
        )
        for xi, yi in zip(x, y):
            plt.annotate(
                f"{yi:.2f}",
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
            )
    scene_names = list(df["数据场景"].unique())
    plt.xticks(list(range(len(scene_names))), scene_names)
    plt.xlabel("数据扰动场景", fontsize=12)
    plt.ylabel("F1分数", fontsize=12)
    plt.title("数据漂移下模型性能衰减对比", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(DRIFT_CURVE_PNG, dpi=150)
    plt.close()
    print(f"衰减曲线图已保存 {DRIFT_CURVE_PNG}")


def plot_metric_bar(df):
    raw_data = df[df["数据场景"] == "原始原图"]
    if raw_data.empty:
        print("[警告] 缺少原始场景结果，跳过指标柱状图绘制")
        return
    metrics = ["准确率", "精确率", "召回率", "F1分数"]
    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (_, row) in enumerate(raw_data.iterrows()):
        vals = [row[m] for m in metrics]
        ax.bar(x + i * w - w / 2, vals, w, label=row["模型"])
    ax.set_ylabel("指标分数")
    ax.set_title("原始数据双模型核心指标对比")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(METRIC_BAR_PNG, dpi=150)
    plt.close()
    print(f"指标柱状图已保存 {METRIC_BAR_PNG}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main() -> int:
    print("=====工业瑕疵模型选型评测平台=====")

    if not os.getenv("ZHIPU_API_KEY", "").strip():
        print("[错误] 未设置环境变量 ZHIPU_API_KEY。", file=sys.stderr)
        print(
            "       请先 export ZHIPU_API_KEY=<your-key>（Docker 用 -e 传入）后重试。",
            file=sys.stderr,
        )
        return 2

    for sub in ("defect", "normal"):
        folder = os.path.join(DATA_DIR, sub)
        if not os.path.isdir(folder):
            print(f"[错误] 缺少数据集目录：{folder}", file=sys.stderr)
            print("       数据不随仓库分发，请按 data/README.md 自行准备。", file=sys.stderr)
            return 1

    print(f"[环境] device={DEVICE}（torch.cuda.is_available()={torch.cuda.is_available()}）")
    print(f"[环境] YOLO 权重={YOLO_WEIGHTS}，云端模型={GLM_MODEL}")

    try:
        get_yolo_model()
    except Exception as exc:  # noqa: BLE001 - 权重缺失/下载失败给出明确提示
        print(f"[错误] 加载 YOLO 权重失败：{YOLO_WEIGHTS}（{type(exc).__name__}: {exc}）", file=sys.stderr)
        print("       可把权重挂载/下载到该路径，或用环境变量 YOLO_WEIGHTS 指定。", file=sys.stderr)
        return 1

    try:
        get_client()
    except RuntimeError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 2

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    res_df = run_benchmark()
    print("\n=====评测结果汇总=====")
    print(res_df.to_string(index=False))
    print("\n正在生成可视化图表...")
    plot_drift_curve(res_df)
    plot_metric_bar(res_df)
    print("\n全部评测完成！目录下生成csv与两张对比图表")
    return 0


if __name__ == "__main__":
    sys.exit(main())
