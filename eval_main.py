import cv2
import base64
import time
import os
import getpass
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from zhipuai import ZhipuAI
from ultralytics import YOLO

# 全局覆盖torch.load，强制weights_only=False，一次性解决所有pickle拦截
original_torch_load = torch.load
def fixed_torch_load(path, map_location=None, **kwargs):
    kwargs["weights_only"] = False
    return original_torch_load(path, map_location,** kwargs)
torch.load = fixed_torch_load

DATA_DIR = "./data"
RESULT_CSV = "./eval_result.csv"

print("请输入你的智谱AI API Key：")
ZHIPU_KEY = getpass.getpass("Key：")
zhipu_client = ZhipuAI(api_key=ZHIPU_KEY)

device = "cuda" if os.path.exists("cuda") else "cpu"
yolo_model = YOLO("yolov8n.pt")

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def shift_brightness(img, delta):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # 转int避免负数溢出，再裁剪回0-255
    v = v.astype(np.int16) + delta
    v = np.clip(v, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


def shift_contrast(img, alpha):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)


def glm4v_detect(img_cv):
    try:
        _, buf = cv2.imencode(".jpg", img_cv)
        img_b64 = base64.b64encode(buf).decode()
        res = zhipu_client.chat.completions.create(
            model="glm-4v",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                        {"type": "text", "text": "只回答【有瑕疵】或【无瑕疵】，判断零件表面是否存在划痕、破损、变形等工业缺陷"}
                    ]
                }
            ],
            temperature=0.1
        )
        ans = res.choices[0].message.content
        return "有瑕疵" in ans
    except Exception:
        return False


def yolo_detect(img_cv):
    res = yolo_model(img_cv, conf=0.3)
    confs = [box.conf.item() for box in res[0].boxes]
    if not confs:
        return False
    return np.mean(confs) > 0.7


def run_benchmark():
    defect_path = os.path.join(DATA_DIR, "defect")
    normal_path = os.path.join(DATA_DIR, "normal")
    defect_files = [os.path.join(defect_path, f) for f in os.listdir(defect_path) if f.endswith(".jpg")]
    normal_files = [os.path.join(normal_path, f) for f in os.listdir(normal_path) if f.endswith(".jpg")]
    print(f"瑕疵样本：{len(defect_files)} 张，正常样本：{len(normal_files)} 张")

    drift_scenes = [
        ("原始原图", None),
        ("亮度-30", lambda x: shift_brightness(x, -30)),
        ("亮度+30", lambda x: shift_brightness(x, 30)),
        ("对比度0.7", lambda x: shift_contrast(x, 0.7)),
        ("对比度1.3", lambda x: shift_contrast(x, 1.3))
    ]
    models = [("YOLOv8n", yolo_detect), ("GLM-4V智谱多模态", glm4v_detect)]
    all_result = []

    for scene_name, drift_func in drift_scenes:
        for model_name, model_func in models:
            tp = tn = fp = fn = 0
            total_cost = 0
            total_cnt = 0
            for f in defect_files:
                img = cv2.imread(f)
                if img is None:
                    continue
                if drift_func:
                    img = drift_func(img)
                t1 = time.time()
                pred = model_func(img)
                total_cost += time.time() - t1
                total_cnt += 1
                if pred:
                    tp += 1
                else:
                    fn += 1
            for f in normal_files:
                img = cv2.imread(f)
                if img is None:
                    continue
                if drift_func:
                    img = drift_func(img)
                t1 = time.time()
                pred = model_func(img)
                total_cost += time.time() - t1
                total_cnt += 1
                if pred:
                    fp += 1
                else:
                    tn += 1
            total = tp + tn + fp + fn
            acc = (tp + tn) / total if total else 0
            prec = tp / (tp + fp) if (tp + fp) else 0
            recall = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0
            avg_time = total_cost / total_cnt if total_cnt else 0
            all_result.append({
                "模型": model_name,
                "数据场景": scene_name,
                "准确率": round(acc, 3),
                "精确率": round(prec, 3),
                "召回率": round(recall, 3),
                "F1分数": round(f1, 3),
                "单张推理耗时(s)": round(avg_time, 4)
            })
            print(f"【{model_name} | {scene_name}】 F1={f1:.3f}, 单张耗时={avg_time:.4f}s")
    df = pd.DataFrame(all_result)
    df.to_csv(RESULT_CSV, encoding="utf-8-sig", index=False)
    print(f"评测指标表格已保存：{RESULT_CSV}")
    return df


def plot_drift_curve(df):
    plt.figure(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e"]
    markers = ["o", "s"]
    model_list = df["模型"].unique()
    for idx, m in enumerate(model_list):
        data = df[df["模型"] == m]
        x = list(range(len(data)))
        y = data["F1分数"].values
        plt.plot(x, y, label=m, color=colors[idx], marker=markers[idx], linewidth=2)
        for xi, yi in zip(x, y):
            plt.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center")
    plt.xticks(list(range(len(df["数据场景"].unique()))), df["数据场景"].unique())
    plt.xlabel("数据扰动场景", fontsize=12)
    plt.ylabel("F1分数", fontsize=12)
    plt.title("数据漂移下模型性能衰减对比", fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("drift_curve.png", dpi=150)
    print("衰减曲线图已保存 drift_curve.png")


def plot_metric_bar(df):
    raw_data = df[df["数据场景"] == "原始原图"]
    metrics = ["准确率", "精确率", "召回率", "F1分数"]
    x = np.arange(len(metrics))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, row in raw_data.iterrows():
        vals = [row[m] for m in metrics]
        ax.bar(x + i * w - w / 2, vals, w, label=row["模型"])
    ax.set_ylabel("指标分数")
    ax.set_title("原始数据双模型核心指标对比")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig("metric_bar.png", dpi=150)
    print("指标柱状图已保存 metric_bar.png")


if __name__ == "__main__":
    print("=====工业瑕疵模型选型评测平台=====")
    res_df = run_benchmark()
    print("\n=====评测结果汇总=====")
    print(res_df)
    print("\n正在生成可视化图表...")
    plot_drift_curve(res_df)
    plot_metric_bar(res_df)
    print("\n全部评测完成！目录下生成csv与两张对比图表")