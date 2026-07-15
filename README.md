\# 工业零件瑕疵检测评测系统

\## 项目介绍

本项目搭建一套自动化评测框架，对比轻量目标检测模型YOLOv8与智谱GLM-4V多模态大模型在工业零件缺陷识别场景下的综合性能。

模拟工厂实际光照变化（亮度增减、对比度变化）模拟数据偏移，自动计算精度、速度指标并生成可视化图表，用于工业质检场景的模型选型对比。



\## 技术栈

Python、OpenCV、YOLOv8、智谱GLM-4V、NumPy、Pandas、Matplotlib



\## 项目目录

defect\_eval/

├── data/                # 数据集文件夹

│   ├── defect/          # 带瑕疵零件图片

│   └── normal/          # 无瑕疵正常零件图片

├── eval\_main.py         # 项目主运行代码

├── eval\_result.csv      # 全部评测数据表格

├── drift\_curve.png      # 光照变化模型性能折线图

├── metric\_bar.png       # YOLO与大模型指标对比柱状图

├── .gitignore           # Git忽略缓存配置

└── README.md            # 项目说明文档



\## 环境与运行方式

1\. 安装依赖

pip install ultralytics opencv-python zhipuai pandas matplotlib numpy

2\. 启动程序

python eval\_main.py

3\. 运行说明

启动后在终端输入你的智谱AI Key，程序自动遍历全部图片，5种光照场景分别完成两套模型推理，运行结束自动输出表格与两张效果图。



\## 核心功能

1\. 多模型横向对比：本地YOLOv8快速检测、云端多模态大模型精准识别

2\. 光照扰动模拟：亮度±30、对比度缩放，模拟车间复杂拍摄环境

3\. 自动化指标计算：自动统计准确率、召回率、F1值、单张推理耗时

4\. 可视化报表：自动生成折线图、柱状图，直观展示模型鲁棒性差异

