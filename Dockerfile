FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# opencv-python 运行时依赖（libGL / libglib）
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依赖单独一层：只改代码时不会重复安装依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 非 root 运行
RUN useradd --create-home --uid 10001 appuser

COPY --chown=appuser:appuser eval_main.py ./
RUN mkdir -p /app/data /app/output && chown -R appuser:appuser /app

USER appuser

# 权重与数据在运行时挂载/下载，不打包进镜像（见 README）
CMD ["python", "eval_main.py"]
