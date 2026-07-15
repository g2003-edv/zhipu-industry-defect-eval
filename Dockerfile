FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN python -c "from ultralytics import YOLO;YOLO('yolov8n.pt')"
CMD ["python", "eval_main.py"]