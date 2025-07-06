# 使用指定版本的 Python 3.10.18
FROM python:3.10.18-slim

# 安装系统依赖，支持 OpenCV、Pillow、ONNX 等
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY . /app

# 安装 Python 依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 设置默认启动命令（Streamlit）
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
