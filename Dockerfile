FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

WORKDIR /app
COPY pyproject.toml README.md LICENSE requirements.lock ./
COPY src ./src
COPY configs ./configs
COPY analysis ./analysis
COPY results ./results
COPY tests ./tests
COPY train_distributed.py train_schedule_accounted.py ./
# PyTorch, torchvision and CUDA come from the base image.
RUN pip install --no-cache-dir -c requirements.lock -e ".[dev,analysis]"
