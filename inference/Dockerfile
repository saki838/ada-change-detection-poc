FROM python:3.13-slim

# System libs for rasterio (GDAL) + opencv headless runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        gdal-bin libgdal-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache the torchvision ResNet34 ImageNet encoder weights at build time
# (non-China, download.pytorch.org) so the first request is fast and offline-safe.
# app/weights/resnet34-b627a593.pth (if COPYed below) is preferred by model.py;
# this cache is the belt-and-suspenders fallback smp resolves through torchvision.
RUN python -c "import torchvision.models as m; m.resnet34(weights=m.ResNet34_Weights.IMAGENET1K_V1)"

COPY app ./app
COPY models ./models

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
