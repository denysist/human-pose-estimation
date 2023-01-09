FROM python:3.10

ARG workdir=/home/human-pose-estimation
WORKDIR $workdir

COPY docker/production_api/requirements.txt ./
RUN apt-get update \
    && apt-get -y install ffmpeg libgl1 \
    && pip install --no-cache-dir -r requirements.txt \
    && git clone -b v3.0.0rc5 https://github.com/open-mmlab/mmdetection.git \
    && pip install --no-cache-dir mmdetection/ \
    && git clone -b v1.0.0rc0 https://github.com/open-mmlab/mmpose.git \
    && pip install --no-cache-dir mmpose/

COPY checkpoints/ ./checkpoints/
COPY configs/ ./configs/
COPY library/ ./library/
COPY api/app.py ./

EXPOSE 8000

CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000" ]
