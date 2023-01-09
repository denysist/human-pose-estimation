FROM python:3.10

ARG workdir=/home/human-pose-estimation
WORKDIR $workdir

COPY requirements.txt ./
RUN apt-get update \
    && apt-get -y install ffmpeg libgl1 \
    && pip install --no-cache-dir -r requirements.txt \
    && git clone -b v3.0.0rc5 https://github.com/open-mmlab/mmdetection.git \
    && pip install --no-cache-dir mmdetection/ \
    && git clone -b v1.0.0rc0 https://github.com/open-mmlab/mmpose.git \
    && pip install --no-cache-dir mmpose/

CMD [ "bash" ]
