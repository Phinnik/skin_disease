FROM nvidia/cuda:11.3.0-devel-ubuntu20.04 as deploy

ENV TZ="Europe/Moscow"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update  \
    && apt upgrade -y  \
    && apt install -y --no-install-recommends software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y

RUN apt-get install -y --no-install-recommends python3.10 python3.10-dev python3.10-distutils python3.10-venv

RUN ln -s /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

RUN python -m ensurepip --upgrade && \
    pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 cache purge

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
COPY setup.py .
ARG RELEASE_VERSION
COPY models/model_releases/$RELEASE_VERSION/* models/model_releases/current
COPY src/ src/
RUN pip3 install --no-cache-dir -e .
WORKDIR src/app/