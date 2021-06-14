FROM ubuntu:focal
#FROM python:3.8.10-slim-buster

RUN echo "nameserver 8.8.8.8" >> sudo tee /etc/resolv.conf \
  && apt-get update \
  && apt-get install -q -y \
  curl \
  xz-utils \
  #&& apt-get upgrade -y \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# --------------------------------- #
# Blender をインストール
# --------------------------------- #
# ssh証明証の発行を無効化して接続
RUN curl -ksL https://ftp.nluug.nl/pub/graphics/blender/release/Blender2.93/blender-2.93.0-linux-x64.tar.xz -o /usr/src/blender.tar.xz \
  && tar -Jxvf /usr/src/blender.tar.xz -C /usr/src

# ---------------------------------------- #
# Blender に pip をインストール
# ---------------------------------------- #
COPY .devcontainer/requirements.txt /tmp/
RUN cd /usr/src/blender-2.93.0-stable+blender-v293-release.84da05a8b806-linux.x86_64-release/2.93/python/bin \
  && curl -ksL https://bootstrap.pypa.io/get-pip.py -o ./get-pip.py \
  && ./python3.9 ./get-pip.py \
  && ./python3.9 pip install -r /tmp/requirements.txt