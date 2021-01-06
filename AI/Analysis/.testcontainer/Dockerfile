#-------------------------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See https://go.microsoft.com/fwlink/?linkid=2090316 for license information.
#-------------------------------------------------------------------------------------------------------------

# base image
FROM continuumio/miniconda3

USER root

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    locales \
    && localedef -f UTF-8 -i ja_JP ja_JP.UTF-8 \
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN conda update -y --all \
    && conda install -y anaconda-client \
    && conda clean --all

RUN conda env create -f ./env/myenv.yml