FROM jupyter/scipy-notebook

# Set Locale
#RUN echo "LC_ALL=ja_JP.UTF-8" >> /etc/environment \
#          && echo "ja_JP.UTF-8 UTF-8" >> /etc/locale.gen \
#          && echo "LANG=ja_JP.UTF-8" > /etc/locale.conf \
#          && locale-gen ja_JP.UTF-8

# 環境をアップデートするために、Environment.ymlを一時的な場所にコピーします（見つかった場合）。
# また、"noop.txt "をコピーして、環境.ymlが存在しない場合にCOPY命令が失敗しないようにします。
#COPY ./environment.yml ./noop.txt /tmp/conda-tmp/
#RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
#  && rm -rf /tmp/conda-tmp

WORKDIR /home
USER jovyan