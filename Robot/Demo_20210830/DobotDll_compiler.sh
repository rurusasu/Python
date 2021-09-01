#!/bin/bash
#DobotDll パッケージを Linux用にコンパイルする関数

# 必要なパッケージのインストール
# https://enjoysoftware.hatenablog.com/entry/2019/12/02/144425
sudo apt -y update &&
sudo apt -y upgrade &&
sudo apt -y install \
build-essential \
cmake \
qtbase5-dev \
qttools5-dev-tools \
qt5-default \
qtcreator \
libqt5serialport5 \
libqt5serialport5-dev

# DOBOT API共有ライブラリの作成
# https://physical-computing-lab.net/dobot/dobotapi-for-jetsonnano.html
qmake -o makefile DobotDll.pro
make