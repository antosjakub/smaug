
# 1. build the image once
# docker build -t smaug-img .

FROM firedrakeproject/firedrake:latest

USER root

RUN apt-get update && \
    apt-get install -y ffmpeg nodejs npm && \
    rm -rf /var/lib/apt/lists/*

RUN npm install -g pyright

RUN echo "alias c='clear'" >> /etc/bash.bashrc && \
    echo "alias python='python3'" >> /etc/bash.bashrc

#USER firedrake


