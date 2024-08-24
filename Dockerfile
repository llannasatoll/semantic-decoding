FROM nvidia/cuda:12.1.0-runtime-ubuntu18.04

# User setting
ARG user
ARG group
ARG uid
ARG gid

# Install required packages
RUN groupadd -g $gid $group && useradd -m -s /bin/bash -u $uid -g $gid $user
# USER $user

RUN apt-get update && apt-get install -y \
    cuda-libraries-12-1 \
    cuda-libraries-dev-12-1 \
    cuda-tools-12-1 \
    libcublas-12-1 \
    libcublas-dev-12-1 \
    build-essential \
    cuda-toolkit-12-1

WORKDIR /home/$user/semantic-decoding
