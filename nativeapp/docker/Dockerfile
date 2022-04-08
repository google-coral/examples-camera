# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG IMAGE
FROM ${IMAGE}

COPY update_sources.sh /
RUN /update_sources.sh

RUN dpkg --add-architecture armhf
RUN dpkg --add-architecture arm64
RUN apt-get update && apt-get install -y \
  sudo \
  debhelper \
  build-essential \
  crossbuild-essential-armhf \
  crossbuild-essential-arm64 \
  libusb-1.0-0-dev \
  libusb-1.0-0-dev:arm64 \
  libusb-1.0-0-dev:armhf \
  libglib2.0-dev \
  libglib2.0-dev:armhf \
  libglib2.0-dev:arm64 \
  libgstreamer1.0-dev \
  libgstreamer1.0-dev:armhf \
  libgstreamer1.0-dev:arm64 \
  libgstreamer-plugins-base1.0-dev \
  libgstreamer-plugins-base1.0-dev:armhf \
  libgstreamer-plugins-base1.0-dev:arm64 \
  libgtk-3-dev \
  libgtk-3-dev:arm64 \
  libgtk-3-dev:armhf \
  python \
  python3-all \
  python3-six \
  zlib1g-dev \
  zlib1g-dev:armhf \
  zlib1g-dev:arm64 \
  pkg-config \
  zip \
  unzip \
  curl \
  wget \
  git \
  subversion \
  vim \
  python3-numpy

ARG BAZEL_VERSION=2.1.0
RUN wget -O /bazel https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    bash /bazel && \
    rm -f /bazel

RUN svn export https://github.com/google-coral/edgetpu/trunk/libedgetpu /libedgetpu

