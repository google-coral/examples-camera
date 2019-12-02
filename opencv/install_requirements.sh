#!/bin/bash
#
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if grep -s -q "MX8MQ" /sys/firmware/devicetree/base/model; then
  MENDEL_VER="$(cat /etc/mendel_version)"
  if [[ "$MENDEL_VER" == "1.0" || "$MENDEL_VER" == "2.0" || "$MENDEL_VER" == "3.0" ]]; then
    echo "Your version of Mendel is not compatible with OpenCV."
    echo "You must upgrade to Mendel 4.0 or higher."
    exit 1
  fi
fi

sudo pip3 install opencv-contrib-python
sudo apt-get -y install libjasper1 libhdf5-100 libqtgui4 libatlas-base-dev libqt4-test
sudo apt install python3-opencv