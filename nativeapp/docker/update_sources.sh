#!/bin/bash
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

. /etc/os-release

[[ "${NAME}" == "Ubuntu" ]] || exit 0

sed -i "s/deb\ /deb \[arch=amd64\]\ /g" /etc/apt/sources.list

cat <<EOT >> /etc/apt/sources.list
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports ${UBUNTU_CODENAME} main universe
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports ${UBUNTU_CODENAME}-updates main universe
deb [arch=arm64,armhf] http://ports.ubuntu.com/ubuntu-ports ${UBUNTU_CODENAME}-security main universe
EOT
