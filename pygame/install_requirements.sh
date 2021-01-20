#!/bin/bash
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if grep -s -q "Mendel" /etc/os-release; then
  echo "Installing DevBoard specific dependencies"
  sudo apt-get install -y python3-pygame
else
  sudo apt-get install -y libsdl-image1.2-dev libsdl-ttf2.0-dev libatlas-base-dev
  sudo pip3 install pygame
fi

