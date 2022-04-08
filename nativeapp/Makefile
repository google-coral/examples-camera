# Copyright 2022 Google LLC
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
SHELL := /bin/bash
MAKEFILE_DIR := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))
OS := $(shell uname -s)

# Allowed CPU values: k8, armv7a, aarch64
ifeq ($(OS),Linux)
CPU ?= k8
else
$(error $(OS) is not supported)
endif
ifeq ($(filter $(CPU),k8 armv7a aarch64),)
$(error CPU must be k8, armv7a, aarch64)
endif

# Allowed COMPILATION_MODE values: opt, dbg
COMPILATION_MODE ?= opt
ifeq ($(filter $(COMPILATION_MODE),opt dbg),)
$(error COMPILATION_MODE must be opt or dbg)
endif

BAZEL_OUT_DIR :=  $(MAKEFILE_DIR)/bazel-out/$(CPU)-$(COMPILATION_MODE)/bin
BAZEL_BUILD_FLAGS := --crosstool_top=@crosstool//:toolchains \
                     --compiler=gcc \
                     --compilation_mode=$(COMPILATION_MODE) \
                     --copt=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
                     --copt=-std=c++14 \
                     --verbose_failures \
                     --cpu=$(CPU) \
                     --experimental_repo_remote_exec

ifeq ($(COMPILATION_MODE), opt)
BAZEL_BUILD_FLAGS += --linkopt=-Wl,--strip-all
else ifeq ($(COMPILATION_MODE), dbg)
# for now, disable arm_neon in dbg.
# see: https://github.com/tensorflow/tensorflow/issues/33360
BAZEL_BUILD_FLAGS += --cxxopt -DTF_LITE_DISABLE_X86_NEON
endif

ifeq ($(CPU),k8)
BAZEL_BUILD_FLAGS += --copt=-includeglibc_compat.h
else ifeq ($(CPU),aarch64)
BAZEL_BUILD_FLAGS += --copt=-ffp-contract=off
else ifeq ($(CPU),armv7a)
BAZEL_BUILD_FLAGS += --copt=-ffp-contract=off
endif

DEMO_OUT_DIR    := $(MAKEFILE_DIR)/out/$(CPU)/demo

mendelcam:
	bazel build $(BAZEL_BUILD_FLAGS) //src:mendelcam
	mkdir -p $(DEMO_OUT_DIR)
	cp -f $(BAZEL_OUT_DIR)/src/mendelcam \
              $(DEMO_OUT_DIR)

clean:
	rm -rf $(MAKEFILE_DIR)/bazel-* \
	       $(MAKEFILE_DIR)/out \

DOCKER_WORKSPACE=$(MAKEFILE_DIR)
DOCKER_CPUS=k8 armv7a aarch64 armv6
DOCKER_TAG_BASE=native-app
include $(MAKEFILE_DIR)/docker/docker.mk
