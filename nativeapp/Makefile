# Makefile for building EnterpriseCameraObjectDetection natively on
# mendel
# Requires a tensorflow tree built for aarch64_armv8-a either natively
# or cross-compiled

TENSORFLOW_DIR ?= # This must be set externally

.PHONY: all clean

LIBS :=	-lrt \
	-ldl \
	-lpthread \
	-lgstreamer-1.0 \
	-lglib-2.0 \
	-lgobject-2.0 \
	-ledgetpu

INCL :=	-I/usr/lib/aarch64-linux-gnu/glib-2.0/include \
	-I/usr/include/gstreamer-1.0 \
	-I/usr/include/glib-2.0 \
	-I$(TENSORFLOW_DIR) \
	-I$(TENSORFLOW_DIR)/tensorflow/lite/tools/make/downloads/flatbuffers/include \

TFLITE_OBJS := $(TENSORFLOW_DIR)/tensorflow/lite/tools/make/gen/aarch64_armv8-a/lib/libtensorflow-lite.a

all: mendelcam

mendelcam:\
	camerastreamer.h \
	camerastreamer.cc \
	inferencewrapper.h \
	inferencewrapper.cc \
	main.cc
ifndef TENSORFLOW_DIR
	$(error TENSORFLOW_DIR is not defined)
endif
	g++  -o "$@" main.cc camerastreamer.cc inferencewrapper.cc $(TFLITE_OBJS) -std=c++11 -O3 -Wall $(INCL) $(LIBS)

clean:
	rm -rf mendelcam *.o
