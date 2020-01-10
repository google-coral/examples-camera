# Native C++ Example using Gstreamer

This directory contains a C++ Gstreamer example to work with the Coral [Dev Board](https://coral.withgoogle.com/products/dev-board/).
The app itself `mendelcam` is built natively on the device. It also requires
the TfLite static library which is easiest built on the host computer, even
though it is possible with some modifications to build it on the target device.

## Prepare the device and copy the source code to the device

1. If you haven't done so already, set up your board according to [Get started with the Dev Board](
https://coral.ai/docs/dev-board/get-started).

1. Run `mdt push * nativeapp` to the source and Makefile to the device

1. Run `mdt shell` to connect to the device

1. On the device run:
```
sudo apt-get install libglib2.0-dev
sudo apt-get install libgstreamer1.0-dev
sudo apt-get install libedgetpu-dev
```

## Build and install libtensorflow-lite.a
The inference code requires to link with the TensorFlow Lite library. The
library needs to be compiled for the ARM architecture of the Dev Board and
to speed things up it's easiest to do cross complie it on the host computer and
to simplify these instructions we then copy the whole tensorflow tree over to
the device.

On the host computer run the below commands. To find the git commit to sync to
look for __TENSORFLOW_COMMIT__  in this [WORKSPACE](https://github.com/google-coral/edgetpu/blob/master/WORKSPACE) file. (For correct behavior it's __important__
that the tensorflow version matches that of libedgetpu.so).

On the host run:

```
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git checkout $TENSORFLOW_COMMIT
sudo apt-get update
sudo apt-get install crossbuild-essential-arm64
./tensorflow/lite/tools/make/download_dependencies.sh
./tensorflow/lite/tools/make/build_aarch64_lib.sh
```
When the cross build is done, run:
```
cd ..
mdt push tensorflow # Takes a while
```
(Even though it is not recommended, it is possible to build tensorflow natively
on the target, but it takes a long time and in some cases requires
`build_aarch64_lib.sh` to be modified by removing "-j 4" from the make command)

## Build and run the application
Connect to the target device:
`mdt shell`
On the device:
```
cd nativeapp
export TENSORFLOW_DIR=~/tensorflow
make
```
If all goes well, there should now be a `mendelcam` file in the directory. To
run the application you need to download a model and label file from [here](https://coral.ai/models/).
Below we use MobileNet V2(ImageNet). Copy these to the device by running this
on the host:
```
mdt push imagenet_labels.txt
mdt push mobilenet_v2_1.0_224_quant_edgetpu.tflite
```
Now, in the nativeapp directory on the target (don't forget `mdt shell`) run:
```
./mendelcam --model ~/mobilenet_v2_1.0_224_quant_edgetpu.tflite  --labels ~/imagenet_labels.txt
```
