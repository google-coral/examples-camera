# Native C++ Example using Gstreamer

This directory contains a C++ Gstreamer example for classification model to work with the [Coral Dev Board](https://coral.withgoogle.com/products/dev-board/).
The app itself `mendelcam` is built from a provided docker image that fetches all required dependencies and build it.
The docker build was tested on the Dev Board, however, it can can support k8, aach64, and armv7a.

## Prepare the device and copy the source code to the device

1. If you haven't done so already, set up your board according to [Get started with the Dev Board](https://coral.ai/docs/dev-board/get-started).

2. Cross-compile for aarch64 from your Linux desktop (do not run this on the Dev Board):
```
cd nativeapp
make DOCKER_TARGETS=mendelcam DOCKER_CPUS=aarch64 docker-build
```

3. Copy models and binaries to the board:
First download model:
```
../download_models.sh
```
Then copy models and binaries to the board:
```
mdt push imagenet_labels.txt
mdt push mobilenet_v2_1.0_224_quant_edgetpu.tflite
mdt push out/aarch64/demo/mendelcam
```

3. Connect to the device:
```
mdt shell
```

4. To run:
In the nativeapp directory on the target (don't forget `mdt shell`) run:
```
./mendelcam --model ~/mobilenet_v2_1.0_224_quant_edgetpu.tflite  --labels ~/imagenet_labels.txt
```

## Troubleshooting:

Resolve permission error:
```
mendel@neat-yarn:~$ ./mendelcam --model ~/mobilenet_v2_1.0_224_quant_edgetpu.tflite  --labels ~/imagenet_labels.txt
-bash: ./mendelcam: Permission denied
mendel@neat-yarn:~$ sudo chmod 770 mendelcam 
```

How to use with USB camera:

To use with USB camera build the app by changing viderosrc `/dev/video0` to  `/dev/video-<usb-camera-source>` [here](src/main.cc#L18)
