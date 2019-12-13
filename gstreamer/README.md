# GStreamer camera examples with Coral

This folder contains example code using [GStreamer](https://github.com/GStreamer/gstreamer) to
obtain camera images and perform image classification and object detection on the Edge TPU.

This code works on Linux using a webcam, Raspberry Pi with the Pi Camera, and on the Coral Dev
Board using the Coral Camera or a webcam. For the first two, you also need a Coral
USB/PCIe/M.2 Accelerator.


## Set up your device

1.  First, be sure you have completed the [setup instructions for your Coral
    device](https://coral.ai/docs/setup/). If it's been a while, repeat to be sure
    you have the latest software.

    Importantly, you should have the latest TensorFlow Lite runtime installed
    (as per the [Python quickstart](
    https://www.tensorflow.org/lite/guide/python)). You can check which version is installed
    using the ```pip3 show tflite_runtime``` command.

2.  Clone this Git repo onto your computer or Dev Board:

    ```
    mkdir google-coral && cd google-coral

    git clone https://github.com/google-coral/examples-camera --depth 1
    ```

3.  Download the models:

    ```
    cd examples-camera

    sh download_models.sh
    ```

4.  Install the GStreamer libraries (if you're using the Coral Dev Board, you can skip this):

    ```
    cd gstreamer

    bash install_requirements.sh
    ```


## Run the classification demo

```
python3 classify.py
```

By default, this uses the ```mobilenet_v2_1.0_224_quant_edgetpu.tflite``` model.

You can change the model and the labels file using flags ```--model``` and ```--labels```.


## Run the detection demo (SSD models)

```
python3 detect.py
```

Likewise, you can change the model and the labels file using ```--model``` and ```--labels```.

By default, both examples use the attached Coral Camera. If you want to use a USB camera,
edit the ```gstreamer.py``` file and change ```device=/dev/video0``` to ```device=/dev/video1```.

