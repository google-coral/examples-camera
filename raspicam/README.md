# Raspberry Pi camera examples with Coral

This folder contains example code using [picamera](https://github.com/waveform80/picamera) to obtain
camera images and perform image classification on the Edge TPU.

This code works on Raspberry Pi with the Pi Camera and Coral USB Accelerator.


## Set up your device

1.  First, be sure you have completed the [setup instructions for the USB
    Accelerator](https://coral.ai/docs/accelerator/get-started/). If it's been a while, repeat to be sure
    you have the latest software.

    Importantly, you should have the latest TensorFlow Lite runtime installed
    (as per the [Python quickstart](
    https://www.tensorflow.org/lite/guide/python)). You can check which version is installed
    using the ```pip3 show tflite_runtime``` command.


2.  Follow the guide to [connect and configure the Pi Camera](
    https://www.raspberrypi.org/documentation/configuration/camera.md).

3.  Clone this Git repo onto your Raspberry Pi:

    ```
    mkdir google-coral && cd google-coral

    git clone https://github.com/google-coral/examples-camera --depth 1
    ```

4.  Download the models:

    ```
    cd examples-camera

    sh download_models.sh
    ```

5.  Install picamera:

    ```
    cd raspicam

    bash install_requirements.sh
    ```


## Run the classification demo

```
python3 classify_capture.py
```

By default, this uses the ```mobilenet_v2_1.0_224_quant_edgetpu.tflite``` model.

You can change the model and the labels file using flags ```--model``` and ```--labels```.


