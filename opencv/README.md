# OpenCV camera examples with Coral

This folder contains example code using [OpenCV](https://github.com/opencv/opencv) to obtain
camera images and perform object detection on the Edge TPU.

This code works on Linux/macOS/Windows using a webcam, Raspberry Pi with the Pi Camera, and on the Coral Dev
Board using the Coral Camera or a webcam. For all settings other than the Coral Dev Board, you also need a Coral
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

4.  Install the OpenCV libraries:

    ```
    cd opencv

    bash install_requirements.sh
    ```


## Run the detection demo (SSD models)

```
python3 detect.py
```

By default, this uses the ```mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite``` model.

You can change the model and the labels file using flags ```--model``` and ```--labels```.
