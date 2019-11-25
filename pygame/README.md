# P examples for inferencing on Coral

This folder contains example code using [pygame](https://github.com/pygame/pygame) to obtain
camera images and then perform image classification or object detection on the Edge TPU.

## Set up your device

1.  First, be sure you have completed the [setup instructions for your Coral
    device](https://coral.ai/docs/setup/).

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

4.  Install pygame:

    ```
    bash pygame/install_requirements.sh
    ```

5.  Navigate to this code:

    ```
    cd pygame
    ```



## Run the classification demo

```
python3 classify_capture.py
```

By default, this uses the ```mobilenet_v2_1.0_224_quant_edgetpu.tflite``` model.

You can change the model and the labels file using flags ```--model``` and ```--labels```.


## Run the detection demo (SSD models)

```
python3 detect.py
```

By default, this uses the ```mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite``` model.

You can change the model and the labels file using flags ```--model``` and ```--labels```.


