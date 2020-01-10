# Edge TPU simple camera examples

This repo contains a collection of examples that use camera streams
together with the [TensorFlow Lite API](https://tensorflow.org/lite) with a
Coral device such as the
[USB Accelerator](https://coral.withgoogle.com/products/accelerator) or
[Dev Board](https://coral.withgoogle.com/products/dev-board).

## Installation

1.  First, be sure you have completed the [setup instructions for your Coral
    device](https://coral.ai/docs/setup/). If it's been a while, repeat to be sure
    you have the latest software.

    Importantly, you should have the latest TensorFlow Lite runtime installed
    (as per the [Python quickstart](
    https://www.tensorflow.org/lite/guide/python)).

2.  Clone this Git repo onto your computer:

    ```
    mkdir google-coral && cd google-coral

    git clone https://github.com/google-coral/examples-camera.git --depth 1
    ```

3.  Download the models:

    ```
    cd examples-camera

    sh download_models.sh
    ```

    These canned models will be downloaded and extracted to a new folder
    ```all_models```.


Further requirements may be needed by the different camera libraries, check the
README file for the respective subfolder.

## Contents

  * __Gstreamer__ Python examples using gstreamer to obtain camera images. These
    examples work on Linux using a webcam, Raspberry Pi with
    the Raspicam and on the Coral DevBoard using the Coral camera. For the
    former two you will also need a Coral USB Accelerator to run the models.
  * __Raspicam__ Python example using picamera. This is only intended for
    Raspberry Pi and will require a Coral USB Accelerator.
    Use ```install_requirements.sh``` to make sure all the dependencies are
    present.
  * __PyGame__ Python example using pygame to obtain camera frames.
    Use ```install_requirements.sh``` to make sure all the dependencies are
    present.
  * __OpenCV__ Python example using OpenCV to obtain camera frames.
    Use ```install_requirements.sh``` to make sure all the dependencies are
    present.
  * __NativeApp__ C++ example using gstreamer to obtain camera frames.
    See README in the nativeapp directory on how to compile for the
    Coral DevBoard.

## Canned models

For all the demos in this repository you can change the model and the labels
file by using the flags flags ```--model``` and
```--labels```. Be sure to use the models labeled _edgetpu, as those are
compiled for the accelerator -  otherwise the model will run on the CPU and
be much slower.

For classification you need to select one of the classification models
and its corresponding labels file:

```
inception_v1_224_quant_edgetpu.tflite, imagenet_labels.txt
inception_v2_224_quant_edgetpu.tflite, imagenet_labels.txt
inception_v3_299_quant_edgetpu.tflite, imagenet_labels.txt
inception_v4_299_quant_edgetpu.tflite, imagenet_labels.txt
mobilenet_v1_1.0_224_quant_edgetpu.tflite, imagenet_labels.txt
mobilenet_v2_1.0_224_quant_edgetpu.tflite, imagenet_labels.txt

mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite, inat_bird_labels.txt
mobilenet_v2_1.0_224_inat_insect_quant_edgetpu.tflite, inat_insect_labels.txt
mobilenet_v2_1.0_224_inat_plant_quant_edgetpu.tflite, inat_plant_labels.txt
```

For detection you need to select one of the SSD detection models
and its corresponding labels file:

```
mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite, coco_labels.txt
mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite, coco_labels.txt
mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite, coco_labels.txt
```


