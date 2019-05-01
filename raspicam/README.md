This folder contains some simple camera classification examples specific to Raspberry
Pi, using the picamera python module to access the camera.

If you dont have picamera installed you can install it by:

```
pip3 install picamera
```

Don't forget to enable your camera using raspi-config under "Interfacing Options":

```
sudo raspi-config 
```

To run the demo execture the following command, which will use the default 
model ```mobilenet_v2_1.0_224_quant_edgetpu.tflite``` 


```
python3 classify_capture.py
``` 

You can change the model and the labels file using flags:

```
python3 classify_capture.py --model ../all_models/inception_v3_299_quant_edgetpu.tflite

``` 
