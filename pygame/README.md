This folder contains some simple camera classification and detection examples using pygame

If you dont have pygame installed you can install it by:
```
pip3 install pygame
```

To run the demo execture the following command, which will use the default 
model ```mobilenet_v2_1.0_224_quant_edgetpu.tflite``` 

You run the classifier with:
```
python3 classify_capture.py
``` 

You can change the model and the labels file using flags:
```
python3 classify_capture.py --model ../all_models/inception_v3_299_quant_edgetpu.tflite
``` 

You run the detector with:
```
python3 detect.py
```

