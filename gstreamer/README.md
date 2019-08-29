This folder contains two examples using gstreamer to obtain camera images. These
examples work on Linux using a webcam, Raspberry Pi with
the Raspicam and on the Coral DevBoard using the Coral camera. For the
former two you will also need a Coral USB Accelerator to run the models.

## Installation

Make sure the gstreamer libraries are install. On the Coral DevBoard this isn't
necessary, but on Raspberry Pi or a general Linux system it will be.

```
sh install_requirements.sh
```


## Classification Demo

```
python3 classify.py
```

You can change the model and the labels file using flags ```--model``` and
```--labels```.
## Detection Demo (SSD models)

```
python3 detect.py
```

As before, you can change the model and the labels file using flags ```--model``` 
and ```--labels```.


