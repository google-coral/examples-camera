#    Copyright 2019 Google LLC
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        https://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""A demo to classify Pygame camera stream."""
import argparse
import collections
from collections import deque
import io
import numpy as np
import operator
import os
import pygame
import pygame.camera
from pygame.locals import *
import tflite_runtime.interpreter as tflite
import time

Class = collections.namedtuple('Class', ['id', 'score'])
EDGETPU_SHARED_LIB = 'libedgetpu.so.1'

def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

def input_size(interpreter):
    """Returns input image size as (width, height, channels) tuple."""
    _, height, width, channels = interpreter.get_input_details()[0]['shape']
    return width, height, channels

def input_tensor(interpreter):
    """Returns input tensor view as numpy array of shape (height, width, 3)."""
    tensor_index = interpreter.get_input_details()[0]['index']
    return interpreter.tensor(tensor_index)()[0]

def output_tensor(interpreter):
    """Returns dequantized output tensor."""
    output_details = interpreter.get_output_details()[0]
    output_data = np.squeeze(interpreter.tensor(output_details['index'])())
    scale, zero_point = output_details['quantization']
    return scale * (output_data - zero_point)

def set_interpreter(interpreter, data):
    """Copies data to input tensor."""
    input_tensor(interpreter)[:,:] = np.reshape(data, (input_size(interpreter)))
    interpreter.invoke()

def get_output(interpreter, top_k, score_threshold):
    """Returns no more than top_k classes with score >= score_threshold."""
    scores = output_tensor(interpreter)
    classes = [
        Class(i, scores[i])
        for i in np.argpartition(scores, -top_k)[-top_k:]
        if scores[i] >= score_threshold
    ]
    return sorted(classes, key=operator.itemgetter(1), reverse=True)

def main():
    default_model_dir = "../all_models"
    default_model = 'mobilenet_v2_1.0_224_quant_edgetpu.tflite'
    default_labels = 'imagenet_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    args = parser.parse_args()

    with open(args.labels, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    pygame.init()
    pygame.camera.init()
    camlist = pygame.camera.list_cameras()

    camera = pygame.camera.Camera(camlist[0], (640, 480)) 
    width, height, channels = input_size(interpreter)
    camera.start()
    try:
        fps = deque(maxlen=20)
        fps.append(time.time())
        while True:
            imagen = camera.get_image()
            imagen = pygame.transform.scale(imagen, (width, height))
            input = np.frombuffer(imagen.get_buffer(), dtype=np.uint8)
            start_ms = time.time()
            set_interpreter(interpreter, input)
            results = get_output(interpreter, top_k=3, score_threshold=0)
            inference_ms = (time.time() - start_ms)*1000.0
            fps.append(time.time())
            fps_ms = len(fps)/(fps[-1] - fps[0])
            annotate_text = "Inference: %5.2fms FPS: %3.1f" % (inference_ms, fps_ms)
            for result in results:
               annotate_text += "\n%.0f%% %s" % (100*result[1], labels[result[0]])
            print(annotate_text)
    finally:
        camera.stop()


if __name__ == '__main__':
    main()
