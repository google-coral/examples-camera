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
import os
import time

import pygame
import pygame.camera
from pygame.locals import *

from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
from pycoral.adapters.common import input_size
from pycoral.adapters.classify import get_classes

def main():
    default_model_dir = '../all_models'
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

    print('By default using camera: ', camlist[-1])
    camera = pygame.camera.Camera(camlist[-1], (640, 480)) 
    inference_size = input_size(interpreter)
    camera.start()
    try:
        last_time = time.monotonic()
        while True:
            imagen = camera.get_image()
            imagen = pygame.transform.scale(imagen, inference_size)
            start_ms = time.time()
            run_inference(interpreter, imagen.get_buffer().raw)
            results = get_classes(interpreter, top_k=3, score_threshold=0)
            stop_time = time.monotonic()
            inference_ms = (time.time() - start_ms)*1000.0
            fps_ms = 1.0 / (stop_time - last_time)
            last_time = stop_time
            annotate_text = 'Inference: {:5.2f}ms FPS: {:3.1f}'.format(inference_ms, fps_ms)
            for result in results:
               annotate_text += '\n{:.0f}% {}'.format(100*result[1], labels[result[0]])
            print(annotate_text)
    finally:
        camera.stop()


if __name__ == '__main__':
    main()
