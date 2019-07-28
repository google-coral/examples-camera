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
import io
import time
from collections import deque
import numpy as np
import pygame
import pygame.camera
from pygame.locals import *

import edgetpu.classification.engine

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

    engine = edgetpu.classification.engine.ClassificationEngine(args.model)

    pygame.init()
    pygame.camera.init()
    camlist = pygame.camera.list_cameras()

    camera = pygame.camera.Camera(camlist[0], (640, 480)) 
    _, width, height, channels = engine.get_input_tensor_shape()
    camera.start()
    try:
        fps = deque(maxlen=20)
        fps.append(time.time())
        while True:
            imagen = camera.get_image()
            imagen = pygame.transform.scale(imagen, (width, height))
            input = np.frombuffer(imagen.get_buffer(), dtype=np.uint8)
            start_ms = time.time()
            results = engine.ClassifyWithInputTensor(input, top_k=3)
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
