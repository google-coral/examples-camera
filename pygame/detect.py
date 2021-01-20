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

"""A demo to run the detector in a Pygame camera stream."""
import argparse
import os
import sys
import time

import pygame
import pygame.camera
from pygame.locals import *

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

def main():
    cam_w, cam_h = 640, 480
    default_model_dir = '../all_models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=5,
                        help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='classifier score threshold')
    args = parser.parse_args()

    with open(args.labels, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    print('Loading {} with {} labels.'.format(args.model, args.labels))

    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)

    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont('Arial', 20)

    pygame.camera.init()
    camlist = pygame.camera.list_cameras()

    inference_size = input_size(interpreter)

    camera = None
    for cam in camlist:
        try:
            camera = pygame.camera.Camera(cam, (cam_w, cam_h))
            camera.start()
            print(str(cam) + ' opened')
            break
        except SystemError as e:
            print('Failed to open {}: {}'.format(str(cam), str(e)))
            camera = None
    if not camera:
      sys.stderr.write("\nERROR: Unable to open a camera.\n")
      sys,exit(1)

    try:
      display = pygame.display.set_mode((cam_w, cam_h), 0)
    except pygame.error as e:
      sys.stderr.write("\nERROR: Unable to open a display window. Make sure a monitor is attached and that "
            "the DISPLAY environment variable is set. Example: \n"
            ">export DISPLAY=\":0\" \n")
      raise e

    red = pygame.Color(255, 0, 0)

    scale_x, scale_y = cam_w / inference_size[0], cam_h / inference_size[1]
    try:
        last_time = time.monotonic()
        while True:
            mysurface = camera.get_image()
            imagen = pygame.transform.scale(mysurface, inference_size)
            start_time = time.monotonic()
            run_inference(interpreter, imagen.get_buffer().raw)
            results = get_objects(interpreter, args.threshold)[:args.top_k]
            stop_time = time.monotonic()
            inference_ms = (stop_time - start_time)*1000.0
            fps_ms = 1.0 / (stop_time - last_time)
            last_time = stop_time
            annotate_text = 'Inference: {:5.2f}ms FPS: {:3.1f}'.format(inference_ms, fps_ms)
            for result in results:
                bbox = result.bbox.scale(scale_x, scale_y)
                rect = pygame.Rect(bbox.xmin, bbox.ymin, bbox.width, bbox.height)
                pygame.draw.rect(mysurface, red, rect, 1)
                label = '{:.0f}% {}'.format(100*result.score, labels.get(result.id, result.id))
                text = font.render(label, True, red)
                print(label, ' ', end='')
                mysurface.blit(text, (bbox.xmin, bbox.ymin))
            text = font.render(annotate_text, True, red)
            print(annotate_text)
            mysurface.blit(text, (0, 0))
            display.blit(mysurface, (0, 0))
            pygame.display.flip()
    finally:
        camera.stop()


if __name__ == '__main__':
    main()
