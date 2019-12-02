# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo which runs object classification on camera frames."""
import argparse
import collections
import common
import gstreamer
import numpy as np
import operator
import os
from PIL import Image
import re
import svgwrite
import tflite_runtime.interpreter as tflite
import time

import gi
from gi.repository import Gst

Class = collections.namedtuple('Class', ['id', 'score'])
EDGETPU_SHARED_LIB = 'libedgetpu.so.1'

def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

def generate_svg(size, text_lines):
    dwg = svgwrite.Drawing('', size=size)
    for y, line in enumerate(text_lines, start=1):
      dwg.add(dwg.text(line, insert=(11, y*20+1), fill='black', font_size='20'))
      dwg.add(dwg.text(line, insert=(10, y*20), fill='white', font_size='20'))
    return dwg.tostring()

def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

def input_size(interpreter):
    """Returns input image size as (width, height) tuple."""
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

def set_input(interpreter, buf):
    """Copies data to input tensor."""
    result, mapinfo = buf.map(Gst.MapFlags.READ)
    if result:
        np_buffer = np.reshape(np.frombuffer(mapinfo.data, dtype=np.uint8), input_size(interpreter))
        input_tensor(interpreter)[:, :] = np_buffer
        buf.unmap(mapinfo)

def set_interpreter(interpreter, data):
    #image = image.resize((input_size(interpreter)), resample)
    set_input(interpreter, data)
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
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of classes with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='class score threshold')
    args = parser.parse_args()

    print("Loading %s with %s labels."%(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = load_labels(args.labels)

    w, h, _  = input_size(interpreter)
    inference_size = (w, h)
    # Average fps over last 30 frames.
    fps_counter = common.avg_fps_counter(30)

    def user_callback(input_tensor, src_size, inference_box):
      nonlocal fps_counter
      start_time = time.monotonic()
      set_interpreter(interpreter, input_tensor)
      results = get_output(interpreter, args.top_k, args.threshold)
      end_time = time.monotonic()
      text_lines = [
          ' ',
          'Inference: %.2f ms' %((end_time - start_time) * 1000),
          'FPS: %d fps' % (round(next(fps_counter))),
      ]
      for result in results:
          text_lines.append('score=%.2f: %s' % (result.score, labels.get(result.id, result.id)))
      print(' '.join(text_lines))
      return generate_svg(src_size, text_lines)

    result = gstreamer.run_pipeline(user_callback, appsink_size=inference_size)

if __name__ == '__main__':
    main()

