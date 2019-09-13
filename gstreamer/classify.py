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
import re
import imp
import os
from edgetpu.classification.engine import ClassificationEngine
import gstreamer
import numpy
import signal
from VideoConverter import *
from FaceDetector import *

class Main:
    def __init__(self):
        signal.signal(signal.SIGINT, self.sigint_handler)
        self.video_converter = VideoConverter(output_path='./videos')
        self.face_detector = FaceDetector(model_path='/home/mendel/cameraSamples/examples-camera/all_models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite')

    def _callback(self, image, svg_canvas):
        # image.save('out.bmp')
        if self.face_detector.check_image_contains_face(image):
            self.video_converter.add_image(numpy.array(image))

    def sigint_handler(self, signum, frame):
        self.video_converter.stop_video()

    def start(self):
        _ = gstreamer.run_pipeline(self._callback, appsink_size=(320, 240))
        self.video_converter.stop_video()



def main():
    mainObject = Main()
    mainObject.start()

if __name__ == '__main__':
    main()
