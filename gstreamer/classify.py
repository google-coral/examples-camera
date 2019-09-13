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

try:
    from .VideoWriter import *
    from .FaceDetector import *
except Exception: #ImportError
    from VideoWriter import *
    from FaceDetector import *

class CONSTANTS:
    # number of seconds that need to pass after which the video recording is stopped and the video saved to disk
    NO_FACE_THRESHOLD_S = 15

class Main:
    def __init__(self):
        signal.signal(signal.SIGINT, self.sigint_handler)
        self.video_writer = VideoWriter(output_path='./videos')
        self.face_detector = FaceDetector(model_path='/home/mendel/cameraSamples/examples-camera/all_models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite')
        self.last_face_seen_timestamp = 0

    def _callback(self, image, svg_canvas):
        face_rois_in_image = self.face_detector.predict(image)

        if self.video_writer.is_video_recording_in_progress():
            self.video_writer.add_image(numpy.array(image))

            if len(face_rois_in_image) > 0:
                self.last_face_seen_timestamp = time.time()
                self.video_writer.save_image_at_same_path(numpy.array(image.crop(face_rois_in_image[0])))

            if time.time() - self.last_face_seen_timestamp >= CONSTANTS.NO_FACE_THRESHOLD_S:
                self.video_writer.stop_video_recording()
                self.last_face_seen_timestamp = 0

        elif len(face_rois_in_image) > 0:
            self.last_face_seen_timestamp = time.time()
            self.video_writer.start_video_recording(numpy.array(image))
            self.video_writer.save_image_at_same_path(numpy.array(image.crop(face_rois_in_image[0])))

    def sigint_handler(self, signum, frame):
        self.video_writer.stop_video_recording()

    def start(self):
        _ = gstreamer.run_pipeline(self._callback, appsink_size=(320, 240))
        self.video_writer.stop_video_recording()


def main():
    mainObject = Main()
    mainObject.start()

if __name__ == '__main__':
    main()
