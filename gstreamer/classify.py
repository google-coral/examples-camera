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
# from edgetpu.classification.engine import ClassificationEngine
import gstreamer
import numpy
import signal
from PIL import Image
from types import Tuple, Union

try:
    from .VideoWriter import *
    from .FaceDetector import *
except Exception:  # ImportError
    from VideoWriter import *
    from FaceDetector import *

class CONSTANTS:
    # number of seconds that need to pass after which the video recording is stopped and the video saved to disk
    NO_FACE_THRESHOLD_SEC = 5 # seconds
    # number of seconds that need to pass for a pullup to be considered valid
    # scope: false positive counts removal
    MIN_SEC_PER_PULLUP = 1

    class RECORD_STATUS:
        OFF, JUST_STARTED, ON, JUST_STOPPED, *_ = range(10)
        POSITIVE_STATS = [JUST_STARTED, ON]
        NEGATIVE_STATS = [JUST_STOPPED, OFF]

    SAVE_FILE_NAME = "db.csv"

class Main:
    def __init__(self):
        signal.signal(signal.SIGINT, self.sigint_handler)
        self.video_writer = VideoWriter(output_path='/home/mendel/mnt/resources/videos')
        self.recording_last_face_seen_timestamp = 0

        self.face_detector = FaceDetector(model_path='/home/mendel/mnt/cameraSamples/examples-camera/all_models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite')

        self.who = None
        self.counter: int = 0
        self.counter_up_down: bool = False  # on off switch. False for human not visible (thus-down). True for face up.
        self.counting_prev_face_seen_timestamp = 0

    def _record(self, image, face_rois_in_image: List[List[int]]) -> Tuple[CONSTANTS.RECORD_STATUS, Union[None, str]]:
        seeing_a_face: bool = len(face_rois_in_image) > 0

        if self.video_writer.is_video_recording_in_progress():
            self.video_writer.add_image(numpy.array(image))

            if time.time() - self.recording_last_face_seen_timestamp >= CONSTANTS.NO_FACE_THRESHOLD_SEC:
                video_path = self.video_writer.video_name  # create a backup since stop_video_recording messes this up
                self.video_writer.stop_video_recording()
                self.recording_last_face_seen_timestamp = 0
                return CONSTANTS.RECORD_STATUS.JUST_STOPPED, video_path

            if seeing_a_face:
                self.recording_last_face_seen_timestamp = time.time()
                self.video_writer.save_image_at_same_path(numpy.array(image.crop(face_rois_in_image[0])))

            return CONSTANTS.RECORD_STATUS.ON, self.video_writer.video_name

        elif seeing_a_face:
            self.recording_last_face_seen_timestamp = time.time()
            self.video_writer.start_video_recording(numpy.array(image))
            self.video_writer.save_image_at_same_path(numpy.array(image.crop(face_rois_in_image[0])))
            return CONSTANTS.RECORD_STATUS.JUST_STARTED, self.video_writer.video_name

        return CONSTANTS.RECORD_STATUS.OFF, None

    def _count(self, face_rois_in_image: List[List[int]]) -> int:
        seeing_a_face: bool = len(face_rois_in_image) > 0

        # if seeing a face and was not seeing a face before
        if seeing_a_face and not self.counter_up_down:
            self.counter_up_down = True  # up

            # if at least MIN_SEC_PER_PULLUP have passed, it means there was a pullup done
            # and the coral camera did not just lose focus
            if time.time() - self.counting_prev_face_seen_timestamp >= CONSTANTS.MIN_SEC_PER_PULLUP:
                self.counting_prev_face_seen_timestamp = time.time()
                self.counter += 1

        elif not seeing_a_face:
            self.counter_up_down = False  # down

        return self.counter

    def _whothis(self, image_of_face: Image) -> str:
        return "Octav"

    def _save(self, who: str, pullup_counts: int, evidence_path: str):
        with open(CONSTANTS.SAVE_FILE_NAME, "a+") as track_file:
            # when,who,how_many,evidence
            track_file.write(f'{time.time()},{who},{pullup_counts},{evidence_path}')

    def _reset_session(self):
        self.counter = 0
        self.who = None

    def _callback(self, image, svg_canvas):
        face_rois_in_image = self.face_detector.predict(image)

        record_status, video_path = self._record(image=image, face_rois_in_image=face_rois_in_image)
        counts = self._count(face_rois_in_image=face_rois_in_image)
        if len(face_rois_in_image) > 0:
            # TODO: ENSEMBLE predictions eventually
            self.who = self._whothis(image_of_face=image.crop(face_rois_in_image[0]))

        if record_status == CONSTANTS.RECORD_STATUS.JUST_STOPPED:
            self._save(video_path)
            self._reset_session()


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
