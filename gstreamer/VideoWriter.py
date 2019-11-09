from typing import Tuple
import cv2
import numpy as np
import os
from datetime import datetime
import time


class VideoWriter:
    DEFAULT_VIDEO_NAME = "video.mpg"

    def __init__(self, output_path: str = './videos', fps: float = 30.0, max_video_len_s: int = 60):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.output_path = output_path
        self.fps = fps
        self.video_writer = None
        self.maximum_video_len_in_seconds = max_video_len_s
        self.video_creation_timestamp = 0
        self.video_name = self.DEFAULT_VIDEO_NAME

    def _init_video_writer_if_needed(self, size: Tuple[int, int] = (320, 240)):
        if time.time() - self.video_creation_timestamp > self.maximum_video_len_in_seconds:
            self.stop_video_recording()

        if self.video_writer is None:
            date_name = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

            video_path = os.path.join(self.output_path, date_name)
            if not os.path.exists(video_path):
                os.mkdir(video_path)

            self.video_name = os.path.join(video_path, self.DEFAULT_VIDEO_NAME)
            self.video_writer = cv2.VideoWriter(self.video_name,
                                                cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'),
                                                self.fps,
                                                size,
                                                True)
            self.video_creation_timestamp = time.time()

        return self.video_writer

    def start_video_recording(self, image: np.array):
        video_shape = (image.shape[1], image.shape[0])
        self._init_video_writer_if_needed(video_shape)
        self.add_image(image)

    def add_image(self, image: np.array):
        open_cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.video_writer.write(open_cv_image)

    def stop_video_recording(self):
        if self.video_writer is not None:
            self.video_writer.release()
            print('Closed ' + self.video_name)
            self.video_writer = None
            self.video_name = self.DEFAULT_VIDEO_NAME
            self.video_creation_timestamp = 0

        cv2.destroyAllWindows()

    def is_video_recording_in_progress(self) -> bool:
        return self.video_writer is not None

    def save_image_at_same_path(self, image: np.array):
        open_cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_path = os.path.join(os.path.dirname(self.video_name), datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".png")
        cv2.imwrite(img_path, open_cv_image)



