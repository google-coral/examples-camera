from typing import Tuple
import cv2
import numpy as np
import os
from datetime import date
import time


class VideoConverter:
    def __init__(self, output_path: str = './videos', fps: float = 24.0, max_video_len_s: int = 40):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        self.output_path: str = output_path
        self.fps: float = fps
        self.video_writer: cv2.VideoWriter = None
        self.maximum_video_len_in_seconds = max_video_len_s

    def _init_video_writer_if_needed(self, size:Tuple[int, int] = (320, 240)):
        if time.time() - self.video_creation_timestamp > self.maximum_video_len_in_seconds:
            self.stop_video()

        if self.video_writer is None:
            video_name = os.path.join(self.output_path, f'video_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.mpg')
            self.video_writer = cv2.VideoWriter(video_name,
                                                cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'),
                                                self.fps,
                                                size,
                                                True)
            self.video_creation_timestamp = time.time()

        return self.video_writer

    def add_image(self, image:np.array):
        #add condition(s) to start the video
        self._init_video_writer_if_needed(image.shape)
        open_cv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.video_writer.write(open_cv_image)

    def stop_video(self):
        if self.video_writer is not None:
            self.video_writer.release()
            print(f'Closed {self.video_writer.filename}')
            self.video_writer = None

        cv2.destroyAllWindows()




