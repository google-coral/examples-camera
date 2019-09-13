from typing import List
import platform
import subprocess
from edgetpu.detection.engine import DetectionEngine
from edgetpu.utils import dataset_utils
from PIL import Image
from PIL import ImageDraw

class FaceDetector:
    def __init__(self, model_path: str='./model'):
        self.engine = DetectionEngine(model_path)

    def predict(self, on_img:Image) -> List[List[int]]:
        ans = self.engine.DetectWithImage(on_img,
                                          threshold=0.05,
                                          keep_aspect_ratio=False,
                                          relative_coord=False,
                                          top_k=10)
        faces = []

        if ans:
            for obj in ans:
                # if labels:
                #     print(labels[obj.label_id])
                # print('score = ', obj.score)
                faces.append(obj.bounding_box.flatten().tolist())

        return faces

    def check_image_contains_face(self, img:Image):
        predictions = self.predict(img)
        return len(predictions) > 0
