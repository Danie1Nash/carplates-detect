import numpy as np
import torch
from PIL import Image


class YoloInference:
    def __init__(self, weight_path, device=None):
        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=weight_path
        )

        self.model.eval().to(self.device)

    def __call__(self, img: Image):
        results = self.model(img)

        if len(results.pandas().xyxy) > 0 and (
            not results.pandas().xyxy[0].empty
        ):
            xmin = results.pandas().xyxy[0]["xmin"].iloc[0]
            ymin = results.pandas().xyxy[0]["ymin"].iloc[0]
            xmax = results.pandas().xyxy[0]["xmax"].iloc[0]
            ymax = results.pandas().xyxy[0]["ymax"].iloc[0]
            return [[xmin, ymin, xmax, ymax]]
        else:
            return None
