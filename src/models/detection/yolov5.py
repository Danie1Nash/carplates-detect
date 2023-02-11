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
        bboxs = results.xyxy[0][:, :4].tolist()
        return bboxs
