import re

import easyocr
import numpy as np
import torch
import yaml


class EasyOCRCustom:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        use_gpu = False
        if self.device.type == "cuda":
            use_gpu = True

        with open("./src/configs/models/custom_easyocr.yaml") as file:
            inference_params = yaml.safe_load(file)

        self.ocr_model = easyocr.Reader(**inference_params, gpu=use_gpu)

    def __call__(self, image: np) -> str:
        prediction = self.ocr_model.recognize(
            image, detail=0, allowlist="012334567890ABEKMHOPCTYX"
        )

        if len(prediction) > 0:
            prediction = re.sub(" ", "", prediction[0])
            return prediction
        else:
            return None
