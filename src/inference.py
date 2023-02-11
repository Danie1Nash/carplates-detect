import time

import cv2
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont


class Inference:
    def __init__(
        self,
        detect_model,
        ocr_model,
        transform_model=None,
        plot_graph: bool = True,
    ):
        self.detect_model = detect_model
        self.ocr_model = ocr_model
        self.transform_model = transform_model

        self.plot_graph = plot_graph

    def _get_number(self, image: Image, bbox):
        return image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

    def _demonstration(self, image: Image, bbox, text: str) -> Image:
        font = font_manager.FontProperties(family="sans-serif", weight="bold")
        file = font_manager.findfont(font)
        font = ImageFont.truetype(file, 32)
        draw = ImageDraw.Draw(image)
        draw.line((bbox[0], bbox[1], bbox[2], bbox[1]), fill=128, width=4)
        draw.line((bbox[2], bbox[1], bbox[2], bbox[3]), fill=128, width=4)
        draw.line((bbox[2], bbox[3], bbox[0], bbox[3]), fill=128, width=4)
        draw.line((bbox[0], bbox[3], bbox[0], bbox[1]), fill=128, width=4)
        draw.text((bbox[0], bbox[3]), text, fill=(255, 42, 0), font=font)
        return image

    def detect_by_image(self, image):
        image_orig = image
        result_number = []
        vis_image = None
        if self.plot_graph:
            vis_image = image_orig.copy()

        stime_detect = time.time()
        detect_results = self.detect_model(image_orig)
        ftime_detect = time.time()

        stime_ocr = time.time()
        if detect_results is not None:
            for bbox in detect_results:
                img_number = np.asarray(self._get_number(image_orig, bbox))
                if self.transform_model is not None:
                    img_number = np.asarray(self.transform_model(img_number))

                text_recognition = self.ocr_model(img_number)
                result_number.append(text_recognition)
                if self.plot_graph:
                    if text_recognition is None:
                        text_recognition = "Не считано."
                    vis_image = self._demonstration(
                        vis_image, bbox, text_recognition
                    )
        ftime_ocr = time.time()

        log_info = {
            "detect_time": ftime_detect - stime_detect,
            "ocr_time": ftime_ocr - stime_ocr,
        }
        return result_number, vis_image, log_info

    def detect_by_image_path(self, path_to_image: str):
        image_orig = Image.open(path_to_image)
        return self.detect_by_image(image_orig)

    def detect_by_video_path(self, path_to_video, save_result_path):
        vid_capture = cv2.VideoCapture(path_to_video)

        width = int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid_capture.get(cv2.CAP_PROP_FPS))

        output = cv2.VideoWriter(
            save_result_path,
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            fps,
            (width, height),
        )

        while vid_capture.isOpened():
            ret, frame = vid_capture.read()
            if ret:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)

                _, result = self.detect_by_image(pil_img)

                open_cv_image = np.array(result)
                open_cv_image = open_cv_image[:, :, ::-1].copy()

                output.write(open_cv_image)
            else:
                break

        vid_capture.release()
        output.release()
