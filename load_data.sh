# ------------------ OCR Model ------------------
# EasyORC
echo "EasyORC_custom"
gdown 1mZ_nfXYpmV91gPYCtGdG89Qh3x0qPfTQ -O ./weights/ocr/

# ------------------ Detect model ------------------
# YoloV5
echo "YoloV5"
gdown 1Tyz5YJGjzPGlivp4DoJyegkzU2IUmcvo -O ./weights/detection/

# ------------------ Transform model ------------------
# STN
echo "STN"
gdown 1cnuUkpxBkMThqdFvR4QPDu2zP53ziaBS -O ./weights/transform/