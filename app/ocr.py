# app/ocr.py

from PIL import Image
from paddleocr import PaddleOCR
# Load OCR and LLM models

# File: app/ocr.py
paddleocr = PaddleOCR(lang="en", ocr_version="PP-OCRv4", show_log=False, use_gpu=False)

def paddle_scan(image_array):
    result = paddleocr.ocr(image_array,cls=True)
    result = result[0]
    txts = [line[1][0] for line in result]     #raw text
    return  txts
