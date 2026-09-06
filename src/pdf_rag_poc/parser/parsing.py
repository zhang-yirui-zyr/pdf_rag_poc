import numpy as np
import pytesseract
import os
from pdf2image import convert_from_path
import cv2

def dump_images(pdf_path: str, img_path: str) -> None:
    file_name = os.path.basename(pdf_path).split('.')[0]
    img_dir = os.path.join(img_path, file_name)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    convert_from_path(pdf_path=pdf_path, output_folder=img_dir, fmt="png")

def load_images(img_path: str) -> list[np.ndarray]:
    imgs = []
    for file in sorted(os.listdir(img_path)):
        if file.endswith(".png"):
            imgs.append(cv2.imread(os.path.join(img_path, file)))
    return imgs

def parse_pdf_images(imgs: list[np.ndarray]) -> str:
    text = ""
    for im in imgs:
        text += pytesseract.image_to_string(im)
    return text

