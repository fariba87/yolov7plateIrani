# code for inference on a single image(based on YOLO detection  and easyOCR recognition)
import fileinput
import os
from pathlib import Path
from typing import Union

import cv2
import torch
import cv2 as cv
import numpy as np
import re
import matplotlib.pyplot as plt
from copy import deepcopy
from deep_sort_realtime.deepsort_tracker import DeepSort
# if not os.path.isfile('weights.pt'):
#     weights_url = 'https://archive.org/download/anpr_weights/weights.pt'
#     os.system(f'wget {weights_url}')
#
# if not os.path.isdir('examples'):
#     examples_url = 'https://archive.org/download/anpr_examples_202208/examples.tar.gz'
#     os.system(f'wget {examples_url}')
#     os.system('tar -xvf examples.tar.gz')
#     os.system('rm -rf examples.tar.gz')

# def prepend_text(filename: Union[str, Path], text: str):
#     with fileinput.input(filename, inplace=True) as file:
#         for line in file:
#             if file.isfirstline():
#                 print(text)
#             print(line, end="")
#
# if not os.path.isdir('yolov7'):
#     yolov7_repo_url = 'https://github.com/WongKinYiu/yolov7'
#     os.system(f'git clone {yolov7_repo_url}')
#     # Fix import errors
#     for file in ['yolov7/models/common.py', 'yolov7/models/experimental.py', 'yolov7/models/yolo.py', 'yolov7/utils/datasets.py']:
#          prepend_text(file, "import sys\nsys.path.insert(0, './yolov7')")
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size
from yolov7.utils.torch_utils import select_device, TracedModel
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.plots import plot_one_box
from myutils import crop
from recognition import ocr_plate

from detection import detect_plate
weights = 'weights/best_v7_plate.pt'
device_id = 'cpu'
image_size = 640
trace = True

# Initialize
device = select_device(device_id)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(image_size, s=stride)  # check img_size

if trace:
    model = TracedModel(model, device, image_size)

if half:
    model.half()  # to FP16

if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
source_image_path = "pelakir_1.jpg"
source_image = cv.imread(source_image_path)
model.eval()


def get_plates_from_image(input):
    if input is None:
        return None
    plate_detections, det_confidences = detect_plate(input,model)
    plate_texts = []
    ocr_confidences = []
    detected_image = deepcopy(input)
    for coords in plate_detections:
        plate_region = crop(input, coords)
        plate_text, ocr_confidence = ocr_plate(plate_region)
        plate_texts.append(plate_text)
        print(plate_text)
        ocr_confidences.append(ocr_confidence)
        plot_one_box(coords, detected_image, label=plate_text, color=[0, 150, 255], line_thickness=2)
    for coords in plate_detections:
        cv.rectangle(source_image, (coords[0], coords[1]), (coords[2], coords[3]), (0,255,0), 2)
    cv.imwrite("det_image_1.jpg", source_image)
    return detected_image
detect1 = get_plates_from_image(source_image)
print(detect1.shape)





