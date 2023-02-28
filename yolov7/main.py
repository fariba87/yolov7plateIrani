# main code for inference whether input in image, saved video, or online video from webcam
import fileinput
import os
from pathlib import Path
from typing import Union
import torch
import cv2 as cv
import numpy as np
import re
import matplotlib.pyplot as plt
from deep_sort_realtime.deepsort_tracker import DeepSort
# if not os.path.isfile('weights.pt'):
#     weights_url = 'https://archive.org/download/anpr_weights/weights.pt'
#     os.system(f'wget {weights_url}')

import os
from pathlib import Path
from typing import Union
import torch
import cv2 as cv
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size
from yolov7.utils.torch_utils import select_device, TracedModel
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.plots import plot_one_box, plot_one_box_PIL
from copy import deepcopy
import easyocr
from myutils import pascal_voc_to_coco
from detection import detect_plate
from recognition import ocr_plate
from myutils import get_best_ocr

from myutils import crop
images_n_vids_path = "C:/PyProjects/ANPRir/ANPRir/images_vids"
image_path = os.path.join(images_n_vids_path, "pelakir_2.jpg")
video_path = os.path.join(images_n_vids_path, "test_video_short.mp4")

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

# Load OCR
reader = easyocr.Reader(['fa'])

def get_plates_from_image(input):
    if input is None:
        return None
    plate_detections, det_confidences = detect_plate(input, model)
    plate_texts = []
    ocr_confidences = []
    detected_image = deepcopy(input)
    for coords in plate_detections:
        plate_region = crop(input, coords)
        plate_text, ocr_confidence = ocr_plate(plate_region)
        plate_texts.append(plate_text)
        ocr_confidences.append(ocr_confidence)
        detected_image = plot_one_box_PIL(coords, detected_image, label=plate_text, color=[0, 150, 255],
                                          line_thickness=2)
    return detected_image




def get_plates_from_video(source):
    if source is None:
        return None

    # Create a VideoCapture object
    video = cv.VideoCapture(source)

    # Default resolutions of the frame are obtained. The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object.
    temp = f'{Path(source).stem}_temp{Path(source).suffix}'
    export = cv.VideoWriter(temp, cv.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Intializing tracker
    tracker = DeepSort(embedder_gpu=False)

    # Initializing some helper variables.
    preds = []
    total_obj = 0

    while (True):
        ret, frame = video.read()
        if ret == True:
            # Run the ANPR algorithm
            bboxes, scores = detect_plate(frame,model)
            # Convert Pascal VOC detections to COCO
            bboxes = list(map(lambda bbox: pascal_voc_to_coco(bbox), bboxes))

            if len(bboxes) > 0:
                # Storing all the required info in a list.
                detections = [(bbox, score, 'number_plate') for bbox, score in zip(bboxes, scores)]

                # Applying tracker.
                # The tracker code flow: kalman filter -> target association(using hungarian algorithm) and appearance descriptor.
                tracks = tracker.update_tracks(detections, frame=frame)

                # Checking if tracks exist.
                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    # Changing track bbox to top left, bottom right coordinates
                    bbox = [int(position) for position in list(track.to_tlbr())]

                    for i in range(len(bbox)):
                        if bbox[i] < 0:
                            bbox[i] = 0

                    # Cropping the license plate and applying the OCR.
                    plate_region = crop(frame, bbox)
                    plate_text, ocr_confidence = ocr_plate(plate_region)

                    # Storing the ocr output for corresponding track id.
                    output_frame = {'track_id': track.track_id, 'ocr_txt': plate_text, 'ocr_conf': ocr_confidence}

                    # Appending track_id to list only if it does not exist in the list
                    # else looking for the current track in the list and updating the highest confidence of it.
                    if track.track_id not in list(set(pred['track_id'] for pred in preds)):
                        total_obj += 1
                        preds.append(output_frame)
                    else:
                        preds, ocr_confidence, plate_text = get_best_ocr(preds, ocr_confidence, plate_text,
                                                                         track.track_id)

                    # Plotting the prediction.
                    frame = plot_one_box_PIL(bbox, frame, label=f'{str(track.track_id)}. {plate_text}',
                                             color=[255, 150, 0], line_thickness=3)
                    cv.imshow("frame ", frame)
                    keyexit = cv.waitKey(0)
                    if keyexit == 27:
                        break
            # Write the frame into the output file
            export.write(frame)
        else:
            break

            # When everything done, release the video capture and video write objects
    cv.destroyAllWindows()
    video.release()
    export.release()

    # Compressing the output video for smaller size and web compatibility.
    output = f'{Path(source).stem}_detected{Path(source).suffix}'
    os.system(
        f'ffmpeg -y -i {temp} -c:v libx264 -b:v 5000k -minrate 1000k -maxrate 8000k -pass 1 -c:a aac -f mp4 /dev/null && ffmpeg -i {temp} -c:v libx264 -b:v 5000k -minrate 1000k -maxrate 8000k -pass 2 -c:a aac -movflags faststart {output}')
    os.system(f'rm -rf {temp} ffmpeg2pass-0.log ffmpeg2pass-0.log.mbtree')

    return output


def get_plates_from_webcam():
    # Create a VideoCapture object
    video = cv.VideoCapture(0)

    # Default resolutions of the frame are obtained. The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object.
    temp = f'cam_temp.mp4'
    export = cv.VideoWriter(temp, cv.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Intializing tracker
    tracker = DeepSort(embedder_gpu=False)

    # Initializing some helper variables.
    preds = []
    total_obj = 0
    fr_count = 0
    while (True):
        ret, frame = video.read()
        if ret == True:

            fr_count += 1
            if fr_count % 10 != 0:
                continue

            # Run the ANPR algorithm
            bboxes, scores = detect_plate(frame,model)
            # Convert Pascal VOC detections to COCO
            bboxes = list(map(lambda bbox: pascal_voc_to_coco(bbox), bboxes))

            if len(bboxes) > 0:
                # Storing all the required info in a list.
                detections = [(bbox, score, 'number_plate') for bbox, score in zip(bboxes, scores)]

                # Applying tracker.
                # The tracker code flow: kalman filter -> target association(using hungarian algorithm) and appearance descriptor.
                tracks = tracker.update_tracks(detections, frame=frame)

                # Checking if tracks exist.
                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    # Changing track bbox to top left, bottom right coordinates
                    bbox = [int(position) for position in list(track.to_tlbr())]

                    for i in range(len(bbox)):
                        if bbox[i] < 0:
                            bbox[i] = 0

                    # Cropping the license plate and applying the OCR.
                    plate_region = crop(frame, bbox)
                    plate_text, ocr_confidence = ocr_plate(plate_region)

                    # Storing the ocr output for corresponding track id.
                    output_frame = {'track_id': track.track_id, 'ocr_txt': plate_text, 'ocr_conf': ocr_confidence}

                    # Appending track_id to list only if it does not exist in the list
                    # else looking for the current track in the list and updating the highest confidence of it.
                    if track.track_id not in list(set(pred['track_id'] for pred in preds)):
                        total_obj += 1
                        preds.append(output_frame)
                    else:
                        preds, ocr_confidence, plate_text = get_best_ocr(preds, ocr_confidence, plate_text,
                                                                         track.track_id)

                    # Plotting the prediction.
                    frame = plot_one_box_PIL(bbox, frame, label=f'{str(track.track_id)}. {plate_text}',
                                             color=[255, 150, 0], line_thickness=3)
                    cv.imshow("frame ", frame)
                    keyexit = cv.waitKey(0)
                    if keyexit == 27:
                        break
            # Write the frame into the output file
            export.write(frame)
        else:
            break

            # When everything done, release the video capture and video write objects
    cv.destroyAllWindows()
    video.release()
    export.release()

    # Compressing the output video for smaller size and web compatibility.
    output = f'cam_detected.mp4'
    os.system(
        f'ffmpeg -y -i {temp} -c:v libx264 -b:v 5000k -minrate 1000k -maxrate 8000k -pass 1 -c:a aac -f mp4 /dev/null && ffmpeg -i {temp} -c:v libx264 -b:v 5000k -minrate 1000k -maxrate 8000k -pass 2 -c:a aac -movflags faststart {output}')
    os.system(f'rm -rf {temp} ffmpeg2pass-0.log ffmpeg2pass-0.log.mbtree')

    return output



detected_plate_webcam = get_plates_from_video(source='D:/Afagh/text generation/images_vids/test_video_short.mp4')