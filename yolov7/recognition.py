# code for recognition based on easyOCR
import os
import random
import numpy as np
import cv2 as cv
import easyocr
savepath = "D:/PyProjects/ANPRir/ANPRir/sidebar"
reader = easyocr.Reader(['fa'])
def ocr_plate(plate_region):
    # Image pre-processing for more accurate OCR
    cv.imwrite(os.path.join(savepath, "plate_img.png"), plate_region)
    rescaled = cv.resize(plate_region, None, fx=1.2, fy=1.2, interpolation=cv.INTER_CUBIC)
    grayscale = cv.cvtColor(rescaled, cv.COLOR_BGR2GRAY)
    # OCR the preprocessed image
    grayscale_blur = cv.medianBlur(grayscale, 1)
    ret, thresh1 = cv.threshold(grayscale_blur, 120, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imwrite(os.path.join(savepath, "grayscale_blur.png"), grayscale_blur)
    plate_text_easyocr = reader.readtext(grayscale_blur)
    if plate_text_easyocr:
        (bbox, text_easyocr, ocr_confidence) = plate_text_easyocr[0]
        print("plate_text Easyocr ", text_easyocr)
    else:
        text_easyocr = "_"
        ocr_confidence = 0
    # if ocr_confidence == 'nan':

    return text_easyocr, ocr_confidence