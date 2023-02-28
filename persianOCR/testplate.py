# test on a plate image (seperate chops for each character)
import numpy as np
import tensorflow as tf
import cv2
from math import sqrt, atan, degrees
import os
import matplotlib.pyplot as plt
import numpy as np

def find_line_angle(line):
    x1,y1,x2,y2 = line[0]
    angle = degrees(atan(((y2-y1)/(x2-x1))))
    return angle

def rotate_image(plate_img_gr, angle):
    (h, w) = plate_img_gr.shape
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    rotated = cv2.warpAffine(plate_img_gr, M, (w, h))
    return rotated

def adjust_cropping(rotated_img):
    h,w = rotated_img.shape
    targ_h = int(w/4)
    crop_h = int((h - targ_h)/2)
    cropped_rotated_img = rotated_img[crop_h:h-crop_h,:]
    return cropped_rotated_img
# Draw the lines on the  image
#lines_edges = cv.addWeighted(plate_img, 0.8, line_image, 1, 0)
def find_longest_line(plate_img, plate_img_gr):
    kernel_size = 3
    blur_gray = cv2.GaussianBlur(plate_img_gr, (kernel_size, kernel_size), 0)

    low_threshold = 150
    high_threshold = 200

    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments
    line_image = np.copy(plate_img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    lls = []
    for indx, line in enumerate(lines):
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
            line_length = sqrt((x2-x1)**2 + (y2-y1)**2)
            lls.append((indx,line_length))
    lls.sort(key = lambda x: x[1])
    linessorted = []
    for (indx,ll) in lls:
        linessorted.append(lines[indx])
    return linessorted
def align_rectangle(plate_img, plate_img_gr):
    linessorted = find_longest_line(plate_img,plate_img_gr)
    rot_angle = find_line_angle(linessorted[-1])
    rotated_img = rotate_image(plate_img_gr, rot_angle)
    cropped_rotated_img = adjust_cropping(rotated_img)
    return  cropped_rotated_img
# from myutils import c
# weights-improvement-41-0.42.

def load_model():
    model_path = 'saved_model/'
    model = tf.keras.models.load_model(model_path + 'weights-improvement-41-0.42.hdf5', custom_objects={'tf': tf})
    print(model)
    return model
ocr_model = load_model()
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'D', 'Gh', 'H', 'J', 'L', 'M', 'N', 'P', 'PuV',
'PwD', 'Sad', 'Sin', 'T', 'Taxi', 'V', 'Y']#
def image_reader_predict(path):
    plate_img = cv2.imread(path)
    plate_img_gr = cv2.imread(path, 0)
    plt.imshow(plate_img_gr)
    # from align_rect import align_rectangle

    cropped_rotated_img = align_rectangle(plate_img, plate_img_gr)
    plt.imshow(cropped_rotated_img)

    h, w = cropped_rotated_img.shape
    chopfactors = [(40, 120), (100, 200), (180, 280), (270, 360), (350, 400), (400, 460), (460, 530), (530, 600)]
    plate_letters= []
    for factor in chopfactors:
        w1 = int((factor[0]/600)*w)
        w2 = int((factor[1]/600)*w)
        letterpatch = cropped_rotated_img[:,w1:w2]
        letterpatch
        #cv.imwrite(f"{str(factor[0])}_{str(factor[1])}.png", letterpatch)
        letterpatch_resized = cv2.resize(letterpatch, (32,32), interpolation= cv2.INTER_LINEAR)
        plate_letters.append(letterpatch_resized)
    plate_letters = np.array(plate_letters)
    plate_letters = np.tile(np.expand_dims(plate_letters,axis=-1),(1,1,1,3))

    predictions = ocr_model.predict(plate_letters)
    text = [class_names[k] for k in list(np.argmax(predictions, axis=1))]
    return text
text = image_reader_predict('pelakir_3.jpg')
print(text)

def align_by_fft(img):
    fft = np.fft.fft2(img)
    max_peak = np.max(np.abs(fft))
    fft[fft < (max_peak * 0.25)] = 0
    abs_data = 1 + np.abs(fft)
    c = 255.0 / np.log(1 + max_peak)
    log_data = c * np.log(abs_data)
    max_scaled_peak = np.max(log_data)
    rows, cols = np.where(log_data > (max_scaled_peak * 0.90))
    min_col, max_col = np.min(cols), np.max(cols)
    min_row, max_row = np.min(rows), np.max(rows)
    dy, dx = max_col - min_col, max_row - min_row
    theta = np.arctan(dy / float(dx))
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    def transform_data(m):
        dpix, dpiy = m.shape
        x_c, y_c = np.unravel_index(np.argmax(m), m.shape)
        angles = np.linspace(0, np.pi*2, min(dpix, dpiy))
        mrc = min(abs(x_c - dpix), abs(y_c - dpiy), x_c, y_c)
        radiuses = np.linspace(0, mrc, max(dpix, dpiy))
        A, R = np.meshgrid(angles, radiuses)
        X = R * np.cos(A)
        Y = R * np.sin(A)
        return A, R, m[X.astype(int) + mrc - 1, Y.astype(int) + mrc - 1]

    angles, radiuses, m = transform_data(magnitude_spectrum)
    c=m
    sample_angles = np.linspace(0,  2 * np.pi, len(c.sum(axis=0))) / np.pi*180
    turn_angle_in_degrees = 90 - sample_angles[np.argmax(c.sum(axis=0))]
    return turn_angle_in_degrees