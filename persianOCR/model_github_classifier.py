
import os
import tensorflow as tf
import keras
import numpy as np
import cv2 as cv
import random
import matplotlib.pyplot as plt
#!pip install roboflow

# from roboflow import Roboflow
# rf = Roboflow(api_key="VdwBRxsJsFmCMW94xJjd")
# project = rf.workspace("object-detection-yolov5").project("plate_ocr_ir")
# dataset = project.version(2).download("folder")


train_path = "/content/plate_ocr_ir-2/train"
valid_path = "/content/plate_ocr_ir-2/valid"
test_path = "/content/plate_ocr_ir-2/test"


class_names = subdirs = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'ch', 'd', 'ein', 'f', 'g', 'ghaf', 'ghein', 'h2', 'hj', 'j', 'k', 'kh', 'l', 'm', 'n', 'p', 'r', 's', 'sad', 'sh', 't', 'ta', 'th', 'v', 'y', 'z', 'za', 'zad', 'zal', 'zh']
num_classes = len(class_names)


labels_nums = [i for i in range(0,len(class_names))]
print(labels_nums)

train_imgs = []
train_labels = []
for indx, subdir in enumerate(subdirs):
  imgfolder = os.path.join(train_path, subdir)
  for imgname in os.listdir(imgfolder):
    img = cv.imread(os.path.join(imgfolder, imgname), 0)
    train_imgs.append(img)
    train_labels.append(labels_nums[indx])

c = list(zip(train_imgs, train_labels))
random.shuffle(c)
train_imgs, train_labels = zip(*c)

train_images = np.array(train_imgs)
train_labels = np.array(train_labels)


valid_imgs = []
valid_labels = []
for indx, subdir in enumerate(subdirs):
  imgfolder = os.path.join(valid_path, subdir)
  if os.path.exists(imgfolder):
    for imgname in os.listdir(imgfolder):
      img = cv.imread(os.path.join(imgfolder, imgname), 0)
      valid_imgs.append(img)
      valid_labels.append(labels_nums[indx])

c = list(zip(valid_imgs, valid_labels))
random.shuffle(c)
valid_imgs, valid_labels = zip(*c)

test_images = np.array(valid_imgs)
test_labels = np.array(valid_labels)


num=90
print(class_names[train_labels[num]])
print(train_labels[num])
plt.imshow(train_images[num])

test_labels.shape

train_images = train_images / 255.0
test_images = test_images / 255.0


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(num_classes, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=200)



predictions = model.predict(test_images)

model.save('saved_model/simple_ocr_plates_model')


ocr_model = tf.keras.models.load_model('simple_ocr_plates_model')


test_img_path = "/content/plate_ocr_ir-2/test/ch/c_250_png.rf.d3aa38e2f0156f1ca06bf9206f7966aa.jpg"
test_img = cv.imread(test_img_path, 0)
test_img = np.expand_dims(test_img, axis=0)
predictions = ocr_model.predict(test_images)


class_names[np.argmax(predictions[0])]

test_imgs = []
test_labels = []
for indx, subdir in enumerate(subdirs):
  imgfolder = os.path.join(test_path, subdir)
  if os.path.exists(imgfolder):
    for imgname in os.listdir(imgfolder):
      img = cv.imread(os.path.join(imgfolder, imgname), 0)
      test_imgs.append(img)
      test_labels.append(labels_nums[indx])

c = list(zip(test_imgs, test_labels))
random.shuffle(c)
test_imgs, test_labels = zip(*c)

test_images = np.array(test_imgs)
test_labels = np.array(test_labels)



predictions = ocr_model.predict(test_images)


num=30
print(class_names[np.argmax(predictions[num])])
print(np.argmax(predictions[num]))
plt.imshow(test_images[num])


y_predicted = model.predict(test_images)
y_predicted_labels = [np.argmax(i) for i in y_predicted]

cm = tf.math.confusion_matrix(labels=test_labels, predictions=y_predicted_labels)

import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')

################################################
#test
import tensorflow as tf
import os
import random
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from math import sqrt, atan, degrees


plates_folder = "/content/gdrive/MyDrive/DL_projects_colab/plate_imgs"
class_names = subdirs = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'ch', 'd', 'ein', 'f', 'g', 'ghaf', 'ghein', 'h2', 'hj', 'j', 'k', 'kh', 'l', 'm', 'n', 'p', 'r', 's', 'sad', 'sh', 't', 'ta', 'th', 'v', 'y', 'z', 'za', 'zad', 'zal', 'zh']
labels_nums = [i for i in range(0,len(class_names))]

plate_img = cv.imread(os.path.join(plates_folder, "plate_4.jpg"))
plate_img_gr = cv.imread(os.path.join(plates_folder, "plate_4.jpg"), 0 )
plt.imshow(plate_img_gr)

from align_rect import align_rectangle

cropped_rotated_img = align_rectangle(plate_img, plate_img_gr)

#cw = cropped_rotated_img.shape[1]

plt.imshow(cropped_rotated_img)

h, w = cropped_rotated_img.shape
chopfactors = [(40, 120), (100, 200), (180, 280), (270, 360), (350, 400), (400, 460), (460, 530), (530, 600)]


plate_letters= []
for factor in chopfactors:
    w1 = int((factor[0]/600)*w)
    w2 = int((factor[1]/600)*w)
    letterpatch = cropped_rotated_img[:,w1:w2]
    #cv.imwrite(f"{str(factor[0])}_{str(factor[1])}.png", letterpatch)
    letterpatch_resized = cv.resize(letterpatch, (28,28), interpolation= cv.INTER_LINEAR)
    plate_letters.append(letterpatch_resized)

plate_letters = np.array(plate_letters)
plate_letters.shape


ocr_model = tf.keras.models.load_model('/content/gdrive/MyDrive/DL_projects_colab/models/simple_ocr_plates_model')
predictions = ocr_model.predict(plate_letters)
[class_names[k] for k in list(np.argmax(predictions, axis=1))]

#for k in list(np.argmax(predictions, axis=1))
# def rotate_image(plate_img_gr, angle):
#     (h, w) = plate_img_gr.shape
#     (cX, cY) = (w // 2, h // 2)
#     M = cv.getRotationMatrix2D((cX, cY), angle, 1.0)
#     rotated = cv.warpAffine(plate_img_gr, M, (w, h))
#     return rotated
#
# def adjust_cropping(rotated_img):
#     h,w = rotated_img.shape
#     targ_h = int(w/4)
#     crop_h = int((h - targ_h)/2)
#     cropped_rotated_img = rotated_img[crop_h:h-crop_h,:]
#     return cropped_rotated_img
# # Draw the lines on the  image
# #lines_edges = cv.addWeighted(plate_img, 0.8, line_image, 1, 0)
# def find_longest_line(plate_img_gr):
#     kernel_size = 3
#     blur_gray = cv.GaussianBlur(plate_img_gr, (kernel_size, kernel_size), 0)
#
#     low_threshold = 150
#     high_threshold = 200
#
#     edges = cv.Canny(blur_gray, low_threshold, high_threshold)
#
#     rho = 1  # distance resolution in pixels of the Hough grid
#     theta = np.pi / 180  # angular resolution in radians of the Hough grid
#     threshold = 15  # minimum number of votes (intersections in Hough grid cell)
#     min_line_length = 50  # minimum number of pixels making up a line
#     max_line_gap = 5  # maximum gap in pixels between connectable line segments
#     line_image = np.copy(plate_img) * 0  # creating a blank to draw lines on
#
#     # Run Hough on edge detected image
#     # Output "lines" is an array containing endpoints of detected line segments
#     lines = cv.HoughLinesP(edges, rho, theta, threshold, np.array([]),
#                         min_line_length, max_line_gap)
#
#     lls = []
#     for indx, line in enumerate(lines):
#         for x1,y1,x2,y2 in line:
#             cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
#             line_length = sqrt((x2-x1)**2 + (y2-y1)**2)
#             lls.append((indx,line_length))
#     lls.sort(key = lambda x: x[1])
#     linessorted = []
#     for (indx,ll) in lls:
#         linessorted.append(lines[indx])
#     return linessorted

# linessorted = find_longest_line(plate_img_gr)
# rot_angle = find_line_angle(linessorted[-1])
# rotated_img = rotate_image(plate_img_gr, rot_angle)
# cropped_rotated_img = adjust_cropping(rotated_img)

# def find_line_angle(line):
#     x1,y1,x2,y2 = line[0]
#     angle = degrees(atan(((y2-y1)/(x2-x1))))
#     return angle