from ultralytics import YOLO

# # Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
#
# # Use the model
# model.train(data="coco128.yaml", epochs=3)  # train the model
# metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
# success = model.export(format="onnx")  # export the model to ONNX format
from ultralytics import YOLO
import cv2
img2 = cv2.imread('B:/pythonProject/Zimage/Zimage/z29.png')

#'B:/pythonProject/Cropped/Cropped/a7.png'
#img2 = cv2.imread(cfg.image_path)
print(img2.shape)
h,w,c = img2.shape
print('original image widths and heights is ({},{})'.format(w,h))
import numpy as np
stride = int(640 * 0.5)
# number of loops to loop on the original image
wnum = np.ceil(w/stride)
image = np.zeros((640,int(wnum*stride),3))
print('image widths and heights (after zero padding) is ({},{})'.format(int(wnum*stride),640))
image[:h, :w,:] = img2[:, :,:]

num_square = int(wnum-1)

model = YOLO("B:/yolov5/best_yolov8.onnx")
print('result=',(model.predict(source=img2))[0].boxes) #[number of boxes *6]
results = (model.predict(source=img2))[0]
yy = results.boxes

#    print(type(y))
count = yy.shape[0]
res_bb =[]
if len(results)!=0:
    for i in range(count):

        x1 = int(yy[i].numpy().data[0][0])
        y1 = int(yy[i].numpy().data[0][1])
        x2 = int(yy[i].numpy().data[0][2])
        y2 = int(yy[i].numpy().data[0][3])
        res_bb.append((int(x1), int(y1), int(x2),int(y2)))
    i0 = np.reshape(np.asarray(res_bb), (-1, 4))
for l in  res_bb:
    x1,y1,x2,y2 = l
    cv2.rectangle(img2, (x1, y1),(x2, y2),color=(255,50,60), thickness=2)
cv2.imshow('image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
# yy = (model.predict(source=img2))[0].boxes
# #num_square =1
# print('number of stride to be applied on image is:', num_square)
# #stride =320
# bb_final_coord =[]
#
# for k in range(num_square):
#     print('stage=', k + 1)
#     img_temp = img2[:,k*stride: (k+2)*stride, :]
#     # res = model([img_temp])
#     # res.print()
#     # print('result', res.pandas().xyxy[0])
#     # results = res.pandas().xyxy[0].to_dict(orient="records")
#     results = (model.predict(source=img2))[0]
#     yy = results.boxes
#
# #    print(type(y))
#     count = yy.shape[0]
#     res_bb =[]
#     if len(results)!=0:
#         for i in range(count):
#
#             x1 = int(yy[i].numpy().data[0][0])
#             y1 = int(yy[i].numpy().data[0][1])
#             x2 = int(yy[i].numpy().data[0][2])
#             y2 = int(yy[i].numpy().data[0][3])
#             res_bb.append((int(x1), int(y1), int(x2),int(y2)))
#         i0 = np.reshape(np.asarray(res_bb), (-1, 4))
#
#         kk = i0.shape[0]
#         # for b in bb:
#         for k1 in range(kk):
#             x, y, w, h = i0[k1]
#             xnew = int(x) + (stride * k)  # since the coordinates were normalized
#             ynew = int(y)  # *standard_size
#             wnew = int(w) + (stride * k)
#             hnew = int(h)
#             if (xnew, ynew, wnew, hnew) not in bb_final_coord:
#                 bb_final_coord.append((xnew, ynew, wnew, hnew))
#
#
# print('list of all unique bb in original image is :', bb_final_coord)
#
# for l in  bb_final_coord:
#     x1,y1,x2,y2 = l
#     cv2.rectangle(img2, (x1, y1),(x2, y2),color=(255,50,60), thickness=2)
# cv2.imshow('image', img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()