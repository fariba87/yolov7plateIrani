''' as in one virtual env we can not have both tensorflow and pytorch (as far I know),two options we have
     1) create a model for OCR in pytorch
     2) create another project in tensorflow, train the model, save the model in saved_model format -> convert to onnx--> convert to pytorch
     [I chose the second one]
     Note: As far i know to do prediction model, it is better to have inference module in the same dir as cloned YOLO model
     Note: maybe we can change the interpretor for each file seperately!
     '''
import onnx
from onnx2pytorch import ConvertModel
from onnx2torch import converter
import cv2
onnx_model = onnx.load('modeltf2onnx.onnx')
pytorch_model = ConvertModel(onnx_model)
torch_model = converter.convert(onnx_model)
impath ='F:/all/archive/chars/M/2/778.jpg'
img = cv2.imread(impath)
img = cv2.resize(img, (32,32))/255.
import numpy as np
img = np.expand_dims(img, axis=0)
import torch
img = torch.from_numpy(img).double()
# pytorch_model(img)
torch_model(img)