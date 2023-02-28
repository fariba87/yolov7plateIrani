# yolo-plate-license
In this project I am going to detect and recognize the Iranian cars license plate.
[ANPR(automatic number plate recognition) application]:

the overall procedure consists of:

1) plate detection
2) plate tracking[if input is video]
3) plate number recognition(OCR)
    
As every ML project, we should consider data and model
1) Data 
    
    (as supervised, we need both data and label):

      1.1) 	for detection:

        input: car image 
	    label: bounding boxes
	    Note 1: There are many labeling tools (CVAT, LabelImg, VoTT), and new emerging online tool: roboflow[either for annotating a raw dataset or converting the existing annotations to appropriate one for the model]
	    Note 2: Yolo annotation for bounding boxes: (xcenter, ycenter, width, height)
	    Note 3: yolo predict the offset for bounding boxes
	    Note 4: yolov8 is anchor free
	    Note 5: it is an Object Detection Problem
	
      1.2) for recognition:
	    
        input: a rectangle image consisting of sequence of numbers and alphabel        
		Note: we can assume the orientation of text line is horizontal,otherwise an alignment is needed as preprocessing
		label: text(sequence of characters)
		Note: it is a sequence to sequence modeling problem(but as there is no free dataset for it(to free annotation for characters at plate level), we consider it as a classification problem)
2) Model:

	2-1) Detection:
   
       traditional : HOG, contour detection
       deep learning based:
               two stages: RCNN, FastRCNN
               one stage : YOLO series, SSD, retinanet, Hugging face' YOLOS(based on transformer)
                   1) YOLO v7
                   2) also I will try the newly released YOLOv8( January 2023) as it outperformes the previous models (also still is in developing stage)
		   Note: my Yolov8 is trained on a different dataset: as in Jupyter Notebook for yolov8  api_key="wx01Phrfycn12jSgeVb8"
		   Note: my Yolo7 is trained on the same dataset as the youtube clip

   2-2) Tracking:

       traditional: kalman filter , particle filter, 
       deep learning based:
            SORT
            deepSORT
   2-3) Recognition:

       traditional: template based
       deep learning based:
               . using already existing packages for OCR(supporting persian/arabic languages) like easyOCR, tesseract, PP-OCR , 
               . writing a model from scratch for this perpose:
                     most developed models in this area are either based on CTC, attention, transformer or the composition of these architectures.
		
			
##  Implementation: 


  1) using google colab (free GPU)	 
  2) using local machine's GPU:

	1) create a new venv[in windows]: py -m venv yolov7env
	2) activate : yolov7env\Scripts\activate
	3) clone yolov7 repo : git clone https://github.com/WongKinYiu/yolov7.git
	4) change dir : cd yolov7
	Note: since I want to train model on GPU i should change the torch installation(to gpu)-> modify the requirements.txt
	5) open requirements.txt and delete these two files: 
		#torch>=1.7.0,!=1.12.0
		#torchvision>=0.8.1,!=0.13.0
	6) create another requirement_gpu.txt file and add the following lines: 
		-i https://download.pytorch.org/whl/cu113
		torch==1.11.0+cu113
		torchvision==0.12.0+cu113
	7) install all requirements by: 
		pip install -r requirements.txt
		pip install -r requirements_gpu.txt
		pip install requiremet_extra.txt [for tracking and ocr]
	8) write python on cmd and check the availability of cuda:
		python
		import torch
		torch.cuda.is_available()
		if yes congratulation :) --> quit()  python
	9) run the command line for training model 
	






## Procedure for YOLO based Detection 
### Detection phase
#### custom object detection with YOLO:
1) Note: training should be done on GPU due to large number of parameters to be tune.
2) YOLO originally trained on COCO dataset consists of 80 classes[in case of one class object detection the mAP should be increased]
	
    you can train on your custom dataset based on the instruction in their GitHub page
	1) clone the repository and run train.py on your own dataset, with your data.yaml configuarion file
	data.yaml is the explanation of train and val directories, number of classes and name of class
	2) your dataset should be customized to data format of YOLO (image, labels) pair of image and text file with same name.
	label coordinate should be (xcenter, ycenter, w, h)
	3) Roboflow can modified your dataset to YOLO format
	you need to upload pairs of images and labels, and choose the architeture(YOLOv7 pytorch) and then roboflow will do every thing for you
	inside your workstation in roboflow(which is with limitation for free account), you can also do whatever augmentation you want
	you can also split data to train/val/test split
	thereafter, you can download the modified dataset to your local machine or use it on google colab
	4) the zip file consist of one data.yaml file that should be used during model training
	
    it was the dataset part
3) Train on custom dataset

    1) train.py also need a pretrained yolov7 model to finetune on it. you can download it from their github page
	now run the command line and weight for model to train
	[Note: you may encounter an error that all tensors are not on the same device--> there is a modification on loss.py file that you should do, just search it on issue of their github page]
	2) after finishing model training some checkpoint of the model is saved in run/train/exp/weighs directory.
	3) chooose the best.pt as your model for inferencing.
	4) you can download it to your local machine to run the model on cpu
	5) inference either can be done on cpu or gpu
	6) for inference you can run command line detect.py based on the saved model and image data you want to do detection[this case bounding boxes coordinates can be saved in a txt file]
	7) if you get txt file(NMS is already applied, you need to scale)
	or your can use your saved model in a python file, then you can do further processes
	(after applying model you should do NMS yourself and scale)
	8) the bounding box needs to be scaled to the original image sizes , since the results are normalized
	9) you can change the coordinates from (xc,yx,w,h)-> (xmin, ymin), (xmax,ymax)->draw rectangle on image or crop image for further analysis in OCR stage

Note:
	  
    since the output BB, maybe not alligned horizontally, you need to do it manually (detection lines->longest line->its orientation->rotate the image)
	there are also some ractification network for doing it in OCR papers
Now the cropped plate number will be fed to OCR
## Procedure for Optical Character Recognition 
1) publicly available OCR (they are not good for persian, specially for alligned text)
2) based on youtube :
	1) synthesize a dataset for every persian characters and numbers, create a classification model, train the model in this dataset and use the trained model for test
	2) crop the rectangle bounding box(cropped BB) in different propopotions 
       
            [(40, 120), (100, 200), (180, 280), (270, 360), (350, 400), (400, 460), (460, 530), (530, 600)]
	        12|پلاک : 10ب 234.
            (8 total character)
	3) consider every piece as a 28*28 input to the classification network and predict and concate the predictions for all character

2) our own trained OCR
	1) based on CTC
	2) based on transformer
	3) based on attention



### Extra Notes
1) what if i want to code in tensorflow?
        
        my answer:as the github repo for yolo is in pytorch, it is better to train the model with command line in torch, save the model(.pt) and export it to .onnx and then convert it to tensorflow model
	    then i can use onnx, onnxruntime to do inference on new images.
2) what if aspect ratio of rectangle is too high or large scale images[it is not the case for license plates] and small scale objects?
	    
        my answer:if using YOLOv7, try to integrate sahi algorithm[vision-library-for-performing-sliced-inference-on-large-images-small-objects:https://github.com/obss/sahi] to yolov7
		yolov8 already solve this problem and no need to sahi
3) challenges in tracking(fast moving car):
 
	    i think deepSORT is for realtime and already solve it
4) is there any improvement in yolo?
	
        my answer:i searched and found by integrating transformer architecture on yolo backbone, there would be improvement
5) challenge of limited dataset size:
	
        my answer: yolo models already use some augmentations like mosaeic,..
	 	we did some augmentation while using roboflow, and we can use more of those available options
		can use GAN to generate images? i dont know if it is applicable for this problem



### Procedure
what she did
1) for detection
  1-1) train YOLOv7 on a dataset(with augmentation: shear, flip[flip also flip the number but,not important for detection phase]) on colab
  1-2) save the model (detection model)
  what i did:
     - I trained yolov7 both on colab and locan machine on the same data as she did
     - I also trained Yolov8 on another dataset (achieved better mAP@0.5)
     
2) for recognition
  2-1) no annotation dataset available-> she generate
  2-2) generation based on some plate template, persian glyph, noise, font,and some transformation
  2-3) after creating this dataset(plate image-annotation sequence[length=8]), she tried to create another dataset since she wanted to Recognize character seperately
  2-4) based on a proportion, she seperated each number and character of every plate, and created a dataset of (char/number image-char/num annotation)
  2-5) she trained a simple Dense Classifier , and saved the model (recognition model)
  2-6) now, for each incoming image
	  image->detection model->cropped plate image->hough transormation based deskew to align the plate horizontally->
	  cropped plate image->hough transormation based deskew to align the plate horizontally-> aligned text image
	  aligned text image->seperate the whole plate image based on some proportion to 8 chops-> eight char images (as a batch)
	  eight char images (as a batch)->recognition model-> Recognition (text sequence output)
	  a .cpp file also attached in yolov8 directory for inference by c++
	  
  what i did:
     - i used a ready dataset  (char/number image-char/num annotation) in internet to train two models (1- Convolutional based classifier 2- anothe model from my github repo noisyMNIST)
     - i also wanted to train a CTC or transformer based model on the generated dataset (plate image-annotation sequence[length=8]), [i will do in future]
       now, for each incoming image
	  image->detection model->cropped plate image
	  cropped plate image->hough transormation based deskew to align the plate horizontally[i also wrote a code for text alignment based on fourier transform]-> aligned text image
	  aligned text image->seperate the whole plate image based on some proportion to 8 chops-> eight char images (as a batch)
	  eight char images (as a batch)->recognition model-> Recognition (text sequence output)
     - 
    

### How to deploy it?
1) on mobile device: convert model by tflite and use android studio for android coding and swift for iOS coding
2) on server : convert to tf.serving
3) onwindows: you can use c# and Microsoft.ML
4) on openVINO
5) on jetnano	
6) an API with django (python web framework API)
7) kerize your application
