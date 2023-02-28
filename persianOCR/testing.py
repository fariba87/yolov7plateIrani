
# test on one character



import numpy as np
import tensorflow as tf
import cv2
# weights-improvement-41-0.42.
model_path='saved_model/'
model = tf.keras.models.load_model(model_path + 'weights-improvement-41-0.42.hdf5',custom_objects={'tf': tf})
print(model)
impath ='F:/all/archive/chars/M/2/778.jpg'
img = cv2.imread(impath)
img = cv2.resize(img, (32,32))/255.
import numpy as np
img = np.expand_dims(img, axis=0)
np.argmax(model.predict(img), axis=1)
# from main import labels
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'D', 'Gh', 'H', 'J', 'L', 'M', 'N', 'P', 'PuV',
'PwD', 'Sad', 'Sin', 'T', 'Taxi', 'V', 'Y']#
num =range(28)
dict={key:val for (key,val) in zip(num,labels)}
print(len(labels))
ind = np.argmax(model.predict(img), axis=1)
print(ind)