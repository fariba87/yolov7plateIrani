
# train the model on dataset from Internet with these labels
# ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'D', 'Gh', 'H', 'J', 'L', 'M', 'N', 'P', 'PuV',
# 'PwD', 'Sad', 'Sin', 'T', 'Taxi', 'V', 'Y']#
# save a model for recognition
import os
import tensorflow as tf
data_path = 'F:/all/dataset char/ocr_plate_dataset'
labels = os.listdir(data_path)
# ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'D', 'Gh', 'H', 'J', 'L', 'M', 'N', 'P', 'PuV',
# 'PwD', 'Sad', 'Sin', 'T', 'Taxi', 'V', 'Y']#
print(len(labels))
print(tf.test.is_gpu_available())
all_data =[]
for label in labels:
    ch_path = os.path.join(data_path, label)
    impaths = os.listdir(ch_path)
    for i in range(len(impaths)):
        impath = os.path.join(ch_path,impaths[i])
        imlabel = label
        all_data.append((impath, imlabel))
AUTOTUNE = tf.data.AUTOTUNE
# 83844
class_names = labels
img_height= 32
img_width = 32
import tensorflow as tf
list_ds = tf.data.Dataset.list_files('F:/all/dataset char/ocr_plate_dataset/*/*', shuffle=True) #false
def get_label(file_path):
  # Convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  one_hot = parts[-2] == class_names
  # Integer encode the label
  return one_hot# tf.argmax(one_hot)
def decode_img(img):
  # Convert the compressed string to a 3D uint8 tensor
  img = tf.io.decode_jpeg(img, channels=3)
  # Resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])
def process_path(file_path):
  label2 = get_label(file_path)
  #print(type(label))
  #label1 = label.numpy()
  #label2 = tf.keras.utils.to_categorical(label1, len(labels))
  # Load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)/ 255.0
  return img, label2  # if i dont need the label just return img
ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
for image_batch, labels_batch in ds.take(1):
  print(image_batch.shape)
  print(labels_batch)
  break
image_count = len(ds)
val_size = int(image_count * 0.2)
val_ds = ds.take(val_size)
train_ds = ds.skip(val_size)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal_and_vertical"),
  tf.keras.layers.RandomRotation(0.2),
])
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=AUTOTUNE)


from model_regularized import my_model_regularized
from model_conv_simple import model
my_model = my_model_regularized

num_classes = len(labels)

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),#'sparse_categorical_crossentropy',
              metrics=['accuracy'])



opt = tf.keras.optimizers.Adam(0.001, beta_1=0.9)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.CategoricalAccuracy()
batch_size=16
epochs=10
my_model.compile(loss=loss,
                 optimizer=opt,
                 metrics=metric)
train_ds = train_ds.batch(batch_size,drop_remainder=True)
val_ds = val_ds.batch(batch_size, drop_remainder= True)
filepath='saved_model/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5'
checkpoint=tf.keras.callbacks.ModelCheckpoint(filepath,
                                              verbose=1,
                                              save_best_only=True)
earlystop=tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                           patience=30 ,
                                           verbose=1 )
lr_callback=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 patience=5,
                                                 factor=0.05,
                                                 min_delta=1e-2)
tbCallBack=tf.keras.callbacks.TensorBoard(log_dir='.\my_logs', histogram_freq=0,  write_graph=True, write_images=True)
history = my_model.fit(train_ds,
                       validation_data=val_ds,
                       batch_size=batch_size,
                       epochs=epochs,
                       #validation_data=(val_data, val_label),
                       callbacks=[checkpoint, tbCallBack, lr_callback, earlystop])
#history = model.fit(train_ds, validation_data=val_ds, epochs=100,callbacks=[checkpoint, tbCallBack, lr_callback, earlystop])
def plot_learning_curves(history):
    import matplotlib.pyplot as pyplot
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.grid()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['categorical_accuracy'], label='train')
    pyplot.plot(history.history['val_categorical_accuracy'], label='test')
    pyplot.legend()
    pyplot.grid()
    pyplot.show()
plot_learning_curves(history)
#%% callbacks
