# simple convolutional based classifier
import tensorflow as tf
num_classes = 28
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,3,1,input_shape=(32, 32,3)),
    tf.keras.layers.Conv2D(16,3,1),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.12),
    tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
])
