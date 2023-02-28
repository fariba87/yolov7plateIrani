# complex classifier (based on different blocks [from my github repo noisyMNIST])

import tensorflow as tf

import numpy as np
import cv2

# from utils import my_Relu6
np.random.seed(11)
tf.random.set_seed(11)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# whether upgrade TF-->2.9(in new environment) or use custom activation function (as my TF version is 2.3)
# x=[-1, 2, 3, -4, 7]


bias = tf.keras.initializers.Constant(
    np.log(0.2))  # for last layer bias initializer (in imbalance setting/ not with data augmentation)
reg = tf.keras.regularizers.l2(0.01)  # kernel regularizer
hu_init = tf.keras.initializers.HeUniform()
zero_init = tf.keras.initializers.Zeros()
w_last_init = tf.keras.initializers.VarianceScaling(scale=5., mode='fan_avg', distribution='normal')
biass = tf.keras.initializers.LecunNormal()
ini = tf.keras.initializers.he_normal(seed=2)


def BlockC(inputs_shape, exp, outputs_shape, resize_C):
    import tensorflow as tf

    CCC = tf.keras.layers.Input(shape=inputs_shape)

    CC = tf.keras.layers.Conv2D(filters=inputs_shape[-1] * exp,
                kernel_size=(1, 1),
                kernel_initializer=hu_init,
                #   kernel_constraint=MaxNorm(5),
                kernel_regularizer=None)(CCC)

    CC = tf.keras.layers.BatchNormalization(trainable=False)(CC)

    # CC=tf.keras.layers.Lambda(my_Relu6, name='relu6_1')(CC)
    CC = tf.keras.layers.ReLU(6.0)(CC)

    #    CC=Dropout(0.15)(CC)

    CC = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                         strides=(2, 2),
                         padding='same',
                         kernel_initializer=hu_init,
                         #        kernel_constraint=MaxNorm(5),
                         kernel_regularizer=None)(CC)
    CC = tf.keras.layers.BatchNormalization(trainable=False)(CC)
    #  CC=tf.keras.layers.Lambda(my_Relu6, name='relu6_2')(CC)
    CC = tf.keras.layers.ReLU(6.0)(CC)

    CC = tf.keras.layers.Dropout(0.15)(CC)

    CC = tf.keras.layers.Conv2D(filters=outputs_shape,
                kernel_size=(1, 1),

                kernel_initializer=ini,  # try
                #  kernel_constraint=MaxNorm(5),
                kernel_regularizer=None)(CC)

    #  CC=BatchNormalization(trainable=False)(CC)
    block_c_model = tf.keras.models.Model(inputs=CCC, outputs=CC)
    # block_c_model.summary()
    # plot_model(block_c_model, show_shapes=True,expand_nested=False )
    return block_c_model


# i try this model as subclassing -but as plot_model does not support visualizing subclasses models, i tried functional one,as well
'''class BlockC_class(tf.keras.Model):
    def __init__(self, activation=my_Relu6,exp=2):
        super().__init__()
        self.activation=activation
        self.exp=exp
        #vali vooroodi inputs to call miad
       # self.conv1=Conv2D(filters=self.exp*inputs.shape[-1], kernel_size=(1,1),activation=self.activation)
        self.dconv1=DepthwiseConv2D(kernel_size=(3,3), strides=(2,2), activation=my_Relu6)
        self.conv2=Conv2D(filters=1, kernel_size=(1,1),activation=None)
    def call(self, inputs, exp):
        #Conv1=self.conv1(inputs)
        Conv1=Conv2D(filters=self.exp*inputs.shape[-1], kernel_size=(1,1),activation=self.activation)
        Dconv1=self.dconv1(Conv1)
        Conv2=self.conv2(Dconv1)
        return Conv2

def BlockC_byclass(inputs, exp):    
    #subclassing Model
    blockc=BlockC_class()
    return blockc(inputs, exp)'''


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def BlockA(inputs_shape, exp, outputs_shape, resize_A, RESIZE_SHORTCUT=None):
    import tensorflow as tf

    # functional
    shortcut = tf.keras.layers.Input(inputs_shape)  # shape=(32,32,16))
    x = shortcut
    # x=tf.keras.layers.experimental.preprocessing.Resizing(resize_A,resize_A, interpolation="bilinear", crop_to_aspect_ratio=False)(x)
    # x=tf.keras.layers.Lambda(lambda x: tf.image.resize(x,(resize_A,resize_A)), name='resize')(x)

    #   x=Conv2D(filters=filters, kernel_size=(1,1))(x)
    x = tf.keras.layers.Conv2D(filters=inputs_shape[-1] * exp,
               kernel_size=(1, 1),
               kernel_initializer=hu_init,
               #  kernel_constraint=MaxNorm(5),
               kernel_regularizer=reg)(x)

    x = tf.keras.layers.BatchNormalization(trainable=False)(x)  # with flag :trainable=False -->constant val c

    #  x=tf.keras.layers.Lambda(my_Relu6, name='relu6_1')(x)
    x = tf.keras.layers.ReLU(6.0)(x)

    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                        padding='same',
                        kernel_initializer=zero_init,
                        #                      kernel_constraint=MaxNorm(5),
                        kernel_regularizer=None)(x)
    x = tf.keras.layers.BatchNormalization(trainable=False)(x)  # with flag :trainable=False -->constant val c

    # x=tf.keras.layers.Lambda(my_Relu6, name='relu6_2')(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    #    x=Dropout(0.15)(x)
    y = tf.keras.layers.Conv2D(filters=outputs_shape,
               kernel_size=(1, 1),
               name='conv',
               kernel_constraint=tf.keras.constraints.MaxNorm(5),
               kernel_initializer=hu_init,  # try
               kernel_regularizer=None)(x)
    #  x=BatchNormalization(trainable=False)(x) # with flag :trainable=False -->constant val c

    if not RESIZE_SHORTCUT:
        z = tf.keras.layers.add([shortcut, y])

    else:
        shortcut1 = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (RESIZE_SHORTCUT, RESIZE_SHORTCUT)))(shortcut)
        z = tf.keras.layers.add([shortcut1, y])

    block_A_model = tf.keras.models.Model(inputs=shortcut, outputs=z)
    #   block_A_model.summary()
    #  plot_model(block_A_model,show_shapes=True,expand_nested=False )
    return block_A_model


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def BlockB(inputs_shape, exp, outputs_shape, resize_B=2):
    # Sequential Model
    import tensorflow as tf

    block_B_model = tf.keras.models.Sequential()
    block_B_model.add(tf.keras.layers.Conv2D(filters=inputs_shape[-1] * exp,  # activation='relu',
                             kernel_size=(1, 1),
                             kernel_constraint=tf.keras.constraints.MaxNorm(5),
                             input_shape=inputs_shape,
                             kernel_initializer=hu_init,
                             kernel_regularizer=None))
    block_B_model.add(tf.keras.layers.BatchNormalization(trainable=False))

    # block_B_model.add(tf.keras.layers.Lambda( my_Relu6, name='relu6_1') )
    block_B_model.add(tf.keras.layers.ReLU(6.0))
    block_B_model.add(tf.keras.layers.UpSampling2D(size=(resize_B, resize_B)))

    block_B_model.add(tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3),
                                      kernel_initializer=hu_init,
                                      #                                        kernel_constraint=MaxNorm(5),
                                      kernel_regularizer=None))
    block_B_model.add(tf.keras.layers.BatchNormalization(trainable=False))

    # block_B_model.add(tf.keras.layers.Lambda(lambda x: my_Relu6(x) ,name='relu6_2'))
    block_B_model.add(tf.keras.layers.ReLU(6.0))

    block_B_model.add(tf.keras.layers.Conv2D(filters=outputs_shape,
                             kernel_size=(1, 1),
                             kernel_constraint=tf.keras.constraints.MaxNorm(5),
                             kernel_initializer=hu_init,  # try
                             kernel_regularizer=None
                             ))

    block_B_model.add(tf.keras.layers.BatchNormalization(trainable=False))

    return block_B_model


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Resize_C = [8, 4, 1]
Resize_A = [6, 4]
RESIZE_SHORTCUT1 = [None, None]


def model_blocks(inputs=(16, 16, 8), iter_stage_2=1, iter_stage_3=1):
    # iter_stage_2, iter_stage_3=1,1
    X1 = tf.keras.layers.Input(inputs)  # shape=(16,16,8))
    # stage 1
    X = BlockC(inputs_shape=(16, 16, 8), exp=1, outputs_shape=8, resize_C=Resize_C[0])(X1)
    # stage 2
    for i in range(iter_stage_2):
        X = BlockC(inputs_shape=(8, 8, 8), exp=2, outputs_shape=16, resize_C=Resize_C[1])(X)  # X, exp=2, 16)
        XX = BlockA(inputs_shape=(4, 4, 16), exp=2, outputs_shape=16, resize_A=Resize_A[0], RESIZE_SHORTCUT=None)(
            X)  # (X, exp=2, 16)

    # stage 4
    X = tf.keras.layers.Dropout(0.2)(X)
    for i in range(iter_stage_3):
        XX = BlockC(inputs_shape=(4, 4, 16), exp=2, outputs_shape=24, resize_C=Resize_C[2])(X)  # (X, exp=2)
        #

        XX = BlockA(inputs_shape=(2, 2, 24), exp=2, outputs_shape=24, resize_A=Resize_A[1], RESIZE_SHORTCUT=None)(
            XX)  # (X, exp=2)
        XX = tf.keras.layers.BatchNormalization(trainable=False)(XX)
    # post_block
    Xout = BlockB(inputs_shape=(2, 2, 24), exp=2, outputs_shape=32)(XX)

    blocks_model = tf.keras.models.Model(inputs=X1, outputs=Xout)
    # blocks_model.summary()
    # plot_model(blocks_model, show_shapes=True,expand_nested=False )
    return blocks_model


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def final_model():
    import tensorflow as tf

    AA = tf.keras.layers.Input(shape=(32, 32, 3))

    X11 = tf.keras.layers.Conv2D(filters=8,
                 kernel_size=(3, 3),
                 strides=(2, 2),
                 padding='same',
                 kernel_regularizer=None,
                 #    kernel_constraint=MaxNorm(5),
                 kernel_initializer=w_last_init)(AA)
    X11 = tf.keras.layers.BatchNormalization()(X11)
    X11 = tf.keras.layers.ReLU()(X11)  # Lambda(my_Relu6, name='relu6_2')(CC)
    X11 = tf.keras.layers.Dropout(0.2)(X11)
    sub_model = model_blocks(inputs=(16, 16, 8), iter_stage_2=1, iter_stage_3=1)
    X2 = sub_model(X11)

    X2 = tf.keras.layers.GlobalAveragePooling2D()(X2)  # with GAP: 8970 parameters
    # X2=Flatten()(X2)  #with flatten 9930 parameters
    X2 = tf.keras.layers.Dropout(0.5)(X2)
    w = tf.keras.initializers.glorot_uniform()
    #   bias = tf.keras.initializers.Constant(-1)  #bias of output layer in multiclass : -log(C)
    bias = tf.keras.initializers.Constant(np.log(
        0.2))  # bias for outout layer to cope with imbalance data : bias = -np.log(proportion of samples in different class) , here : 200:1000=0.2
    X2 = tf.keras.layers.Dropout(0.2)(X2)

    X2 = tf.keras.layers.Dense(units=28,
               kernel_regularizer=reg,
               bias_initializer=bias,
               kernel_initializer=w)(X2)  # activation='softmax',bias_initializer=bias

    my_model_raw = tf.keras.models.Model(inputs=AA, outputs=X2)
    #    my_model_raw.summary()
    #    plot_model(my_model_raw, show_shapes=True,expand_nested=False )
    return my_model_raw


my_model_regularized = final_model()
my_model_regularized.summary()


# counting  the number of trainable parameters in model
def count_trainable_variables(model):
    A = 0
    for i in range(len(model.trainable_variables)):
        A += np.prod((model.trainable_variables[i].shape))
    #  A+=np.prod((my_model.get_weights()[i].shape))
    print('number of trainable parameters={}'.format(A))


count_trainable_variables(my_model_regularized)
