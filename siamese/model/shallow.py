from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Activation, Add, AveragePooling2D, BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPool2D, ZeroPadding2D
from keras.initializers import glorot_uniform
from keras.models import Model

import keras.backend as K #tmp?
from keras.layers import Lambda #tmp?

def mnist_5(input_shape=(28, 28, 3), embedding_size=64, train_hidden_layers=True):
    img = Input(input_shape)

    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', name='conv_1', kernel_initializer=glorot_uniform())(img)
    x = BatchNormalization(axis=3, name='bn_1')(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name='conv_2', kernel_initializer=glorot_uniform())(x)
    x = BatchNormalization(axis=3, name='bn_2')(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv_3', kernel_initializer=glorot_uniform())(x)
    x = BatchNormalization(axis=3, name='bn_3')(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu', kernel_initializer=glorot_uniform())(x)
    x = Dense(embedding_size, kernel_initializer=glorot_uniform(), name='embeddings')(x)

    x = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)  # tmp?

    model = Model(inputs=img, outputs=x, name='shallow_mnist')
    model.summary()
    return model
