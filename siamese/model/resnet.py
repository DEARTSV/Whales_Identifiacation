from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Activation, Add, AveragePooling2D, BatchNormalization, Conv2D, Dense, Flatten, Input, MaxPool2D, ZeroPadding2D
from keras.initializers import glorot_uniform
from keras.models import Model


def identity(x, filters, kernel_size, stage, block, regularizer=None, trainable=True):
    """Create an identity residual block

    :param x:
    :param filters: tuple of length 3 that contains the number of filters in each convolutional layer
    :param kernel_size: tuple of length 2
    :param stage: integer, goes into names of the layers
    :param block: integer, goes into names of the layers
    :param regularizer: keras regularizer to be used in the block
    :param trainable: bool, False is the layer is supposed to be frozen during training
    :return:
    """
    f1, f2, f3 = filters
    shortcut = x

    x = Conv2D(filters=f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name='conv%d_%d_a' % (stage, block),
               kernel_initializer=glorot_uniform(), kernel_regularizer=regularizer, trainable=trainable)(x)
    x = BatchNormalization(axis=3, name='bn%d_%d_a' % (stage, block))(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f2, kernel_size=kernel_size, strides=(1, 1), padding='same', name='conv%d_%d_b' % (stage, block),
               kernel_initializer=glorot_uniform(), kernel_regularizer=regularizer, trainable=trainable)(x)
    x = BatchNormalization(axis=3, name='bn%d_%d_b' % (stage, block))(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name='conv%d_%d_c' % (stage, block),
               kernel_initializer=glorot_uniform(), kernel_regularizer=regularizer, trainable=trainable)(x)
    x = BatchNormalization(axis=3, name='bn%d_%d_c' % (stage, block))(x)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def conv(x, filters, kernel_size, stage, block, strides=(2, 2), regularizer=None, trainable=True):
    """Create a convolutional residual block

    :param x:
    :param filters: tuple of length 3 that contains the number of filters in each convolutional layer
    :param kernel_size: tuple of length 2
    :param stage: integer, goes into names of the layers
    :param block: integer, goes into names of the layers
    :param strides: tuple of length 2
    :param regularizer: keras regularizer to be used in the block
    :param trainable: bool, False is the layer is supposed to be frozen during training
    :return:
    """
    f1, f2, f3 = filters
    shortcut = x

    x = Conv2D(filters=f1, kernel_size=(1, 1), strides=strides, padding='valid', name='conv%d_%d_a' % (stage, block),
               kernel_initializer=glorot_uniform(), kernel_regularizer=regularizer, trainable=trainable)(x)
    x = BatchNormalization(axis=3, name='bn%d_%d_a' % (stage, block))(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f2, kernel_size=kernel_size, strides=(1, 1), padding='same', name='conv%d_%d_b' % (stage, block),
               kernel_initializer=glorot_uniform(), kernel_regularizer=regularizer, trainable=trainable)(x)
    x = BatchNormalization(axis=3, name='bn%d_%d_b' % (stage, block))(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=f3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name='conv%d_%d_c' % (stage, block),
               kernel_initializer=glorot_uniform(), kernel_regularizer=regularizer, trainable=trainable)(x)
    x = BatchNormalization(axis=3, name='bn%d_%d_c' % (stage, block))(x)

    shortcut = Conv2D(filters=f3, kernel_size=(1, 1), strides=strides, padding='valid', name='conv%d_%d_s' % (stage, block),
                      kernel_initializer=glorot_uniform(), kernel_regularizer=regularizer, trainable=trainable)(shortcut)
    shortcut = BatchNormalization(axis=3, name='bn%d_%d_s' % (stage, block))(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=(224, 224, 3), num_classes=6):
    img = Input(input_shape)
    x = ZeroPadding2D((3, 3))(img)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1', kernel_initializer=glorot_uniform())(x)
    x = BatchNormalization(axis=3, name='bn1')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv(x, filters=[64, 64, 256], kernel_size=(3, 3), strides=(1, 1), stage=2, block=1)
    x = identity(x, filters=[64, 64, 256], kernel_size=(3, 3), stage=2, block=2)
    x = identity(x, filters=[64, 64, 256], kernel_size=(3, 3), stage=2, block=3)

    x = conv(x, filters=[128, 128, 512], kernel_size=(3, 3), stage=3, block=1)
    x = identity(x, filters=[128, 128, 512], kernel_size=(3, 3), stage=3, block=2)
    x = identity(x, filters=[128, 128, 512], kernel_size=(3, 3), stage=3, block=3)
    x = identity(x, filters=[128, 128, 512], kernel_size=(3, 3), stage=3, block=4)

    x = conv(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=4, block=1)
    x = identity(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=4, block=2)
    x = identity(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=4, block=3)
    x = identity(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=4, block=4)
    x = identity(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=4, block=5)
    x = identity(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=4, block=6)

    x = conv(x, filters=[512, 512, 2048], kernel_size=(3, 3), stage=5, block=1)
    x = identity(x, filters=[512, 512, 2048], kernel_size=(3, 3), stage=5, block=2)
    x = identity(x, filters=[512, 512, 2048], kernel_size=(3, 3), stage=5, block=3)

    x = AveragePooling2D((int(input_shape[0]//32), int(input_shape[0]//32)))(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name='fc%d' % num_classes, kernel_initializer=glorot_uniform())(x)

    model = Model(inputs=img, outputs=x, name='ResNet50')
    return model


def resnet_like_33(input_shape=(384, 512, 3), embedding_size=128):
    img = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1', kernel_initializer=glorot_uniform())(img)
    x = BatchNormalization(axis=3, name='bn1')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv(x, filters=[64, 64, 256], kernel_size=(3, 3), strides=(1, 1), stage=2, block=1)
    x = identity(x, filters=[64, 64, 256], kernel_size=(3, 3), stage=2, block=2)
    x = identity(x, filters=[64, 64, 256], kernel_size=(3, 3), stage=2, block=3)

    x = conv(x, filters=[128, 128, 512], kernel_size=(3, 3), stage=3, block=1)
    x = identity(x, filters=[128, 128, 512], kernel_size=(3, 3), stage=3, block=2)
    x = identity(x, filters=[128, 128, 512], kernel_size=(3, 3), stage=3, block=3)
    x = identity(x, filters=[128, 128, 512], kernel_size=(3, 3), stage=3, block=4)

    x = conv(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=4, block=1)
    x = conv(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=5, block=1)
    x = conv(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=6, block=1)

    x = AveragePooling2D((int(input_shape[0]//64), int(input_shape[0]//64)))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_initializer=glorot_uniform())(x)
    x = Dense(embedding_size, kernel_initializer=glorot_uniform(), name='embeddings')(x)

    model = Model(inputs=img, outputs=x, name='resnet_like_33')
    return model


def resnet_like_36(input_shape=(768, 1024, 3), embedding_size=128):
    img = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='conv1', kernel_initializer=glorot_uniform())(img)
    x = BatchNormalization(axis=3, name='bn1')(x)
    x = Activation('relu')(x)
    x = MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv(x, filters=[64, 64, 256], kernel_size=(3, 3), strides=(1, 1), stage=2, block=1)
    x = identity(x, filters=[64, 64, 256], kernel_size=(3, 3), stage=2, block=2)
    x = identity(x, filters=[64, 64, 256], kernel_size=(3, 3), stage=2, block=3)

    x = conv(x, filters=[128, 128, 512], kernel_size=(3, 3), stage=3, block=1)
    x = identity(x, filters=[128, 128, 512], kernel_size=(3, 3), stage=3, block=2)
    x = identity(x, filters=[128, 128, 512], kernel_size=(3, 3), stage=3, block=3)
    x = identity(x, filters=[128, 128, 512], kernel_size=(3, 3), stage=3, block=4)

    x = conv(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=4, block=1)
    x = conv(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=5, block=1)
    x = conv(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=6, block=1)
    x = conv(x, filters=[256, 256, 1024], kernel_size=(3, 3), stage=7, block=1)

    x = AveragePooling2D((int(input_shape[0]//128), int(input_shape[0]//128)))(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu', kernel_initializer=glorot_uniform())(x)
    x = Dense(embedding_size, kernel_initializer=glorot_uniform(), name='embeddings')(x)

    model = Model(inputs=img, outputs=x, name='ResNet_siamese')
    model.summary()
    return model





