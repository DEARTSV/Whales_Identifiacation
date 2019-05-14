from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import AveragePooling2D, BatchNormalization, Conv2D, Dense, DepthwiseConv2D, Flatten
from keras.layers import Input, ZeroPadding2D, ReLU, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dropout, Activation
from keras.initializers import glorot_uniform
from keras.models import Model

import keras.backend as K #tmp?
from keras.layers import Lambda #tmp?


def separable(x, filters_pw, block_num, strides=(1, 1), trainable=True):
    if strides != (1, 1):
        x = ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_%d' % block_num)(x)
    x = DepthwiseConv2D(kernel_size=(3, 3),
                        strides=strides,
                        padding='same' if strides == (1, 1) else 'valid',
                        use_bias=False,
                        depthwise_initializer=glorot_uniform(),
                        trainable=trainable,
                        name='conv_dw_%d' % block_num)(x)
    x = BatchNormalization(trainable=trainable, name='conv_dw_%d_bn' % block_num)(x)
    x = ReLU(6., name='conv_dw_%d_relu' % block_num)(x)

    x = Conv2D(filters=filters_pw,
               kernel_size=(1, 1),
               strides=(1, 1),
               padding='same',
               use_bias=False,
               kernel_initializer=glorot_uniform(),
               trainable=trainable,
               name='conv_pw_%d' % block_num)(x)
    x = BatchNormalization(trainable=trainable, name='conv_pw_%d_bn' % block_num)(x)
    x = ReLU(6., name='conv_pw_%d_relu' % block_num)(x)
    return x


def MobileNet(input_shape=(224, 224, 3), num_classes=1000):
    img = Input(input_shape)

    x = ZeroPadding2D(((0, 1), (0, 1)), name='conv1_pad')(img)
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='valid', use_bias=False, kernel_initializer=glorot_uniform(), name='conv1')(x)
    x = BatchNormalization(name='conv1_bn')(x)
    x = ReLU(6., name='conv1_relu')(x)

    x = separable(x, filters_pw=64, block_num=1, strides=(1, 1))

    x = separable(x, filters_pw=128, block_num=2, strides=(2, 2))
    x = separable(x, filters_pw=128, block_num=3, strides=(1, 1))

    x = separable(x, filters_pw=256, block_num=4, strides=(2, 2))
    x = separable(x, filters_pw=256, block_num=5, strides=(1, 1))

    x = separable(x, filters_pw=512, block_num=6, strides=(2, 2))
    x = separable(x, filters_pw=512, block_num=7, strides=(1, 1))
    x = separable(x, filters_pw=512, block_num=8, strides=(1, 1))
    x = separable(x, filters_pw=512, block_num=9, strides=(1, 1))
    x = separable(x, filters_pw=512, block_num=10, strides=(1, 1))
    x = separable(x, filters_pw=512, block_num=11, strides=(1, 1))

    x = separable(x, filters_pw=1024, block_num=12, strides=(2, 2))
    x = separable(x, filters_pw=1024, block_num=13, strides=(1, 1))

    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, -1), name='reshape_1')(x)
    x = Dropout(0.5, name='dropout')(x)
    x = Conv2D(num_classes, (1, 1),
               padding='same',
               name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((num_classes,), name='reshape_2')(x)

    model = Model(inputs=img, outputs=x)
    return model


def mobilenet_like(input_shape=(672, 896, 3), embedding_size=128, train_hidden_layers=True):
    img = Input(input_shape)

    x = Conv2D(32, (3, 3), strides=(3, 4), padding='valid', use_bias=False, kernel_initializer=glorot_uniform(), name='conv0')(img)
    x = BatchNormalization(name='conv0_bn')(x)
    x = ReLU(6., name='conv0_relu')(x)

    x = ZeroPadding2D(((0, 1), (0, 1)), name='conv1_pad')(x)
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='valid', use_bias=False, kernel_initializer=glorot_uniform(), name='conv1')(x)
    x = BatchNormalization(name='conv1_bn')(x)
    x = ReLU(6., name='conv1_relu')(x)

    x = separable(x, filters_pw=64, block_num=1, strides=(1, 1), trainable=train_hidden_layers)

    x = separable(x, filters_pw=128, block_num=2, strides=(2, 2), trainable=train_hidden_layers)
    x = separable(x, filters_pw=128, block_num=3, strides=(1, 1), trainable=train_hidden_layers)

    x = separable(x, filters_pw=256, block_num=4, strides=(2, 2), trainable=train_hidden_layers)
    x = separable(x, filters_pw=256, block_num=5, strides=(1, 1), trainable=train_hidden_layers)

    x = separable(x, filters_pw=512, block_num=6, strides=(2, 2), trainable=train_hidden_layers)
    x = separable(x, filters_pw=512, block_num=7, strides=(1, 1), trainable=train_hidden_layers)
    x = separable(x, filters_pw=512, block_num=8, strides=(1, 1), trainable=train_hidden_layers)
    x = separable(x, filters_pw=512, block_num=9, strides=(1, 1), trainable=train_hidden_layers)
    x = separable(x, filters_pw=512, block_num=10, strides=(1, 1), trainable=train_hidden_layers)
    x = separable(x, filters_pw=512, block_num=11, strides=(1, 1), trainable=train_hidden_layers)

    x = separable(x, filters_pw=1024, block_num=12, strides=(2, 2), trainable=train_hidden_layers)
    x = separable(x, filters_pw=1024, block_num=13, strides=(1, 1), trainable=train_hidden_layers)

    x = GlobalMaxPooling2D(name='glob_max_pool')(x)
    x = Reshape((1, 1, -1), name='reshape_1')(x)

    x = Conv2D(512, (1, 1), padding='same', name='conv_emb_1')(x)
    x = Conv2D(embedding_size, (1, 1), padding='same', name='conv_emb_2')(x)

    x = Reshape((embedding_size,), name='embeddings')(x)

    x = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)  # tmp?

    model = Model(inputs=img, outputs=x)
    return model


if __name__ == '__main__':
    model = mobilenet_like()
    model.summary()


