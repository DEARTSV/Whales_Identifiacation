from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, Lambda
from keras.backend import constant


def dummy(input_shape=(6, 8, 3), embedding_size=3, train_hidden_layers=False):
    img = Input(shape=input_shape)
    embeddings = Lambda(lambda x: constant([[1, 0, 0]]))(img)
    model = Model(img, embeddings)
    return model



