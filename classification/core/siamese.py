from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import pandas as pd

import os
import math
from time import strftime, gmtime
from sklearn.neighbors import KNeighborsClassifier

from keras.optimizers import Adam, SGD, RMSprop, Adagrad
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras import applications
from keras.layers import Activation, Dense, Dropout, Flatten, Layer

from utils.sequence import WhalesSequence


class DenseCosFaceLoss(Layer):
    def __init__(self, output_dim, margin=30, scale=0.35, **kwargs):
        self.output_dim = output_dim
        self.margin = margin
        self.scale = scale
        super(DenseCosFaceLoss, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='glorot_normal',
                                      trainable=True)
        super(DenseCosFaceLoss, self).build(input_shape)

    def call(self, x):
        self.kernel = tf.nn.l2_normalize(self.kernel, 0, 1e-10)
        x = tf.nn.l2_normalize(x, 1, 1e-10)
        self.cos_t = tf.matmul(x, self.kernel)
        return self.cos_t

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def loss(self, labels, _features):
        cosine = tf.clip_by_value(self.cos_t, -1, 1) - self.margin * tf.cast(labels, tf.float32)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=self.scale * cosine))

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'm': self.margin,
                  'scale': self.scale}
        base_config = super(DenseCosFaceLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DenseArcFaceLoss(Layer):
    def __init__(self, output_dim, margin=64, scale=0.5, **kwargs):
        # margin should be in [0, pi/2)
        self.output_dim = output_dim
        self.margin = margin
        self.scale = scale
        super(DenseArcFaceLoss, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='glorot_normal',
                                      trainable=True)
        super(DenseArcFaceLoss, self).build(input_shape)

    def call(self, x):
        self.kernel = tf.nn.l2_normalize(self.kernel, 0, 1e-10)
        x = tf.nn.l2_normalize(x, 1, 1e-10)
        self.cos_t = tf.matmul(x, self.kernel)
        return self.cos_t

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def loss(self, labels, _features):
        cos_m = math.cos(self.margin)
        sin_m = math.sin(self.margin)

        # get cos(t + m) without uning arccos(cos(t)). arccos is bad computationally
        sin_t = tf.sqrt(1. - tf.square(self.cos_t))
        cos_margined = self.scale * (cos_m * self.cos_t - sin_m * sin_t)

        # for "cos(t1 + m) > cos(t2)" we want "t1 + m" to be in [0, pi]: otherwise m only makes "cos(t1 + m)" bigger,
        # and it's not what is margin for. So when cos(t) < cos(pi - m) -> cos(t + m) > pi, and in this case
        # we will just use cosface with some adaptive cosface margin built from arcface margin via m * sin(m)
        threshold = math.cos(math.pi - self.margin)
        switch_cosface = tf.to_float(self.cos_t >= threshold) * (self.cos_t - sin_m * self.margin)

        arc = self.cos_t * (1. - labels) + tf.where(switch_cosface > 0., cos_margined, switch_cosface) * labels
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=self.scale * arc))

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'm': self.margin,
                  'scale': self.scale}
        base_config = super(DenseArcFaceLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def build_model_cosface(input_shape, train_hidden_layers, n_classes, train):
    model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape, pooling='max')
    # model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    if not train_hidden_layers:
        for layer in model.layers:
            layer.trainable = False

    head = Sequential()
    # head.add(Flatten(input_shape=model.output_shape[1:]))
    head.add(Dropout(0.01))
    head.add(Activation(activation='relu'))
    head.add(Dense(128))

    if train:
        head.add(Activation(activation='relu'))
        head.add(Dropout(0.01))
        head.add(DenseCosFaceLoss(n_classes, margin=30, scale=0.35))

    model = Model(inputs=model.input, outputs=head(model.output))

    return model


def build_model_arcface(input_shape, train_hidden_layers, n_classes, train):
    model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape, pooling='max')
    # model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    if not train_hidden_layers:
        for layer in model.layers:
            layer.trainable = False

    head = Sequential()
    # head.add(Flatten(input_shape=model.output_shape[1:]))
    head.add(Dropout(0.01))
    head.add(Activation(activation='relu'))
    head.add(Dense(128))

    if train:
        head.add(Activation(activation='relu'))
        head.add(Dropout(0.01))
        head.add(DenseArcFaceLoss(n_classes, margin=60, scale=0.5))

    model = Model(inputs=model.input, outputs=head(model.output))

    return model


def build_model_classification(input_shape, train_hidden_layers, n_classes):
    # model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape, pooling='max')
    model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    if not train_hidden_layers:
        for layer in model.layers:
            layer.trainable = False

    head = Sequential()
    head.add(Flatten(input_shape=model.output_shape[1:]))
    head.add(Dropout(0.01))
    head.add(Activation(activation='relu'))
    head.add(Dense(500, activation='relu'))
    head.add(Dropout(0.01))
    head.add(Dense(n_classes, activation='softmax'))

    model = Model(inputs=model.input, outputs=head(model.output))

    return model


class Siamese(object):
    def __init__(self, input_shape=(224, 224, 3), train_hidden_layers=True, n_classes=5004, mode='classification', train=True):
        self.input_shape = input_shape
        self.predictions = None
        self.n_classes = n_classes
        self.mode = mode
        self.embeddings = None

        self.cache_dir = os.path.join('cache', strftime("cache-%y%m%d-%H%M%S", gmtime()))
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        if not os.path.isdir(os.path.join(self.cache_dir, 'training')):
            os.makedirs(os.path.join(self.cache_dir, 'training'))
        if not os.path.isdir(os.path.join(self.cache_dir, 'debug')):
            os.makedirs(os.path.join(self.cache_dir, 'debug'))

        if mode == 'classification':
            self.model = build_model_classification(input_shape, train_hidden_layers, n_classes)
        elif mode == 'cosface':
            self.model = build_model_cosface(input_shape, train_hidden_layers, n_classes, train)
        elif mode == 'arcface':
            self.model = build_model_arcface(input_shape, train_hidden_layers, n_classes, train)

    def train(self, img_dir, csv, meta_dir, epochs=10, batch_size=10, learning_rate=0.001):
        self.model.summary()

        if self.mode == 'classification':
            self.model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy')
        elif self.mode in ['cosface', 'arcface']:
            self.model.compile(optimizer=Adam(learning_rate), loss=self.model.layers[-1].layers[-1].loss)

        bboxes = pd.read_pickle(os.path.join(meta_dir, 'bboxes.pkl')).set_index('filename')
        whales_data = self._read_csv(csv, mappings_filename=os.path.join(meta_dir, 'whales_to_idx_mapping.npy'))

        # binary classifier
        # img_names, labels = whales_data[:, 0], whales_data[:, 1]
        # labels[labels != 0] = 1

        not_new_shale_idxs = np.where(whales_data[:, 1] != 0)[0]
        img_names, labels = whales_data[:, 0][not_new_shale_idxs], whales_data[:, 1][not_new_shale_idxs]  # without new_whale

        whales = WhalesSequence(img_dir, bboxes=bboxes, input_shape=self.input_shape, x_set=img_names, y_set=labels, batch_size=batch_size, n_classes=self.n_classes)
        self.model.fit_generator(whales,
                                 max_queue_size=3,
                                 shuffle=False,
                                 epochs=epochs,
                                 callbacks=[ModelCheckpoint(filepath=os.path.join(self.cache_dir, 'training', 'checkpoint-{epoch:02d}.h5'), save_weights_only=True)])
        self.model.save(os.path.join(self.cache_dir, 'final_model.h5'))
        self.save_weights(os.path.join(self.cache_dir, 'final_weights.h5'))

    def predict(self, img_dir, csv='', meta_dir='data/meta'):
        img_names = np.array(os.listdir(img_dir)) if csv == '' else pd.read_csv(csv)['Image'].values
        bboxes = pd.read_pickle(os.path.join(meta_dir, 'bboxes.pkl')).set_index('filename')
        whales_seq = WhalesSequence(img_dir, bboxes=bboxes, input_shape=self.input_shape, x_set=img_names, batch_size=1)
        pred = self.model.predict_generator(whales_seq, verbose=1)
        predictions = np.argsort(-pred, axis=1)[:, :5] + 1  # +1 to compensate 'new_whale'

        self.predictions = pd.DataFrame(data=predictions)
        self.predictions = pd.concat([pd.DataFrame(data=img_names), self.predictions], axis=1)
        self.predictions.columns = ['Image'] + list(range(5))
        self.save_predictions(os.path.join(self.cache_dir, 'predictions.pkl'))

    def predict_new_whales(self, img_dir, filename, meta_dir='data/meta'):
        #img_dir = 'data/train'
        #data = pd.read_csv('data/train.csv')[:100]

        data = pd.read_csv(filename)
        # data = pd.read_csv(filename)[:100]
        img_names = data['Image'].values
        bboxes = pd.read_pickle(os.path.join(meta_dir, 'bboxes.pkl')).set_index('filename')
        whales_seq = WhalesSequence(img_dir, bboxes=bboxes, input_shape=self.input_shape, x_set=img_names, batch_size=1)
        pred = self.model.predict_generator(whales_seq, verbose=1)

        data = data.values
        # data[:, 1] = '1 2 3 4 5'
        for i, binary_pr, in enumerate(pred):
            print(binary_pr)
            if binary_pr[0] > binary_pr[1]:
                img_predictions = ('new_whale ' + data[i, 1]).split()
                data[i, 1] = (' ').join(img_predictions)

        data = pd.DataFrame(data, columns=['Image', 'Id'])
        data.to_csv(os.path.join(self.cache_dir, 'submission.csv'), index=False, columns=['Image', 'Id'])

    def make_embeddings(self, img_dir, csv, mappings_filename='data/meta/whales_to_idx_mapping.npy', batch_size=10, meta_dir='data/meta'):
        whales_data = self._read_csv(csv, mappings_filename=mappings_filename)
        whales_data = whales_data[np.where(whales_data[:, 1] != 0)[0]]  # no need for new_whales
        bboxes = pd.read_pickle(os.path.join(meta_dir, 'bboxes.pkl')).set_index('filename')
        whales = WhalesSequence(img_dir, bboxes=bboxes, input_shape=self.input_shape, x_set=whales_data[:, 0], batch_size=batch_size)
        pred = self.model.predict_generator(whales, verbose=1)

        whales_df = pd.DataFrame(data=whales_data, columns=['Image', 'Id'])
        pred_df = pd.DataFrame(data=pred)
        pred_df = pd.concat([pred_df, whales_df], axis=1)
        pred_df = pred_df.drop(['Image'], axis=1)
        #pred_df = pred_df.groupby(['Id']).mean().reset_index()
        self.embeddings = pred_df.sort_values(by=['Id'])
        self.save_embeddings(os.path.join(self.cache_dir, 'embeddings.pkl'))

    def predict_using_embeddings(self, img_dir, csv='', meta_dir='data/meta'):
        assert self.embeddings is not None
        img_names = np.array(os.listdir(img_dir)) if csv == '' else pd.read_csv(csv)['Image'].values
        bboxes = pd.read_pickle(os.path.join(meta_dir, 'bboxes.pkl')).set_index('filename')
        whales_seq = WhalesSequence(img_dir, bboxes=bboxes, input_shape=self.input_shape, x_set=img_names, batch_size=1)
        whales = self.model.predict_generator(whales_seq, verbose=1)

        np.save(os.path.join(self.cache_dir, 'debug', 'raw_predictions'), whales)
        #whales = np.load('trained/raw_predictions.npy', allow_pickle=True)

        ids = self.embeddings['Id'].values.astype('int')
        embeddings = self.embeddings.drop(['Id'], axis=1).values

        KNN = KNeighborsClassifier(n_neighbors=50, metric='sqeuclidean', weights='distance')
        KNN.fit(embeddings, ids)

        pred = KNN.predict_proba(whales)
        predictions = np.argsort(-pred, axis=1)[:, :5] + 1  # +1 to compensate 'new_whale'

        # dists, neighbours = KNN.kneighbors(whales, n_neighbors=200)
        # neighbours_labels = ids[neighbours.flat].reshape(neighbours.shape)
        #
        # # get 5 nearest neighbours with different labels
        # predictions = np.zeros((len(whales), 5))
        # for i, labels in enumerate(neighbours_labels):
        #     j = 0
        #     prev_labels = []
        #     for label in labels:
        #         if label not in prev_labels:
        #             prev_labels.append(label)
        #             predictions[i, j] = label
        #             j += 1
        #         if j == 5:
        #             break

        # new whales nearest neighbours distance cutoff
        # threshold = 0.01
        # threshold = 0.01
        # new_whale_inds = np.where(dists[:, 0] >= threshold)[0]
        # predictions[new_whale_inds, 1:] = predictions[new_whale_inds, :-1]
        # predictions[new_whale_inds, 0] = 0

        self.predictions = pd.DataFrame(data=predictions)
        self.predictions = pd.concat([pd.DataFrame(data=img_names), self.predictions], axis=1)
        self.predictions.columns = ['Image'] + list(range(5))
        self.save_predictions(os.path.join(self.cache_dir, 'predictions.pkl'))

    def save_weights(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename, by_name=True, skip_mismatch=True)

    def save_embeddings(self, filename):
        self.embeddings.to_pickle(filename)

    def load_embeddings(self, filename):
        self.embeddings = pd.read_pickle(filename)

    def save_predictions(self, filename):
        self.predictions.to_pickle(filename)

    def load_predictions(self, filename):
        self.predictions = pd.read_pickle(filename)

    def _read_csv(self, csv, write_mappings=False, mappings_filename=None):
        csv_data = pd.read_csv(csv)
        if mappings_filename is not None:
            mapping = np.load(mappings_filename, allow_pickle=True).item()
        else:
            mapping = {}
            reverse_mapping = {}
            whales = np.sort(csv_data['Id'].unique())
            for i, w in enumerate(whales):
                mapping[w] = i
                reverse_mapping[i] = [w]
            if write_mappings:
                np.save(os.path.join(self.cache_dir, 'whales_to_idx_mapping'), mapping)
                np.save(os.path.join(self.cache_dir, 'idx_to_whales_mapping'), reverse_mapping)
        data = csv_data.replace({'Id': mapping})

        return data.values

    def make_kaggle_csv(self, mapping_file):
        mapping = np.load(mapping_file, allow_pickle=True).item()
        for k in mapping:
            mapping[k] = mapping[k][0]
        predictions = self.predictions.replace({0: mapping, 1: mapping, 2: mapping, 3: mapping, 4: mapping})
        predictions['Id'] = predictions[0].astype('str') + ' ' + predictions[1].astype('str') + ' ' + predictions[2].astype('str') + ' ' + predictions[3].astype('str') + ' ' + predictions[4].astype('str')
        predictions.to_csv(os.path.join(self.cache_dir, 'submission.csv'), index=False, columns=['Image', 'Id'])






