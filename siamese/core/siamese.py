from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import cv2

import os
import shutil
import json
from time import strftime, gmtime

from keras.optimizers import Adam, SGD, RMSprop, Adagrad
from keras.callbacks import ModelCheckpoint
from utils.callbacks import TensorBoard, ProgbarLossLogger
# from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

from model.mobilenet import mobilenet_like
from model.resnet import resnet_like_33, resnet_like_36
from model.shallow import mnist_5
from model.dummy import dummy
from loss.triplet_loss import triplet_loss
from utils.sequence import WhalesSequence

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
#
# config = tf.ConfigProto(device_count={'GPU': 0})
# config.gpu_options.per_process_gpu_memory_fraction = 0.6
# set_session(tf.Session(config=config))

class Siamese(object):
    def __init__(self, model, strategy='batch_semi_hard', input_shape=(384, 512, 3), embedding_size=128, train_hidden_layers=True):
        self.model_name = model
        self.strategy = strategy
        self.input_shape = input_shape
        self.embedding_size = embedding_size

        self.embeddings = None
        self.predictions = None

        self.cache_dir = os.path.join('cache', strftime("cache-%y%m%d-%H%M%S", gmtime()))
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        if not os.path.isdir(os.path.join(self.cache_dir, 'training')):
            os.makedirs(os.path.join(self.cache_dir, 'training'))
        if not os.path.isdir(os.path.join(self.cache_dir, 'debug')):
            os.makedirs(os.path.join(self.cache_dir, 'debug'))
        self.model = Siamese.build_model(model, input_shape, embedding_size, train_hidden_layers)

        self.write_config()

    @staticmethod
    def build_model(model_name, input_shape, embedding_size, train_hidden_layers):
        if model_name == 'mobilenet_like':
            return mobilenet_like(input_shape=input_shape, embedding_size=embedding_size,
                                  train_hidden_layers=train_hidden_layers)
        elif model_name == 'resnet_like_33':
            return resnet_like_33(input_shape=input_shape, embedding_size=embedding_size)
        elif model_name == 'resnet_like_36':
            return resnet_like_36(input_shape=input_shape, embedding_size=embedding_size)
        elif model_name == 'shallow_mnist':
            return mnist_5(input_shape=input_shape, embedding_size=embedding_size)
        elif model_name == 'dummy':
            return dummy(input_shape=input_shape, embedding_size=embedding_size)
        else:
            raise ValueError('no such model: %s' % model_name)

    @staticmethod
    def get_vis_data(meta_dir):
        names = os.listdir(os.path.join(meta_dir, 'img'))
        imgs = [cv2.cvtColor(cv2.imread(os.path.join(meta_dir, 'img', name)), cv2.COLOR_BGR2RGB) for name in names]
        return np.array(imgs).reshape((-1, 672, 896, 3))

    def train(self, img_dir, csv, meta_dir, epochs=10, batch_size=10, learning_rate=0.001, margin=0.5):
        self.model.summary()
        self.model.compile(optimizer=Adam(learning_rate), loss=triplet_loss(margin, self.strategy))

        whales_data = self._read_csv(csv, mappings_filename=os.path.join(meta_dir, 'whales_to_idx_mapping.npy'))
        img_names, labels = whales_data[:, 0], self.new_whale_to_fictive_labels(whales_data[:, 1])
        bboxes = pd.read_pickle(os.path.join(meta_dir, 'bboxes.pkl')).set_index('filename')

        # exclude classes with single img (faster start of learning)
        cntr = Counter(labels)
        mask = np.array([cntr[label] > 1 for label in labels], dtype='bool')
        img_names, labels = img_names[mask], labels[mask]

        whales = WhalesSequence(img_dir, bboxes=bboxes, input_shape=self.input_shape, x_set=img_names, y_set=labels, batch_size=batch_size)
        self.model.fit_generator(whales,
                                 shuffle=False,
                                 epochs=epochs,
                                 verbose=0,  # turn off default ProgbarLogger (it averaging losses through the epoch)
                                 callbacks=[ModelCheckpoint(filepath=os.path.join(self.cache_dir, 'training', 'checkpoint-{epoch:02d}.h5'), save_weights_only=True),
                                            ProgbarLossLogger(),
                                            TensorBoard(update_freq='epoch',
                                                        # embeddings_freq=1,
                                                        # embeddings_data=self.get_vis_data(meta_dir),
                                                        # embeddings_metadata=os.path.join(meta_dir, 'metadata.tsv'),
                                                        # embeddings_sprite=os.path.join(meta_dir, 'sprite.png'),
                                                        # embeddings_sprite_single_image_size=(50, 50),
                                                        embeddings_layer_names=['embeddings'],
                                                        log_dir=os.path.join(self.cache_dir, 'tensorboard_logs'))])
        self.model.save(os.path.join(self.cache_dir, 'final_model.h5'))
        self.save_weights(os.path.join(self.cache_dir, 'final_weights.h5'))

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

    def predict(self, img_dir, csv='', meta_dir='data/meta'):
        assert self.embeddings is not None
        img_names = np.array(os.listdir(img_dir)) if csv == '' else pd.read_csv(csv)['Image'].values
        # bboxes = pd.read_pickle(os.path.join(meta_dir, 'bboxes.pkl')).set_index('filename')
        # whales_seq = WhalesSequence(img_dir, bboxes=bboxes, input_shape=self.input_shape, x_set=img_names, batch_size=1)
        # whales = self.model.predict_generator(whales_seq, verbose=1)
        #
        # np.save(os.path.join(self.cache_dir, 'debug', 'raw_predictions'), whales)
        whales = np.load('trained/raw_predictions.npy', allow_pickle=True)

        ids = self.embeddings['Id'].values.astype('int')
        embeddings = self.embeddings.drop(['Id'], axis=1).values

        KNN = KNeighborsClassifier(n_neighbors=50, metric='sqeuclidean', weights='distance')
        KNN.fit(embeddings, ids)

        # pred = KNN.predict_proba(whales)
        # predictions = np.argsort(-pred, axis=1)[:, :5] + 1  # +1 to compensate 'new_whale'

        dists, neighbours = KNN.kneighbors(whales, n_neighbors=200)
        neighbours_labels = ids[neighbours.flat].reshape(neighbours.shape)

        # get 5 nearest neighbours with different labels
        predictions = np.zeros((len(whales), 5))
        for i, labels in enumerate(neighbours_labels):
            j = 0
            prev_labels = []
            for label in labels:
                if label not in prev_labels:
                    prev_labels.append(label)
                    predictions[i, j] = label
                    j += 1
                if j == 5:
                    break

        # new whales nearest neighbours distance cutoff
        #threshold = 0.0452 # ~ 0.276
        threshold = 0.015
        new_whale_inds = np.where(dists[:, 0] >= threshold)[0]
        predictions[new_whale_inds, 1:] = predictions[new_whale_inds, :-1]
        predictions[new_whale_inds, 0] = 0

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

    def save_state(self, filename):
        pass

    def load_state(self, filename):
        pass

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

    def new_whale_to_fictive_labels(self, labels):
        new_whale_idxs = np.where(labels == 0)[0]
        fictive_labels = -np.arange(1, len(new_whale_idxs) + 1)
        labels[new_whale_idxs] = fictive_labels
        return labels

    # @staticmethod
    # def draw_tsne(vectors):
    #     tsne = TSNE(n_components=3, verbose=1, n_iter=300, perplexity=50).fit_transform(vectors)
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(tsne[:, 0], tsne[:, 1], tsne[:, 2])
    #     plt.show()

    def make_kaggle_csv(self, mapping_file):
        mapping = np.load(mapping_file, allow_pickle=True).item()
        for k in mapping:
            mapping[k] = mapping[k][0]
        predictions = self.predictions.replace({0: mapping, 1: mapping, 2: mapping, 3: mapping, 4: mapping})
        predictions['Id'] = predictions[0].astype('str') + ' ' + predictions[1].astype('str') + ' ' + predictions[2].astype('str') + ' ' + predictions[3].astype('str') + ' ' + predictions[4].astype('str')
        predictions.to_csv(os.path.join(self.cache_dir, 'submission.csv'), index=False, columns=['Image', 'Id'])

    def make_csv(self, mapping_file):
        mapping = np.load(mapping_file, allow_pickle=True).item()
        for k in mapping:
            mapping[k] = mapping[k][0]
        predictions = self.predictions.replace({0: mapping, 1: mapping, 2: mapping, 3: mapping, 4: mapping})
        predictions['Id'] = predictions[0]
        predictions.to_csv(os.path.join(self.cache_dir, 'prediction.csv'), index=False, columns=['Image', 'Id'])

    def write_config(self):
        config = {
            'model': self.model_name,
            'input_shape': self.input_shape,
            'embedding_size': self.embedding_size,
            'strategy': self.strategy,
            'cache_dir': self.cache_dir
        }
        with open(os.path.join(self.cache_dir, 'config.json'), 'w') as f:
            json.dump(config, f)

    @staticmethod
    def restore_from_config(filename):
        with open(filename) as f:
            config = json.load(f)
            assert 'model' in config
            assert 'input_shape' in config
            assert 'embedding_size' in config
            assert 'strategy' in config
            assert 'cache_dir' in config

            model = Siamese(config['model'], config['strategy'], config['input_shape'], config['embedding_size'])

            shutil.rmtree(model.cache_dir)
            cd = config['cache_dir']
            model.cache_dir = cd

            if os.path.isfile(os.path.join(cd, 'final_weights.h5')):
                model.load_weights(os.path.join(cd, 'final_weights.h5'))
                print('loaded weights from %s' % os.path.join(cd, 'final_weights.h5'))
            elif os.path.isdir(os.path.join(cd, 'training')):
                model.load_weights(os.path.join(cd, 'training', os.listdir(os.path.join(cd, 'training'))[-1]))
                print('loaded weights from %s' % os.path.join(cd, 'training', os.listdir(os.path.join(cd, 'training'))[-1]))

            if os.path.isfile(os.path.join(cd, 'embeddings.pkl')):
                model.load_embeddings(os.path.join(cd, 'embeddings.pkl'))
                print('loaded embeddings from %s' % os.path.join(cd, 'embeddings.pkl'))
            if os.path.isfile(os.path.join(cd, 'predictions.pkl')):
                model.load_predictions(os.path.join(cd, 'predictions.pkl'))
                print('loaded predictions from %s' % os.path.join(cd, 'predictions.pkl'))

            return model






