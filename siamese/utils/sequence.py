from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
# from sklearn.utils import shuffle

from .preprocessing import fetch, resize, pad

class WhalesSequence(Sequence):
    def __init__(self, img_dir, bboxes, input_shape, x_set, y_set=None, batch_size=16):
        if y_set is not None:
            self.x, self.y = x_set, y_set
            self.dataset = pd.DataFrame(data={'x': self.x, 'y': self.y, 'used': np.zeros_like(self.y)})
            self.dataset['class_count'] = self.dataset.groupby('y')['y'].transform('count')
        else:
            self.x, self.y = x_set, None
            
        self.img_dir = img_dir
        self.bboxes = bboxes
        self.input_shape = input_shape
        self.batch_size = batch_size

        self.aug = ImageDataGenerator(rotation_range=15,
                                      width_shift_range=0.1,
                                      height_shift_range=0.1,
                                      zoom_range=0.05,
                                      channel_shift_range=50)

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        if self.y is None:
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            return np.array([self.preprocess(fetch(self.img_dir, name), name) for name in batch_x])

        n_single, n_single_not = 0, 15  # !!!!!!!!!!!!!
        unused = self.dataset.loc[self.dataset['used'] == 0]
        single_img_not = unused.loc[self.dataset['class_count'] > 1]
        if len(single_img_not) >= n_single_not:
            include_classes = unused['y'].sample(n=n_single_not).unique()
        else:
            include_classes = unused['y'].unique()

        # n_single, n_single_not = 5, 15
        # unused = self.dataset.loc[self.dataset['used'] == 0]
        # single_img = unused.loc[self.dataset['class_count'] == 1]
        # single_img_not = unused.loc[self.dataset['class_count'] > 1]
        # if len(single_img_not) >= n_single_not and len(single_img) >= n_single:
        #     good = single_img_not['y'].sample(n=n_single_not)
        #     bad = single_img['y'].sample(n=n_single)
        #     include_classes = pd.concat([good, bad], axis=0).unique()
        # else:
        #     include_classes = unused['y'].unique()

        sample_candidates = unused.loc[self.dataset['y'].isin(include_classes)]
        if len(sample_candidates) >= self.batch_size:
            batch_indices = sample_candidates.sample(n=self.batch_size).index
        elif len(unused) >= self.batch_size:
            batch_indices = unused.sample(n=self.batch_size).index
        else:
            batch_indices = unused.sample(n=self.batch_size, replace=True).index

        self.dataset.loc[batch_indices, 'used'] = 1
        batch_x = self.dataset.iloc[batch_indices]['x'].values
        batch_y = self.dataset.iloc[batch_indices]['y'].values
        return np.array([self.preprocess(fetch(self.img_dir, name), name) for name in batch_x]), np.array(batch_y)

    def preprocess(self, img, name):
        assert len(img.shape) == 3

        bbox = self.bboxes.loc[name][0]
        img = img[bbox[0]:bbox[2], bbox[1]:bbox[3]]

        h, w, _ = img.shape
        if h / w <= self.input_shape[0] / self.input_shape[1]:
            img = resize(img, (self.input_shape[1], int(self.input_shape[1] * h / w)))
        else:
            img = resize(img, (int(self.input_shape[0] * w / h), self.input_shape[0]))

        if self.y is not None:
            img = self.aug.flow(np.expand_dims(img, axis=0), batch_size=1, shuffle=False)[0][0]

        img = pad(img, (self.input_shape[1], self.input_shape[0]))
        return img / 255.  # pixel normalization

    def on_epoch_end(self):
        if self.y is not None:
            self.dataset = pd.DataFrame(data={'x': self.x, 'y': self.y, 'used': np.zeros_like(self.y)})
            self.dataset['class_count'] = self.dataset.groupby('y')['y'].transform('count')

