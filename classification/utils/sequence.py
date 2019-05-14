from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle

from .preprocessing import fetch, resize, pad

class WhalesSequence(Sequence):
    def __init__(self, img_dir, bboxes, input_shape, x_set, y_set=None, batch_size=16, n_classes=5004):
        if y_set is not None:
            #  for classification without new_whale
            y_set -= 1  # compensate new_whale
            if n_classes < 5000:
                fictive_label = 0
                for clss in np.sort(np.unique(y_set)):
                    y_set[np.where(y_set == clss)] = fictive_label
                    fictive_label += 1

            y = np.zeros((len(y_set), n_classes), dtype='float')
            for i, label in enumerate(y_set):
                y[i][label] = 1
            self.x, self.y = shuffle(x_set, y)
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
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array([self.preprocess(fetch(self.img_dir, name), name) for name in batch_x])
        if self.y is None:
            return batch_x
        else:
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            return batch_x, batch_y

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
            self.x, self.y = shuffle(self.x, self.y)

