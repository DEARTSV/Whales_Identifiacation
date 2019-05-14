from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from os.path import join


def fetch(img_dir, name):
    img = cv2.imread(join(img_dir, name))
    if img.shape == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def resize(img, size=(1024, 768)):
    assert len(size) == 2
    return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)


def pad(img, size=(1024, 768)):
    assert len(img.shape) == 3
    assert len(size) == 2
    h, w, _ = img.shape
    assert w <= size[0] and h <= size[1]
    pad_vert = np.ceil((size[1]-h) / 2).astype(np.uint32)
    pad_hor = np.ceil((size[0]-w) / 2).astype(np.uint32)

    padded = np.zeros((size[1], size[0], 3)).astype(np.uint8)
    padded[pad_vert:pad_vert+h, pad_hor:pad_hor+w, :] = img.copy()
    return padded







