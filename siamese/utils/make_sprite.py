import numpy as np
import pandas as pd
import cv2
import os
import math

DATA_DIR = 'D:\\garbage\\whales\\data_mnist\\visualize'


def fetch_imgs():
    names = os.listdir(os.path.join(DATA_DIR, 'img'))
    imgs = [cv2.cvtColor(cv2.imread(os.path.join(DATA_DIR, 'img', name)), cv2.COLOR_BGR2RGB) for name in names]
    return np.array(imgs).reshape((-1, 28, 28, 3))


def make_sprite(imgs):
    n, h, w, _ = imgs.shape
    n_sprite = math.ceil(math.sqrt(n))
    print(n, n_sprite)

    sprite = np.zeros((n_sprite*h, n_sprite*w, 3))

    for i, img in enumerate(imgs):
        offset_hor = (i % n_sprite) * w
        offset_vert = (i // n_sprite) * h
        sprite[offset_vert:offset_vert+h, offset_hor:offset_hor+w, :] = img.copy()
        print('%d/%d' % (i + 1, n))
    cv2.imwrite(os.path.join(DATA_DIR, 'sprite.png'), sprite)


if __name__ == '__main__':
    imgs = fetch_imgs()
    make_sprite(imgs)


