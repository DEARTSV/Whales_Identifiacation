import numpy as np
import pandas as pd
import cv2
import os
from utils.preprocessing import fetch, pad, resize

DATA_DIR = 'D:\\IdeaProjects\\whales\\data\\train'
META_DIR = 'D:\\IdeaProjects\\whales\\data\\meta'
THUMB_SHAPE = (50, 50, 3)
NUM_ROWS = 23


def fetch_img(filename, target_shape=(50, 50, 3)):
    img = cv2.cvtColor(cv2.imread(os.path.join(DATA_DIR, filename)), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)


def preprocess(img, target_shape):
    assert len(img.shape) == 3
    h, w, _ = img.shape
    if h / w <= 0.75:
        img = resize(img, (target_shape[1], int(target_shape[1] * h / w)))
    else:
        img = resize(img, (int(target_shape[0] * w / h), target_shape[0]))
    img = pad(img, (target_shape[1], target_shape[0]))
    return img


def sample(csv):
    csv_data = pd.read_csv(csv)
    csv_data = csv_data.sample(500, random_state=42)
    csv_data = csv_data.sort_values(['Id'])
    csv_data.to_csv(os.path.join(META_DIR, 'sample.csv'), index=False)

    for i, img_name in enumerate(csv_data['Image']):
        # img = fetch_img(os.path.join(DATA_DIR, img_name), target_shape=(672, 896, 3))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess(fetch(DATA_DIR, img_name), target_shape=(672, 896, 3))
        cv2.imwrite(os.path.join(META_DIR, 'img', img_name), img)
        print('%d/%d' % (i + 1, len(csv_data['Image'])))


def make_sprite(csv):
    csv_data = pd.read_csv(csv)
    csv_data = csv_data.sort_values(['Id'])

    sprite = np.zeros((THUMB_SHAPE[0] * NUM_ROWS, THUMB_SHAPE[0] * NUM_ROWS, 3))

    for i, img_name in enumerate(csv_data['Image']):
        img = fetch_img(os.path.join(META_DIR, 'img', img_name))
        offset_hor = (i % NUM_ROWS) * THUMB_SHAPE[1]
        offset_vert = (i // NUM_ROWS) * THUMB_SHAPE[0]
        sprite[offset_vert:offset_vert + THUMB_SHAPE[0], offset_hor:offset_hor + THUMB_SHAPE[1], :] = img.copy()
        print('%d/%d' % (i + 1, len(csv_data['Image'])))
    cv2.imwrite(os.path.join(META_DIR, 'sprite.png'), sprite)
    ids = csv_data['Id']
    ids.to_csv(os.path.join(META_DIR, 'metadata.tsv'), index=False)


if __name__ == '__main__':
    sample('data/train.csv')
    make_sprite('data/meta/sample.csv')
    cv2.imwrite(os.path.join(META_DIR, 'sprite.png'), cv2.cvtColor(cv2.imread(os.path.join(META_DIR, 'sprite.png')), cv2.COLOR_BGR2RGB))




