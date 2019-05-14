from __future__ import print_function

import pandas as pd
import numpy as np
import sys
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, '../')
from utils.sequence import WhalesSequence
from core.siamese import Siamese

csv = 'val.csv'
#csv = 'train.csv'
mode = 'cos_angular'
#mode = 'classification'
input_shape = (224, 224, 3)
img_dir = '../data/train'

val = pd.read_csv(csv)
true_labels = val["Id"].values

if mode == 'classification':
    model = Siamese(input_shape=(224, 224, 3), train_hidden_layers=True, n_classes=8)
    model.load_weights('final_weights.h5')

    img_names = val['Image'].values
    bboxes = pd.read_pickle('../data/meta/bboxes.pkl').set_index('filename')
    whales_seq = WhalesSequence(img_dir, bboxes=bboxes, input_shape=input_shape, x_set=img_names, batch_size=1)
    pred = model.model.predict_generator(whales_seq, verbose=1)
    pred_labels = np.argmax(pred, axis=1).reshape(-1)

    for i, whale in enumerate(np.sort(np.unique(true_labels))):
        true_labels[np.where(true_labels == whale)] = i

elif mode == 'cos_angular':
    embeddings = pd.read_pickle('trained/embeddings.pkl')
    labels = embeddings['Id'].values.astype('int')
    embeddings = embeddings.drop(['Id'], axis=1).values
    whales = np.load('trained/raw_predictions.npy')

    KNN = KNeighborsClassifier(n_neighbors=5, metric='sqeuclidean', weights='distance', algorithm='brute')
    KNN.fit(embeddings, labels)
    pred = KNN.predict(whales)

    # dists, neighbours = KNN.kneighbors(whales, n_neighbors=5)
    # neighbours_labels = labels[neighbours.flat].reshape(neighbours.shape)
    # pred = neighbours_labels[:, 0].flatten()

    mapping = np.load('../data/meta/idx_to_whales_mapping.npy', allow_pickle=True).item()
    pred_labels = [mapping[x][0] for x in pred]


print('true labels: \n', true_labels)
print('pred labels: \n', pred_labels)

acc = sum(true_labels == pred_labels) / len(true_labels)
print('accuracy: ', acc)
