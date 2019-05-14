from __future__ import print_function

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

val = pd.read_csv('val.csv')
#val = pd.read_csv('train.csv')
true_labels = val["Id"].values

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

mapping = np.load('../data/meta/idx_to_whales_mapping.npy').item()
pred_labels = [mapping[x][0] for x in pred]

print('true labels: \n', true_labels)
print('pred labels: \n', pred_labels)

acc = sum(true_labels == pred_labels) / len(true_labels)
print('accuracy: ', acc)
