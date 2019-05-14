import numpy as np
import pandas as pd
import os
from core.siamese import Siamese


def test_embeddings():
    model = Siamese('dummy', input_shape=(6, 8, 3), embedding_size=3)

    model.make_embeddings('data_tiny/train', 'data_tiny/train.csv', batch_size=1)
    emb = pd.read_pickle(os.path.join(model.cache_dir, 'embeddings.pkl'))
    print(emb)

    model.predict('data_tiny/train')
    pred = pd.read_pickle(os.path.join(model.cache_dir, 'predictions.pkl'))
    print(pred)


def test_predictions():
    embeddings = pd.DataFrame(data=[
        {'Id': 0, 0: 1.0, 1: 0.0, 2: 0.0},
        {'Id': 1, 0: 0.0, 1: 1.0, 2: 0.0},
        {'Id': 2, 0: 0.0, 1: 0.0, 2: 1.0},
        {'Id': 3, 0: 1.0, 1: 0.0, 2: 1.0},
        {'Id': 4, 0: 1.0, 1: 1.0, 2: 0.0},
    ])
    print(embeddings)

    whales = pd.DataFrame(data=[
        {0: 1.0, 1: 0.0, 2: 1.0},
        {0: 1.0, 1: 1.0, 2: 0.0},
        {0: 1.0, 1: 0.0, 2: 0.0},
        {0: 0.0, 1: 1.0, 2: 0.0},
        {0: 0.0, 1: 0.0, 2: 1.0},
        {0: 1.0, 1: 0.0, 2: 1.0},
        {0: 1.0, 1: 1.0, 2: 0.0},
        {0: 1.0, 1: 0.0, 2: 0.0},
        {0: 0.0, 1: 1.0, 2: 0.0},
        {0: 0.0, 1: 0.0, 2: 1.0},
    ])
    print(whales)

    embeddings = embeddings.drop(['Id'], axis=1)
    concat = np.concatenate((embeddings, whales), axis=0)
    # print('CONCAT')
    # print(concat)
    prod = np.dot(concat, np.transpose(concat))
    # print('PROD')
    # print(prod)
    sq_norms = np.reshape(np.diag(prod), (-1, 1))
    # print('SQ_NORMS')
    # print(sq_norms)

    dist = sq_norms - 2.0 * prod + np.transpose(sq_norms)

    dist_1 = np.maximum(dist, 0.0)
    assert dist_1.all() == dist.all()
    dist = np.maximum(dist, 0.0)
    print('DIST')
    print(dist)
    dist = dist[embeddings.shape[0]:, :embeddings.shape[0]]
    print('DIST_PART')
    print(dist)

    predictions = np.apply_along_axis(np.argpartition, 1, dist, 2)
    print('PRED')
    print(predictions)


if __name__ == '__main__':
    test_predictions()

