import numpy as np
import tensorflow as tf

from loss.triplet_loss import euclidean_distance
from loss.triplet_loss import triplet_loss
from loss.triplet_loss import valid_triplets_mask
from loss.triplet_loss import valid_anchor_positive_mask
from loss.triplet_loss import valid_anchor_negative_mask


def euclidean_distance_np(feature):
    triu = np.triu_indices(feature.shape[0], 1)
    upper_tri_pdists = np.linalg.norm(feature[triu[1]] - feature[triu[0]], axis=1)
    num_data = feature.shape[0]
    dist = np.zeros((num_data, num_data))
    dist[np.triu_indices(num_data, 1)] = upper_tri_pdists
    dist = dist + dist.T - np.diag(dist.diagonal())
    return dist


def test_euclidean_distance():
    num_data = 64
    feat_dim = 6

    embeddings = np.random.randn(num_data, feat_dim).astype(np.float32)
    embeddings[1] = embeddings[0]

    with tf.Session() as sess:
        res_np = euclidean_distance_np(embeddings)
        res_tf = sess.run(euclidean_distance(embeddings))
        assert np.allclose(res_np, res_tf)


def test_pairwise_distances_are_positive():
    num_data = 64
    feat_dim = 6

    embeddings = 1.0 + 2e-7 * np.random.randn(num_data, feat_dim).astype(np.float32)
    embeddings[1] = embeddings[0]

    with tf.Session() as sess:
        res_tf = sess.run(euclidean_distance(embeddings))
        assert np.all(res_tf >= 0.0)


def test_gradients_pairwise_distances():
    num_data = 64
    feat_dim = 6

    embeddings = np.random.randn(num_data, feat_dim).astype(np.float32)
    embeddings[1] = embeddings[0]
    embeddings[num_data - 10: num_data] = 1.0 + 2e-7 * np.random.randn(10, feat_dim)
    embeddings = tf.constant(embeddings)

    with tf.Session() as sess:
        dists = euclidean_distance(embeddings)
        grads = tf.gradients(dists, embeddings)
        g = sess.run(grads)
        assert not np.any(np.isnan(g))


def test_triplet_mask():
    num_data = 64
    num_classes = 10

    labels = np.random.randint(0, num_classes, size=num_data).astype(np.float32)

    mask_np = np.zeros((num_data, num_data, num_data))
    for i in range(num_data):
        for j in range(num_data):
            for k in range(num_data):
                distinct = (i != j and i != k and j != k)
                valid = (labels[i] == labels[j]) and (labels[i] != labels[k])
                mask_np[i, j, k] = (distinct and valid)

    mask_tf = valid_triplets_mask(labels)
    with tf.Session() as sess:
        mask_tf_val = sess.run(mask_tf)
    assert np.allclose(mask_np, mask_tf_val)


def test_anchor_positive_mask():
    num_data = 64
    num_classes = 10

    labels = np.random.randint(0, num_classes, size=num_data).astype(np.float32)

    mask_np = np.zeros((num_data, num_data))
    for i in range(num_data):
        for j in range(num_data):
            distinct = (i != j)
            valid = labels[i] == labels[j]
            mask_np[i, j] = (distinct and valid)

    mask_tf = valid_anchor_positive_mask(labels)
    with tf.Session() as sess:
        mask_tf_val = sess.run(mask_tf)
    assert np.allclose(mask_np, mask_tf_val)


def test_anchor_negative_mask():
    num_data = 64
    num_classes = 10

    labels = np.random.randint(0, num_classes, size=num_data).astype(np.float32)

    mask_np = np.zeros((num_data, num_data))
    for i in range(num_data):
        for k in range(num_data):
            distinct = (i != k)
            valid = (labels[i] != labels[k])
            mask_np[i, k] = (distinct and valid)

    mask_tf = valid_anchor_negative_mask(labels)
    with tf.Session() as sess:
        mask_tf_val = sess.run(mask_tf)
    assert np.allclose(mask_np, mask_tf_val)


def test_simple_batch_all_triplet_loss():
    num_data = 10
    feat_dim = 6
    margin = 0.2
    num_classes = 1

    embeddings = np.random.rand(num_data, feat_dim).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=num_data).astype(np.float32)

    loss_np = 0.0
    loss_tf = triplet_loss(margin, 'batch_all')(labels, embeddings)
    with tf.Session() as sess:
        loss_tf_val = sess.run(loss_tf)
    assert np.allclose(loss_np, loss_tf_val)


def test_batch_all_triplet_loss():
    num_data = 10
    feat_dim = 6
    margin = 0.2
    num_classes = 5

    embeddings = np.random.rand(num_data, feat_dim).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=num_data).astype(np.float32)

    pdist_matrix = euclidean_distance_np(embeddings)

    loss_np = 0.0
    num_positives = 0.0
    num_valid = 0.0
    for i in range(num_data):
        for j in range(num_data):
            for k in range(num_data):
                distinct = (i != j and i != k and j != k)
                valid = (labels[i] == labels[j]) and (labels[i] != labels[k])
                if distinct and valid:
                    num_valid += 1.0

                    pos_distance = pdist_matrix[i][j]
                    neg_distance = pdist_matrix[i][k]

                    loss = np.maximum(0.0, pos_distance - neg_distance + margin)
                    loss_np += loss

                    num_positives += (loss > 0)

    loss_np /= num_positives

    loss_tf = triplet_loss(margin, 'batch_all')(labels, embeddings)
    with tf.Session() as sess:
        loss_tf_val = sess.run(loss_tf)
    assert np.allclose(loss_np, loss_tf_val)


def test_batch_hard_triplet_loss():
    """Test the triplet loss with batch hard triplet mining"""
    num_data = 50
    feat_dim = 6
    margin = 0.2
    num_classes = 5

    embeddings = np.random.rand(num_data, feat_dim).astype(np.float32)
    labels = np.random.randint(0, num_classes, size=num_data).astype(np.float32)

    pdist_matrix = euclidean_distance_np(embeddings)

    loss_np = 0.0
    for i in range(num_data):
        max_pos_dist = np.max(pdist_matrix[i][labels == labels[i]])
        min_neg_dist = np.min(pdist_matrix[i][labels != labels[i]])

        loss = np.maximum(0.0, max_pos_dist - min_neg_dist + margin)
        loss_np += loss

    loss_np /= num_data

    loss_tf = triplet_loss(margin, 'batch_hard')(labels, embeddings)
    with tf.Session() as sess:
        loss_tf_val = sess.run(loss_tf)
    assert np.allclose(loss_np, loss_tf_val)


if __name__ == '__main__':
    test_euclidean_distance()
    test_pairwise_distances_are_positive()
    test_gradients_pairwise_distances()
    test_triplet_mask()
    test_anchor_positive_mask()
    test_anchor_negative_mask()
    test_simple_batch_all_triplet_loss()
    test_batch_all_triplet_loss()
    test_batch_hard_triplet_loss()


