from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# def euclidean_distance(embeddings):
#     """Compute the 2D matrix of pairwise euclidean distances between embeddings.
#
#     :param embeddings: tensor of shape (batch_size, embedding_size)
#     :return dist: tensor of shape (batch_size, batch_size)
#     """
#     prod = tf.matmul(embeddings, tf.transpose(embeddings))
#     sq_norms = tf.diag_part(prod)
#
#     dist = tf.expand_dims(sq_norms, 0) - 2.0*prod + tf.expand_dims(sq_norms, 1)
#     dist = tf.maximum(dist, 0.0)
#
#     zeros_mask = tf.to_float(tf.equal(dist, 0.0))
#     dist = tf.sqrt(dist + zeros_mask*1e-16)
#     dist = dist * (1.0-zeros_mask)
#     return dist


# def valid_triplets_mask(labels):
#     """Compute the 3D boolean mask where mask[a, p, n] is True if (a, p, n) is a valid triplet,
#     as in a, p, n are distinct and labels[a] == labels[p], labels[a] != labels[n].
#
#     :param labels: tensor of shape (batch_size,)
#     :return mask: tf.bool tensor of shape (batch_size, batch_size, batch_size)
#     """
#     indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
#     indices_not_equal = tf.logical_not(indices_equal)
#     i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
#     i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
#     j_not_equal_k = tf.expand_dims(indices_not_equal, 0)
#     distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)
#
#     label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
#     i_equal_j = tf.expand_dims(label_equal, 2)
#     i_equal_k = tf.expand_dims(label_equal, 1)
#     valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))
#
#     mask = tf.logical_and(distinct_indices, valid_labels)
#     return mask


# def valid_anchor_positive_mask(labels):
#     """Compute a 2D boolean mask where mask[a, p] is True if a and p are distinct and have the same label.
#
#     :param labels: tensor of shape (batch_size,)
#     :return mask: tf.bool tensor of shape (batch_size, batch_size)
#     """
#     indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
#     indices_not_equal = tf.logical_not(indices_equal)
#
#     labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
#     mask = tf.logical_and(indices_not_equal, labels_equal)
#     return mask
#
#
# def valid_anchor_negative_mask(labels):
#     """Compute a 2D boolean mask where mask[a, n] is True if a and n have distinct label.
#
#     :param labels: tensor of shape (batch_size,)
#     :return mask: tf.bool tensor of shape (batch_size, batch_size)
#     """
#     labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
#     mask = tf.logical_not(labels_equal)
#     return mask


# def triplet_loss_(margin=0.2, strategy='batch_all', metric=euclidean_distance):
#     """Compute the triplet loss over the batch of embeddings.
#
#     :param margin: margin that is going to be enforced by the triplet loss
#     :param strategy: string, that indicated whether we're using the 'batch hard' or the 'batch all' mining strategy
#     :param metric: a callback function that we use to calculate the distance between each pair of vectors
#     :return: a callback function that calculates the loss according to the specified mining strategy
#     """
#     def batch_all(labels, embeddings):
#         """Compute the loss by generating all the valid triplets and averaging over the positive ones
#
#         :param labels: tensor of shape (batch_size,)
#         :param embeddings: tensor of shape (batch_size, embedding_size)
#         :return loss: scalar tensor
#         """
#         dist = metric(embeddings)
#
#         anchor_positive_dist = tf.expand_dims(dist, 2)
#         anchor_negative_dist = tf.expand_dims(dist, 1)
#
#         loss = anchor_positive_dist - anchor_negative_dist + margin
#
#         mask = tf.to_float(valid_triplets_mask(labels))
#         loss = tf.multiply(loss, mask)
#         loss = tf.maximum(loss, 0.0)
#
#         num_non_easy_triplets = tf.reduce_sum(tf.to_float(tf.greater(loss, 1e-16)))
#         loss = tf.reduce_sum(loss) / (num_non_easy_triplets + 1e-16)
#         return loss
#
#     def batch_hard(labels, embeddings):
#         """Compute the loss on the triplet consisting of the hardest positive and the hardest negative
#
#         :param labels: tensor of shape (batch_size,)
#         :param embeddings: tensor of shape (batch_size, embedding_size)
#         :return loss: scalar tensor
#         """
#         dist = metric(embeddings)
#
#         ap_mask = tf.to_float(valid_anchor_positive_mask(labels))
#         ap_dist = tf.multiply(dist, ap_mask)
#         hardest_positive_dist = tf.reduce_max(ap_dist, axis=1, keepdims=True)
#
#         an_mask = tf.to_float(valid_anchor_negative_mask(labels))
#         an_dist = dist + tf.reduce_max(dist, axis=1, keepdims=True) * (1.0-an_mask)
#         hardest_negative_dist = tf.reduce_min(an_dist, axis=1, keepdims=True)
#
#         loss = tf.reduce_mean(tf.maximum(hardest_positive_dist - hardest_negative_dist + margin, 0.0))
#         return loss
#
#     if strategy == 'batch_all':
#         return batch_all
#     elif strategy == 'batch_hard':
#         return batch_hard
#     else:
#         raise ValueError('unknown strategy: %s' % strategy)


# def euclidean_dist(embeddings):
#     prod = tf.matmul(embeddings, tf.transpose(embeddings))
#     #sq_norms = tf.reduce_sum(tf.square(embeddings), axis=1)
#     sq_norms = tf.diag_part(prod)
#     dist = tf.reshape(sq_norms, (-1, 1)) - 2 * prod + tf.reshape(sq_norms, (1, -1))
#     return dist
#
#
# def soft_margin_triplet_loss(labels, embeddings):
#     inf = tf.constant(1e+9, tf.float32)
#     epsilon = tf.constant(1e-6, tf.float32)
#     zero = tf.constant(0, tf.float32)
#
#     #dist = tf.sqrt(tf.maximum(zero, epsilon + euclidean_dist(embeddings)))
#     dist = tf.maximum(zero, epsilon + euclidean_dist(embeddings))  # sqeuclidean
#     # mask matrix showing equal labels of embeddings
#     equal_label_mask = tf.cast(tf.equal(tf.reshape(labels, (-1, 1)), tf.reshape(labels, (1, -1))), tf.float32)
#
#     pos_dist = tf.reduce_max(equal_label_mask * dist, axis=1)
#     neg_dist = tf.reduce_min((equal_label_mask * inf) + dist, axis=1)
#
#     loss = tf.reduce_mean(tf.nn.softplus(pos_dist - neg_dist))
#
#     #hard loss
#     #margin = tf.constant(1.5, tf.float32)
#     #loss = tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
#     return loss


# ----------------------------------------------------------------------------------------------------------------------

def valid_triplets_mask(labels):
    """Compute the 3D boolean mask where mask[a, p, n] is True if (a, p, n) is a valid triplet,
    as in a, p, n are distinct and labels[a] == labels[p], labels[a] != labels[n].

    :param labels: tensor of shape (batch_size,)
    :return mask: tf.bool tensor of shape (batch_size, batch_size, batch_size)
    """
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)
    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)
    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

    mask = tf.logical_and(distinct_indices, valid_labels)
    return mask


def euclidean_distance(embeddings, squared=False):
    """Computes pairwise euclidean distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2

    :param embeddings: 2-D Tensor of size [number of data, feature dimension].
    :param squared: Boolean, whether or not to square the pairwise distances.
    :return dist: 2-D Tensor of size [number of data, number of data].
    """
    dist_squared = tf.add(tf.reduce_sum(tf.square(embeddings), axis=1, keepdims=True),
                          tf.reduce_sum(tf.square(tf.transpose(embeddings)), axis=0, keepdims=True)
                          ) - 2.0 * tf.matmul(embeddings, tf.transpose(embeddings))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    dist_squared = tf.maximum(dist_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = tf.less_equal(dist_squared, 0.0)
    # Optionally take the sqrt.
    dist = dist_squared if squared else tf.sqrt(dist_squared + tf.cast(error_mask, dtype=tf.float32) * 1e-16)
    # Undo conditionally adding 1e-16.
    dist = tf.multiply(dist, tf.cast(tf.logical_not(error_mask), dtype=tf.float32))

    n_data = tf.shape(embeddings)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = tf.ones_like(dist) - tf.diag(tf.ones([n_data]))
    dist = tf.multiply(dist, mask_offdiagonals)
    return dist


def masked_maximum(data, mask, dim=1):
    """Computes the axis wise maximum over chosen elements.
    :param data: 2-D float `Tensor` of size [n, m].
    :param mask: 2-D Boolean `Tensor` of size [n, m].
    :param dim: The dimension over which to compute the maximum.
    :return masked_maximums: N-D `Tensor`. The maximized dimension is of size 1 after the operation.
    """
    axis_minimums = tf.reduce_min(data, axis=dim, keepdims=True)
    masked_maximums = tf.reduce_max(tf.multiply(data - axis_minimums, mask), axis=dim, keepdims=True) + axis_minimums
    return masked_maximums


def masked_minimum(data, mask, dim=1):
    """Computes the axis wise minimum over chosen elements.
    :param data: 2-D float `Tensor` of size [n, m].
    :param mask: 2-D Boolean `Tensor` of size [n, m].
    :param dim: The dimension over which to compute the minimum.
    :return masked_minimums: N-D `Tensor`. The minimized dimension is of size 1 after the operation.
    """
    axis_maximums = tf.reduce_max(data, axis=dim, keepdims=True)
    masked_minimums = tf.reduce_min(tf.multiply(data - axis_maximums, mask), axis=dim, keepdims=True) + axis_maximums
    return masked_minimums


def triplet_loss(margin=1.0, strategy='batch_semi_hard'):
    """Compute the triplet loss over the batch of embeddings. tf contrib inspired:
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/losses/python/metric_learning/metric_loss_ops.py

    :param margin: margin that is going to be enforced by the triplet loss
    :param strategy: string, that indicated whether we're using the 'batch hard', 'batch all' or 'batch_semi_hard' mining strategy
    :return: a callback function that calculates the loss according to the specified strategy
    """
    def get_loss_tensor(positive_dists, negative_dists):
        """Compute the triplet loss function tensor using specified margin:

        :param positive_dists: positive distances tensor
        :param negative_dists:  negative distances tensor
        :return: resulting triplet loss tensor
        """
        if margin == 'soft':
            return tf.nn.softplus(positive_dists - negative_dists)

        return tf.maximum(positive_dists - negative_dists + margin, 0.0)

    def batch_semi_hard(labels, embeddings):
        """Computes the triplet loss with semi-hard negative mining.
        The loss encourages the positive distances (between a pair of embeddings with
        the same labels) to be smaller than the minimum negative distance among
        which are at least greater than the positive distance plus the margin constant
        (called semi-hard negative) in the mini-batch. If no such negative exists,
        uses the largest negative distance instead.
        See: https://arxiv.org/abs/1503.03832.

        :param labels: 1-D tf.int32 `Tensor` with shape [batch_size] of multiclass integer labels.
        :param embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should be l2 normalized.
        :return loss: tf.float32 scalar.
        """
        labels = tf.reshape(labels, [-1, 1])
        batch_size = tf.size(labels)
        # Build pairwise squared distance matrix.
        dist = euclidean_distance(embeddings, squared=True)
        # Build pairwise binary adjacency matrix (equal label mask).
        adjacency = tf.equal(labels, tf.transpose(labels))
        # Invert so we can select negatives only.
        adjacency_not = tf.logical_not(adjacency)

        # Compute the mask.
        dist_tile = tf.tile(dist, [batch_size, 1])  # stack dist matrix batch_size times, axis=0
        mask = tf.logical_and(tf.tile(adjacency_not, [batch_size, 1]), tf.greater(dist_tile, tf.reshape(dist, [-1, 1])))
        mask = tf.cast(mask, dtype=tf.float32)
        is_negatives_outside = tf.reshape(tf.greater(tf.reduce_sum(mask, axis=1, keepdims=True), 0.0), [batch_size, batch_size])
        is_negatives_outside = tf.transpose(is_negatives_outside)

        # negatives_outside: smallest D_an where D_an > D_ap.
        negatives_outside = tf.reshape(masked_minimum(dist_tile, mask), [batch_size, batch_size])
        negatives_outside = tf.transpose(negatives_outside)

        # negatives_inside: largest D_an.
        adjacency_not = tf.cast(adjacency_not, dtype=tf.float32)
        negatives_inside = tf.tile(masked_maximum(dist, adjacency_not), [1, batch_size])

        semi_hard_negatives = tf.where(is_negatives_outside, negatives_outside, negatives_inside)

        # In lifted-struct, the authors multiply 0.5 for upper triangular
        #   in semihard, they take all positive pairs except the diagonal.
        mask_positives = tf.cast(adjacency, dtype=tf.float32) - tf.diag(tf.ones([batch_size]))
        n_positives = tf.reduce_sum(mask_positives)

        loss_mat = get_loss_tensor(dist, semi_hard_negatives)
        loss = tf.div_no_nan(tf.reduce_sum(tf.multiply(loss_mat, mask_positives)), n_positives)
        return loss

    def batch_all(labels, embeddings):
        """Compute the loss by generating all the valid triplets and averaging over the positive ones

        :param labels: 1-D tf.int32 `Tensor` with shape [batch_size] of multiclass integer labels.
        :param embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should be l2 normalized.
        :return loss: tf.float32 scalar.
        """
        dist = euclidean_distance(embeddings, squared=True)
        mask = tf.to_float(valid_triplets_mask(labels))

        anchor_positive_dist = tf.expand_dims(dist, 2)
        anchor_negative_dist = tf.expand_dims(dist, 1)

        loss_tensor = get_loss_tensor(anchor_positive_dist, anchor_negative_dist)
        loss_tensor = tf.multiply(loss_tensor, mask)

        num_non_easy_triplets = tf.reduce_sum(tf.to_float(tf.greater(loss_tensor, 1e-16)))
        loss = tf.div_no_nan(tf.reduce_sum(loss_tensor), num_non_easy_triplets)
        return loss

    def batch_hard(labels, embeddings):
        """Compute the loss by generating only hardest valid triplets and averaging over the positive ones.
        One triplet per embedding, i.e. per anchor

        :param labels: 1-D tf.int32 `Tensor` with shape [batch_size] of multiclass integer labels.
        :param embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should be l2 normalized.
        :return loss: tf.float32 scalar.
        """
        dist = euclidean_distance(embeddings, squared=True)
        adjacency = tf.cast(tf.equal(tf.reshape(labels, (-1, 1)), tf.reshape(labels, (1, -1))), tf.float32)

        pos_dist = tf.reduce_max(adjacency * dist, axis=1)
        inf = tf.constant(1e+9, tf.float32)
        neg_dist = tf.reduce_min((adjacency * inf) + dist, axis=1)

        loss_mat = get_loss_tensor(pos_dist, neg_dist)

        num_non_easy_triplets = tf.reduce_sum(tf.to_float(tf.greater(loss_mat, 1e-16)))
        loss = tf.div_no_nan(tf.reduce_sum(loss_mat), num_non_easy_triplets)
        return loss

    if strategy == 'batch_semi_hard':
        return batch_semi_hard
    elif strategy == 'batch hard':
        return batch_hard
    else:
        return batch_all
