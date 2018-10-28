import tensorflow as tf


def state_preprocess(state):
    return tf.cast(state, 'float32') / 255.


def sample_action(n_action):
    p = tf.ones([n_action], dtype='float32') / n_action
    sampler = tf.distributions.Categorical(probs=p, dtype=tf.int64)
    return sampler.sample()


def softmax(x, axis):
    e = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))
    s = tf.reduce_sum(e, axis=axis, keepdims=True)
    x = e / s
    return x


def project_distribution(target_q_dist, support, reward, done, gamma, batch_size):
    support = tf.convert_to_tensor(support)
    ndim = tf.shape(support)[0]

    tiled_support = tf.tile(support, [batch_size])
    tiled_support = tf.reshape(tiled_support, [batch_size, ndim])

    is_done_multiplier = 1. - tf.cast(done, tf.float32)
    gamma_and_done = gamma * is_done_multiplier
    target_support = reward[:, None] + gamma_and_done[:, None] * tiled_support

    target_q = tf.reduce_sum(target_q_dist * support, axis=2)
    action = tf.argmax(target_q, axis=1)

    batch_indices = tf.range(batch_size, dtype=tf.int64)
    indices = tf.stack([batch_indices, action], axis=1)

    prob = tf.gather_nd(target_q_dist, indices)

    # project target_support distribution back to support distribution
    delta_z = support[1] - support[0]

    vmin, vmax = support[0], support[-1]
    target_support = tf.clip_by_value(target_support, vmin, vmax)
    target_support_tiled = tf.tile(target_support, [1, ndim])
    target_support_tiled = tf.reshape(
        target_support_tiled, [batch_size, ndim, ndim])

    diff = tf.abs(target_support_tiled - tiled_support[:, :, None])
    quotient = tf.clip_by_value(1. - diff / delta_z, 0., 1.)

    inner_prod = quotient * prob[:, None, :]
    projection = tf.reduce_sum(inner_prod, axis=2)
    return projection
