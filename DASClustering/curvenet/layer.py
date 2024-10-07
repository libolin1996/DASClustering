import tensorflow as tf
from tensorflow_probability import distributions

def gmm(mu, cov, tau, sx_proj, n_clusters=None, gmm_type='natural',
        trainable=False, cov_diag=None):
    """Gaussian mixture clustering."""

    if trainable is True:
        log_tau = tf.compat.v1.log(tf.nn.softmax(tau))
        n_clusters = cov.get_shape().as_list()[0]
        gm = [distributions.MultivariateNormalTriL(mu[c], scale_tril=tf.linalg.cholesky(cov[c])) for c in range(n_clusters)]
    else:
        n_clusters = cov.get_shape().as_list()[0]
        log_tau = tf.compat.v1.log(tau)
        gm = [distributions.MultivariateNormalTriL(mu[c], scale_tril=tf.linalg.cholesky(cov[c])) for c in range(n_clusters)]
    log_p = [gm[c].log_prob(sx_proj) for c in range(n_clusters)]
    cat = tf.stack([log_tau[c] + log_p[c] for c in range(n_clusters)], 1)

    if gmm_type == 'natural':
        q = tf.reduce_logsumexp(cat, axis=1)
        loss = - tf.reduce_mean(q)
    else:
        y = tf.argmax(cat, axis=1)
        y_un, _ = tf.unique(y)
        q = tf.reduce_sum(tf.stop_gradient(tf.nn.softmax(cat)) * cat, 1)
        loss = tf.map_fn(
            lambda c: tf.reduce_sum(
                q * tf.cast(tf.equal(y, tf.ones_like(y) * c),
                            tf.float32)) /
            tf.reduce_sum(
                tf.cast(tf.equal(y, tf.ones_like(y) * c), tf.float32)),
            y_un, dtype=tf.float32) * 1e3

    return loss, cat




