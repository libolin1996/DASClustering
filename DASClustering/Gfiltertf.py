import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

def Gfilter(I, D0, M, N):
    if M<N:
        u = tf.range(-M//2, M//2)
        v = tf.range(-M//2, M//2)
    else:
        u = tf.range(-N//2, N//2)
        v = tf.range(-N//2, N//2)
    U, V = tf.meshgrid(u, v)
    D = tf.abs(U) + tf.abs(V)
    n = 6
    H = 1 / (1 + tf.pow(tf.cast(D, dtype=tf.float32) / tf.cast(D0, dtype=tf.float32), tf.cast(2*n, dtype=tf.float32)))
    H = tfa.image.rotate(H, angles=3.14159265359/4)
    H = tf.expand_dims(H, axis=-1)
    H = tf.image.rot90(H) 
    H = tf.image.resize(tf.cast(tf.math.real(H), dtype=tf.float32), (M,N))
    H = tf.squeeze(H, axis=-1)
    H = tf.cast(H,dtype=tf.complex64)
    return H
