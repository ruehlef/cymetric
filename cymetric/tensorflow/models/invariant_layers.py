"""Discrete-group invariant layers for the TensorFlow backend.

Transcription of ``cymetric.models.invariant_layers_ref``, which is the
framework-free specification and the oracle to test against. Follows the layer
style of Hendi, Larfors and Walden (arXiv:2407.06914), extended from the
projective and permutation redundancies to a user-supplied finite group. See
the reference module for canonicalization versus averaging.
"""

import numpy as np
import tensorflow as tf

__all__ = ["GroupCanonicalization", "GroupAveraging"]


def _blocks(ambient):
    out, start = [], 0
    for k in np.asarray(ambient).astype(int):
        out.append((start, start + int(k) + 1))
        start += int(k) + 1
    return out


def _rescale_to_patch(z, blocks):
    parts = []
    for (s, e) in blocks:
        block = z[..., s:e]
        pivot = tf.argmax(tf.abs(block), axis=-1)
        scale = tf.gather(block, pivot, axis=-1, batch_dims=len(
            block.shape) - 1)
        parts.append(block / tf.expand_dims(scale, -1))
    return tf.concat(parts, axis=-1)


class GroupCanonicalization(tf.keras.layers.Layer):
    r"""Map each point to a fixed representative of its Gamma-orbit.

    Args:
        ambient (np.array([n_ambient], np.int)): ambient dimensions.
        group_matrices (list): ``[n_coords, n_coords]`` complex matrices.
        seed (int): fixes the ordering functional.

    Inputs and outputs are in cymetric's real format
    ``[Re z_0 ... Re z_{n-1} Im z_0 ... Im z_{n-1}]``.
    """

    def __init__(self, ambient, group_matrices, seed=0, **kwargs):
        super(GroupCanonicalization, self).__init__(**kwargs)
        self.blocks = _blocks(ambient)
        mats = np.stack([np.asarray(g, dtype=np.complex128)
                         for g in group_matrices])
        self.ncoords = int(mats.shape[-1])
        self.mats = tf.constant(mats, dtype=tf.complex64)
        rng = np.random.default_rng(seed)
        self.w = tf.constant(rng.normal(size=self.ncoords), dtype=tf.float32)
        self.v = tf.constant(rng.normal(size=self.ncoords), dtype=tf.float32)

    def call(self, inputs):
        n = self.ncoords
        real_in, imag_in = tf.split(inputs, 2, axis=-1)
        z = tf.complex(real_in, imag_in)
        images = tf.stack(
            [_rescale_to_patch(tf.linalg.matmul(z, self.mats[i],
                                                transpose_b=True),
                               self.blocks)
             for i in range(self.mats.shape[0])], axis=0)
        keys = (tf.linalg.matvec(tf.math.real(images), self.w)
                + tf.linalg.matvec(tf.math.imag(images), self.v))
        pick = tf.argmin(keys, axis=0, output_type=tf.int32)
        chosen = tf.gather_nd(
            images,
            tf.stack([pick, tf.range(tf.shape(pick)[0], dtype=tf.int32)],
                     axis=-1))
        return tf.concat([tf.math.real(chosen), tf.math.imag(chosen)],
                         axis=-1)


class GroupAveraging(tf.keras.layers.Layer):
    r"""Average a sub-network over the Gamma-orbit.

    Exactly invariant and smooth, at ``|Gamma|`` evaluations. Preferred for the
    Phi-models, whose loss involves second derivatives of the network.

    Args:
        inner (tf.keras.Model): the network to average.
        ambient (np.array): ambient dimensions.
        group_matrices (list): complex matrices.
    """

    def __init__(self, inner, ambient, group_matrices, **kwargs):
        super(GroupAveraging, self).__init__(**kwargs)
        self.inner = inner
        self.blocks = _blocks(ambient)
        mats = np.stack([np.asarray(g, dtype=np.complex128)
                         for g in group_matrices])
        self.ncoords = int(mats.shape[-1])
        self.mats = tf.constant(mats, dtype=tf.complex64)

    def call(self, inputs):
        real_in, imag_in = tf.split(inputs, 2, axis=-1)
        z = tf.complex(real_in, imag_in)
        vals = []
        for i in range(self.mats.shape[0]):
            moved = _rescale_to_patch(
                tf.linalg.matmul(z, self.mats[i], transpose_b=True),
                self.blocks)
            feats = tf.concat([tf.math.real(moved), tf.math.imag(moved)],
                              axis=-1)
            vals.append(self.inner(feats))
        return tf.reduce_mean(tf.stack(vals, axis=0), axis=0)
