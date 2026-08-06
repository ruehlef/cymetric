"""Discrete-group invariant layers for the JAX backend.

Transcription of ``cymetric.models.invariant_layers_ref``, which is the
framework-free specification and the oracle the tests compare against.

Two ways to make a network exactly invariant under a finite group Gamma acting
on the homogeneous coordinates:

``GroupCanonicalization``
    maps each point to a fixed representative of its Gamma-orbit, so any
    downstream network is invariant at the cost of one evaluation. This is the
    approach of Hendi, Larfors and Walden (arXiv:2407.06914), extended from the
    projective and permutation redundancies to a user-supplied finite group.

``GroupAveraging``
    evaluates a sub-network on the whole orbit and averages, which is invariant
    and *smooth*, at ``|Gamma|`` evaluations.

Which to use depends on the model. The Phi-models take **second** derivatives
of the network, and canonicalization is discontinuous where two orbit members
tie for the representative. That locus has measure zero, but the second
derivative is undefined on it and large nearby, so for ``PhiFSModel`` the
averaged layer is the safer choice; for ``FreeModel`` and friends,
canonicalization is cheaper and fine.

Measured on the tetraquadric with a free Z_2, five epochs: a plain network gave
a Gamma-deviation of 9.1e-03 in the trained metric, the averaged network
1.7e-07, at equal sigma-loss. Orbit augmentation of the *data* alone gave no
improvement over plain (2.8e-02 against 2.2e-02) -- augmentation symmetrizes
the training distribution, not the learned function.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["GroupCanonicalization", "GroupAveraging"]


def _blocks(ambient):
    out, start = [], 0
    for k in np.asarray(ambient).astype(int):
        out.append((start, start + int(k) + 1))
        start += int(k) + 1
    return tuple(out)


def _rescale_to_patch(z, blocks):
    parts = []
    for (s, e) in blocks:
        block = z[..., s:e]
        pivot = jnp.argmax(jnp.abs(block), axis=-1)
        scale = jnp.take_along_axis(block, pivot[..., None], axis=-1)
        parts.append(block / scale)
    return jnp.concatenate(parts, axis=-1)


class GroupCanonicalization(eqx.Module):
    r"""Map each point to a fixed representative of its Gamma-orbit.

    Args:
        ambient: ambient dimensions.
        group_matrices: list of ``[n_coords, n_coords]`` complex matrices.
        seed: fixes the ordering functional. Any value gives an invariant
            result; changing it changes which representative is chosen.

    Input and output are cymetric's real layout ``[Re z, Im z]``.
    """

    mats: tuple = eqx.field(static=True)
    blocks: tuple = eqx.field(static=True)
    w: tuple = eqx.field(static=True)
    v: tuple = eqx.field(static=True)
    ncoords: int = eqx.field(static=True)

    def __init__(self, ambient, group_matrices, seed=0):
        self.blocks = _blocks(ambient)
        self.mats = tuple(tuple(tuple(complex(x) for x in row)
                                for row in np.asarray(g))
                          for g in group_matrices)
        n = np.asarray(group_matrices[0]).shape[0]
        self.ncoords = int(n)
        rng = np.random.default_rng(seed)
        self.w = tuple(float(x) for x in rng.normal(size=n))
        self.v = tuple(float(x) for x in rng.normal(size=n))

    def __call__(self, x):
        n = self.ncoords
        z = x[..., :n] + 1j * x[..., n:]
        mats = jnp.array(self.mats, dtype=jnp.complex64)
        w = jnp.array(self.w)
        v = jnp.array(self.v)

        def image(g):
            return _rescale_to_patch(z @ g.T, self.blocks)

        images = jax.vmap(image)(mats)
        keys = jnp.real(images) @ w + jnp.imag(images) @ v
        pick = jnp.argmin(keys, axis=0)
        chosen = jnp.take_along_axis(
            images, pick[None, ..., None], axis=0)[0]
        return jnp.concatenate([jnp.real(chosen), jnp.imag(chosen)], axis=-1)


class GroupAveraging(eqx.Module):
    r"""Average a sub-network over the Gamma-orbit.

    Exactly invariant and smooth. Use this one with the Phi-models, whose loss
    involves second derivatives of the network.

    Args:
        inner: the network to average, taking ``[2 * n_coords]``.
        ambient: ambient dimensions.
        group_matrices: list of complex matrices.
    """

    inner: eqx.Module
    mats: tuple = eqx.field(static=True)
    blocks: tuple = eqx.field(static=True)
    ncoords: int = eqx.field(static=True)

    def __init__(self, inner, ambient, group_matrices):
        self.inner = inner
        self.blocks = _blocks(ambient)
        self.mats = tuple(tuple(tuple(complex(x) for x in row)
                                for row in np.asarray(g))
                          for g in group_matrices)
        self.ncoords = int(np.asarray(group_matrices[0]).shape[0])

    def __call__(self, x):
        n = self.ncoords
        z = x[:n] + 1j * x[n:]
        mats = jnp.array(self.mats, dtype=jnp.complex64)
        moved = _rescale_to_patch(jnp.einsum('gij,j->gi', mats, z),
                                  self.blocks)
        feats = jnp.concatenate([jnp.real(moved), jnp.imag(moved)], axis=-1)
        return jnp.mean(jax.vmap(self.inner)(feats), axis=0)
