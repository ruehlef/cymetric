"""Reference implementation of the discrete-group invariant layers, in numpy.

The layers in ``cymetric.tensorflow.models.invariant_layers``,
``cymetric.torch.models.invariant_layers`` and
``cymetric.jax.models.invariant_layers`` are transcriptions of the functions
here. Keeping a framework-free version has two purposes: it is the
specification, and it is the oracle the backend tests compare against, so that
three implementations of the same arithmetic cannot silently drift apart.

Background
----------
A model on a quotient X/Gamma needs a Gamma-invariant metric. Two ways to get
one from a neural network:

*Canonicalization.* Map each point to a fixed representative of its
Gamma-orbit before the network sees it. Any downstream network is then exactly
invariant, at the cost of one network evaluation. This is the approach of Hendi,
Larfors and Walden (arXiv:2407.06914), whose layers canonicalize the
projective and permutation redundancies; the layer here does the same for a
user-supplied finite group.

*Averaging.* Evaluate the network on the whole orbit and average. Also exactly
invariant, and *smooth*, at the cost of ``|Gamma|`` evaluations.

The difference matters more than it looks for the Phi-models, which take
**second** derivatives of the network. Canonicalization is discontinuous where
two orbit members tie for the representative -- a measure-zero set, but the
second derivative is undefined there and large nearby. Averaging has no such
locus. Canonicalization is cheaper; averaging is smoother. Both are provided.
"""

import numpy as np


def real_to_complex(x):
    """cymetric's real layout ``[Re z, Im z]`` to complex."""
    n = x.shape[-1] // 2
    return x[..., :n] + 1j * x[..., n:]


def complex_to_real(z):
    """Complex back to cymetric's real layout."""
    return np.concatenate([np.real(z), np.imag(z)], axis=-1)


def rescale_to_patch(z, ambient):
    """Divide each ambient factor by its largest-modulus coordinate.

    A group element multiplies coordinates by phases and may permute ambient
    factors, which moves a point out of the one-coordinate-equals-one patch the
    rest of the package assumes. This restores it. Because the rescaling is
    projective and acts on one factor at a time, it does not move the point.
    """
    out = []
    start = 0
    for k in np.asarray(ambient).astype(int):
        end = start + int(k) + 1
        block = z[..., start:end]
        pivot = np.argmax(np.abs(block), axis=-1)
        scale = np.take_along_axis(block, pivot[..., None], axis=-1)
        out.append(block / scale)
        start = end
    return np.concatenate(out, axis=-1)


def orbit_key(z, seed=0):
    """A generic linear functional, used to order an orbit deterministically.

    Any function that separates the points of a finite orbit will do. A fixed
    pseudo-random real functional of the real and imaginary parts separates
    them for all but a measure-zero set of configurations, and unlike a
    lexicographic comparison it vectorizes in every backend.
    """
    n = z.shape[-1]
    rng = np.random.default_rng(seed)
    w = rng.normal(size=n)
    v = rng.normal(size=n)
    return np.real(z) @ w + np.imag(z) @ v


def canonicalize(x, ambient, group_matrices, seed=0):
    """Map each point to a fixed representative of its Gamma-orbit.

    Args:
        x: real array ``[batch, 2 * n_coords]`` in cymetric layout.
        ambient: ambient dimensions.
        group_matrices: list of ``[n_coords, n_coords]`` complex matrices.
        seed: fixes the ordering functional; any value gives an invariant
            result, and changing it changes which representative is chosen.

    Returns:
        real array of the same shape, constant on Gamma-orbits.
    """
    z = real_to_complex(np.asarray(x))
    mats = [np.asarray(g) for g in group_matrices]
    images = np.stack([rescale_to_patch(z @ g.T, ambient) for g in mats],
                      axis=0)
    keys = np.stack([orbit_key(im, seed) for im in images], axis=0)
    pick = np.argmin(keys, axis=0)
    chosen = images[pick, np.arange(images.shape[1])]
    return complex_to_real(chosen)


def average(fn, x, ambient, group_matrices):
    """Average a scalar network over the Gamma-orbit.

    ``fn`` maps ``[batch, 2 * n_coords]`` to ``[batch, ...]``. The result is
    exactly Gamma-invariant and smooth, at ``|Gamma|`` times the cost.
    """
    z = real_to_complex(np.asarray(x))
    vals = [fn(complex_to_real(rescale_to_patch(z @ np.asarray(g).T, ambient)))
            for g in group_matrices]
    return np.mean(np.stack(vals, axis=0), axis=0)
