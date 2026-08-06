"""Discrete-group invariant layers for the PyTorch backend.

Transcription of ``cymetric.models.invariant_layers_ref``, which is the
framework-free specification and the oracle to test against. See that module,
or the JAX version, for the discussion of canonicalization versus averaging and
of why the Phi-models want the averaged layer.
"""

import numpy as np
import torch
import torch.nn as nn

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
        pivot = torch.argmax(torch.abs(block), dim=-1, keepdim=True)
        scale = torch.gather(block, -1, pivot)
        parts.append(block / scale)
    return torch.cat(parts, dim=-1)


class GroupCanonicalization(nn.Module):
    r"""Map each point to a fixed representative of its Gamma-orbit.

    Args:
        ambient (np.array): ambient dimensions.
        group_matrices (list): ``[n_coords, n_coords]`` complex matrices.
        seed (int): fixes the ordering functional.

    Input and output are cymetric's real layout ``[Re z, Im z]``.
    """

    def __init__(self, ambient, group_matrices, seed=0):
        super(GroupCanonicalization, self).__init__()
        self.blocks = _blocks(ambient)
        mats = np.stack([np.asarray(g, dtype=np.complex128)
                         for g in group_matrices])
        self.ncoords = mats.shape[-1]
        rng = np.random.default_rng(seed)
        self.register_buffer('mats', torch.tensor(mats, dtype=torch.cfloat))
        self.register_buffer('w', torch.tensor(rng.normal(size=self.ncoords),
                                               dtype=torch.float32))
        self.register_buffer('v', torch.tensor(rng.normal(size=self.ncoords),
                                               dtype=torch.float32))

    def forward(self, x):
        n = self.ncoords
        z = torch.complex(x[..., :n], x[..., n:])
        images = torch.stack(
            [_rescale_to_patch(z @ self.mats[i].transpose(-1, -2),
                               self.blocks)
             for i in range(self.mats.shape[0])], dim=0)
        keys = (torch.real(images) @ self.w) + (torch.imag(images) @ self.v)
        pick = torch.argmin(keys, dim=0)
        idx = pick[None, ..., None].expand(1, *images.shape[1:])
        chosen = torch.gather(images, 0, idx)[0]
        return torch.cat([torch.real(chosen), torch.imag(chosen)], dim=-1)


class GroupAveraging(nn.Module):
    r"""Average a sub-network over the Gamma-orbit.

    Exactly invariant and smooth, at ``|Gamma|`` evaluations. Preferred for the
    Phi-models, whose loss involves second derivatives of the network.

    Args:
        inner (nn.Module): the network to average.
        ambient (np.array): ambient dimensions.
        group_matrices (list): complex matrices.
    """

    def __init__(self, inner, ambient, group_matrices):
        super(GroupAveraging, self).__init__()
        self.inner = inner
        self.blocks = _blocks(ambient)
        mats = np.stack([np.asarray(g, dtype=np.complex128)
                         for g in group_matrices])
        self.ncoords = mats.shape[-1]
        self.register_buffer('mats', torch.tensor(mats, dtype=torch.cfloat))

    def forward(self, x):
        n = self.ncoords
        z = torch.complex(x[..., :n], x[..., n:])
        vals = []
        for i in range(self.mats.shape[0]):
            moved = _rescale_to_patch(z @ self.mats[i].transpose(-1, -2),
                                      self.blocks)
            feats = torch.cat([torch.real(moved), torch.imag(moved)], dim=-1)
            vals.append(self.inner(feats))
        return torch.mean(torch.stack(vals, dim=0), dim=0)
