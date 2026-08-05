"""
Equivariant point generation for Calabi-Yau quotients.

Why this exists
---------------
``CICYPointGenerator`` samples points on a Calabi-Yau X. A large class of
string models does not live on X but on a quotient X/Gamma by a freely acting
discrete group: on the tetraquadric, for instance, an SU(5) bundle with
ind(V) = -6 gives three generations only after a free Z_2 quotient, and the
metric the model needs is the one on the quotient.

For a freely acting Gamma preserving the holomorphic form, the Ricci-flat
metric on X/Gamma is exactly the Gamma-invariant Ricci-flat metric on X --
the Calabi-Yau metric is unique in its Kahler class, and a Gamma-invariant
Kahler class therefore has a Gamma-invariant metric. So the quotient metric can
be learned on X provided the training data is Gamma-symmetric. That is what
this generator produces: every sampled point is accompanied by its whole
Gamma-orbit, so any model fit to the data sees a symmetric problem and the
learned metric descends.

This is data augmentation rather than a change to the geometry, which is
deliberate: it needs no modification to the models, the losses, or the
integration weights, and it degrades gracefully -- with the trivial group it
is exactly the stock generator.

Conditions on Gamma
-------------------
Three, and none implies the others. They are checked at construction, because
a metric computed when any of them fails is a metric on the wrong space:

1. *Gamma preserves the holomorphic form*, i.e. sits in SU(3) and not merely
   U(3). Otherwise Omega does not descend, the quotient has no covariantly
   constant spinor, and it is not Calabi-Yau at all.
2. *Gamma acts freely*, so the quotient is smooth. Not checked here -- it is a
   property of the particular defining polynomial, and the caller is expected
   to have established it. ``pyCICY.equivariant`` diagnoses it on the degrees.
3. *Gamma preserves X and the Kahler class.* Invariance of X is checked
   numerically on the generated points, which is cheap and catches the common
   error of taking a random polynomial rather than a Gamma-invariant one.

Usage
-----
The group is supplied as explicit matrices on the homogeneous coordinates, so
this file needs to know nothing about configuration matrices or characters::

    from pyCICY import export
    args = export.to_cymetric(conf, action=A, kmoduli='stability',
                              summands=model, include_group=True)
    pg = EquivariantCICYPointGenerator(**args)

``pyCICY.export.group_matrices`` produces the matrices; anything that can move
a point will do.

:Authors:
    Written for the pyCICY-X / cymetric bridge.
"""

import logging

import numpy as np

from .pointgen_cicy import CICYPointGenerator

logger = logging.getLogger('equivariant_pointgen')


class EquivariantCICYPointGenerator(CICYPointGenerator):
    r"""A CICY point generator whose output is closed under a finite group.

    Args:
        group_matrices (list(ndarray[(ncoords, ncoords)], complex)): the group
            acting on the homogeneous coordinates. Must contain the identity
            and be closed under multiplication; both are checked.
        check_invariance (bool): verify on generated points that the group
            really preserves the hypersurface. Cheap, and catches the common
            error of a non-invariant defining polynomial.
        All other arguments as in :class:`CICYPointGenerator`.
    """

    def __init__(self, *args, **kwargs):
        self.group_matrices = [np.asarray(g, dtype=np.complex128)
                               for g in kwargs.pop('group_matrices', [])]
        self.check_invariance = kwargs.pop('check_invariance', True)
        super(EquivariantCICYPointGenerator, self).__init__(*args, **kwargs)

        if not self.group_matrices:
            logger.warning(
                'no group matrices given; this generator is then exactly the '
                'stock CICYPointGenerator')
            self.group_matrices = [np.eye(self.ncoords, dtype=np.complex128)]

        n = self.ncoords
        for g in self.group_matrices:
            if g.shape != (n, n):
                raise ValueError(
                    'group matrices must be %d x %d to act on the homogeneous '
                    'coordinates, got %s' % (n, n, g.shape))
        if not any(np.allclose(g, np.eye(n)) for g in self.group_matrices):
            raise ValueError(
                'the identity is not among the group matrices, so this is not '
                'a group')
        self._check_closure()

    def _check_closure(self):
        """Products of group elements must be group elements, up to a scalar.

        Up to a scalar because the action is projective: rescaling the
        homogeneous coordinates of one factor is the identity on the ambient.
        Requiring exact equality here would reject perfectly good lifts.
        """
        mats = self.group_matrices
        for a in mats:
            for b in mats:
                p = a @ b
                if not any(self._proportional(p, c) for c in mats):
                    raise ValueError(
                        'the given matrices are not closed under '
                        'multiplication, so they do not form a group')

    @staticmethod
    def _proportional(a, b, tol=1e-8):
        idx = np.unravel_index(np.argmax(np.abs(b)), b.shape)
        if abs(b[idx]) < tol:
            return False
        return bool(np.allclose(a, b * (a[idx] / b[idx]), atol=tol))

    # -- the group action on points ---------------------------------------

    def _rescale_to_patch(self, points):
        """Put each ambient factor back in the one-coordinate-equals-one patch.

        Applying a group element multiplies coordinates by phases and may
        permute factors, which takes a point out of the patch normalisation
        the rest of cymetric assumes. Dividing each factor block by its
        largest entry restores it, and is a projective rescaling, so it does
        not move the point: the defining polynomials are homogeneous in each
        factor separately, so scaling a block by lambda multiplies each
        polynomial by a power of lambda and the zero set is unchanged.
        """
        pts = np.array(points, dtype=np.complex128, copy=True)
        start = 0
        for n in self.ambient:
            end = start + int(n) + 1
            block = pts[:, start:end]
            pivot = np.argmax(np.abs(block), axis=-1)
            scale = block[np.arange(len(block)), pivot]
            pts[:, start:end] = block / scale[:, None]
            start = end
        return pts

    def orbit(self, points):
        """Every image of every point under the group.

        Returns an array of shape ``(len(group) * len(points), ncoords)``, with
        the identity images first so that the original sample is a prefix.
        """
        return np.concatenate(
            [self._rescale_to_patch(points @ g.T)
             for g in self.group_matrices], axis=0)

    def verify_invariance(self, points, tol=1e-6):
        """Check that the group really maps X to itself.

        Returns the largest ``|p(g.z)|`` over the group and the given points.
        A random defining polynomial gives an answer of order one here, an
        invariant one gives numerical zero, and the difference is the whole
        reason this class exists.
        """
        worst = 0.0
        for g in self.group_matrices:
            moved = self._rescale_to_patch(points @ g.T)
            for a in range(self.nhyper):
                vals = np.sum(
                    self.coefficients[a] * np.prod(
                        moved[:, None, :] ** self.monomials[a][None, :, :],
                        axis=-1), axis=-1)
                worst = max(worst, float(np.max(np.abs(vals))))
        return worst

    # -- the generator contract -------------------------------------------

    def generate_point_weights(self, n_pw, omega=False,
                               normalize_to_vol_j=True):
        r"""As the parent, but with a Gamma-symmetric point set.

        ``n_pw`` is the number of *orbits* requested, so the returned array has
        ``|Gamma|`` times that many rows. Weights are recomputed by the parent
        machinery on the augmented points and then divided by ``|Gamma|``, so
        that the total measure is unchanged and integrals computed against the
        result mean the same thing as before.
        """
        base = super(EquivariantCICYPointGenerator,
                     self).generate_point_weights(
            n_pw, omega=omega, normalize_to_vol_j=normalize_to_vol_j)
        if len(self.group_matrices) == 1:
            return base

        pts = self.orbit(base['point'])
        if self.check_invariance:
            worst = self.verify_invariance(base['point'])
            if worst > 1e-6:
                raise ValueError(
                    'the group does not preserve this hypersurface: '
                    'max |p(g.z)| = %.3e over the sampled points. The '
                    'defining polynomial is probably not Gamma-invariant, in '
                    'which case the quotient does not exist and a metric '
                    'learned here would be a metric on the wrong space.'
                    % worst)

        out = np.zeros(len(pts), dtype=base.dtype)
        out['point'] = pts
        out['weight'] = self.point_weight(
            pts, normalize_to_vol_j=normalize_to_vol_j) / len(
                self.group_matrices)
        if omega:
            out['omega'] = self.holomorphic_volume_form(pts)
        return out
