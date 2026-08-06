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
Gamma-orbit, so any model fit to the data sees a symmetric problem.

Normalisation and volumes
-------------------------
Monte Carlo integrals in this package are means, not sums, and
``normalize_to_vol_j`` already fixes the weights so that they integrate to
``int_X J^3``. Replicating each point over its orbit therefore changes
nothing: the mean of the augmented weights equals the mean of the base
weights, so the volume of X comes out the same as from the stock generator.
No rescaling is applied here and none is needed.

The volume of the quotient is simply

    Vol(X/Gamma) = Vol(X) / |Gamma| ,

which the caller applies if that is what they want. It is documented rather
than baked into the weights, because the sample really is a sample of the
cover.

Augmentation is for training, not for integration
-------------------------------------------------
Requesting ``n`` points gives ``n`` points, of which only ``n / |Gamma|`` are
independent -- the rest are symmetry images. For fitting a model that is the
intent: the network sees a symmetric problem. For a Monte Carlo integral it is
waste, and worse, it is misleading: an error bar computed from ``n`` samples
overestimates the accuracy by a factor of ``sqrt(|Gamma|)``, because the
effective sample size is ``n / |Gamma|``.

If you want an integral, either use the stock generator, or take one
representative per orbit -- ``orbit_index`` in the returned structure says
which points are images of which, so the independent subset is
``orbit_index == 0``. The same field lets you split train and validation
without leakage: putting a point in one and its image in the other is not a
split at all, and this generator cannot prevent that on your behalf, so it
gives you what you need to do it yourself.

Conditions on Gamma, and what is checked
----------------------------------------
Three conditions must hold for X/Gamma to be a Calabi-Yau that a model can live
on, and none implies the others:

1. *Gamma preserves the holomorphic form* -- it sits in SU(3), not merely
   U(3). Otherwise Omega does not descend and the quotient is not Calabi-Yau.
   **Checked**, numerically: Omega picks up ``det(g)`` from the ambient volume
   form and the inverse of each defining polynomial's character, both of which
   are measurable on sampled points.
2. *Gamma preserves X* -- each defining polynomial is an eigenvector.
   **Checked** on generated points.
3. *The Kahler class is Gamma-invariant.* If the group permutes ambient
   factors, the corresponding Kahler parameters must agree, or the class does
   not descend. **Checked** against ``kmoduli``.

*Freeness* is **not** checked and cannot be, from the data available here: it
is a property of the particular defining polynomial, not of the degrees or the
matrices. If Gamma has fixed points on X then the quotient is singular, this
class will not notice, and the result is a metric on something that is not a
manifold. Establish freeness beforehand.

:Authors:
    Contributed for the pyCICY-X / cymetric bridge.
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
            and be closed under multiplication up to a scalar on each ambient
            factor; both are checked.
        check_invariance (bool): verify on generated points that the group
            preserves the hypersurface and the holomorphic form. Cheap, and
            catches the common error of a defining polynomial that is not
            Gamma-invariant.
        All other arguments as in :class:`CICYPointGenerator`.

    Attributes:
        gamma_order (int): |Gamma|.
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

        # Sort the identity to the front, so that orbit() can promise it and a
        # caller can rely on the first block being the independent sample.
        ident = [i for i, g in enumerate(self.group_matrices)
                 if np.allclose(g, np.eye(n))]
        if not ident:
            raise ValueError(
                'the identity is not among the group matrices, so this is not '
                'a group')
        i0 = ident[0]
        self.group_matrices = ([self.group_matrices[i0]]
                               + [g for i, g in enumerate(self.group_matrices)
                                  if i != i0])
        self.gamma_order = len(self.group_matrices)

        self._blocks = []
        start = 0
        for k in self.ambient:
            self._blocks.append((start, start + int(k) + 1))
            start += int(k) + 1

        self._check_closure()
        self._check_kmoduli()

    # -- consistency -------------------------------------------------------

    def _check_closure(self):
        """Products of group elements must be group elements.

        Equality is up to a scalar **per ambient factor**, not up to a single
        global scalar: rescaling the homogeneous coordinates of one factor is
        the identity on that projective space and says nothing about the
        others. Requiring one global scalar would reject perfectly good lifts
        of a genuine group action on the ambient space.
        """
        mats = self.group_matrices
        for a in mats:
            for b in mats:
                p = a @ b
                if not any(self._proportional_blockwise(p, c) for c in mats):
                    raise ValueError(
                        'the given matrices are not closed under '
                        'multiplication (up to a scalar on each ambient '
                        'factor), so they do not form a group action on the '
                        'ambient space')

    def _proportional_blockwise(self, a, b, tol=1e-8):
        """Whether ``a`` and ``b`` agree up to one scalar per ambient factor."""
        for (s, e) in self._blocks:
            ab, bb = a[:, s:e], b[:, s:e]
            idx = np.unravel_index(np.argmax(np.abs(bb)), bb.shape)
            if abs(bb[idx]) < tol:
                if np.max(np.abs(ab)) > tol:
                    return False
                continue
            if not np.allclose(ab, bb * (ab[idx] / bb[idx]), atol=tol):
                return False
        return True

    def _check_kmoduli(self):
        """A permuted ambient factor must carry the same Kahler parameter.

        If ``g`` maps factor i to factor j then ``t_i`` and ``t_j`` have to
        agree, or the Kahler class is not Gamma-invariant, does not descend to
        the quotient, and the metric being asked for is not the one on
        X/Gamma.
        """
        t = np.asarray(self.kmoduli)
        for g in self.group_matrices:
            perm = self._factor_permutation(g)
            if perm is None:
                continue
            for i, j in enumerate(perm):
                if abs(t[i] - t[j]) > 1e-9:
                    raise ValueError(
                        'a group element maps ambient factor %d to factor %d, '
                        'but their Kahler parameters differ (%s vs %s). The '
                        'Kahler class is then not Gamma-invariant and does '
                        'not descend to the quotient.' % (i, j, t[i], t[j]))

    def _factor_permutation(self, g, tol=1e-8):
        """Which ambient factor each factor maps to, or None if unclear."""
        perm = []
        for (s, e) in self._blocks:
            rows = np.max(np.abs(g[:, s:e]), axis=1)
            hits = [k for k, (a, b) in enumerate(self._blocks)
                    if np.max(rows[a:b]) > tol]
            if len(hits) != 1:
                return None
            perm.append(hits[0])
        return perm

    def verify_invariance(self, points, tol=1e-6):
        """Check that the group maps X to itself and preserves Omega.

        Returns ``(worst_polynomial, worst_omega)``.

        ``worst_polynomial`` is the largest ``|p(g.z)|`` over the group and the
        given points: of order one for a random defining polynomial, numerical
        zero for an invariant one.

        ``worst_omega`` measures the failure of Gamma to lie in SU(3). Omega is
        the ambient holomorphic volume form divided by the Jacobian of the
        defining polynomials, so under ``g`` it picks up ``det(g)`` divided by
        the product of the polynomials' characters. That combination must be
        one, and the deviation from one is what is returned.
        """
        worst_p = 0.0
        worst_o = 0.0
        # The character of each defining polynomial is measured on *ambient*
        # points, deliberately not on the sampled ones: those lie on X, where
        # the polynomial vanishes, so a ratio p(g.z)/p(z) there is 0/0 and
        # numerically meaningless. Off the zero set it is exact.
        rng = np.random.default_rng(0)
        probe = (rng.normal(size=(32, self.ncoords))
                 + 1j * rng.normal(size=(32, self.ncoords)))
        for g in self.group_matrices:
            moved = self._rescale_to_patch(points @ g.T)
            for a in range(self.nhyper):
                worst_p = max(worst_p,
                              float(np.max(np.abs(self._evaluate(moved, a)))))

            # Characters are read off the unrescaled images, so that the
            # projective rescaling does not contaminate them.
            chi = 1.0 + 0j
            ok = True
            for a in range(self.nhyper):
                num = self._evaluate(probe @ g.T, a)
                den = self._evaluate(probe, a)
                good = np.abs(den) > 1e-6 * np.max(np.abs(den))
                if not np.any(good):
                    ok = False
                    break
                ratios = num[good] / den[good]
                # median, not mean: the character is one number, so a robust
                # estimator is the right one and an outlier from a nearly
                # vanishing denominator should not move it
                chi = chi * np.median(ratios.real) + 0j \
                    if np.max(np.abs(ratios.imag)) < 1e-8 \
                    else chi * np.mean(ratios)
            if ok:
                worst_o = max(worst_o,
                              float(abs(np.linalg.det(g) / chi - 1.0)))
        return worst_p, worst_o

    def _evaluate(self, points, a):
        """The a-th defining polynomial at each point."""
        return np.sum(
            self.coefficients[a] * np.prod(
                points[:, None, :] ** self.monomials[a][None, :, :], axis=-1),
            axis=-1)

    # -- the group action on points ---------------------------------------

    def _rescale_to_patch(self, points):
        """Put each ambient factor back in the one-coordinate-equals-one patch.

        Applying a group element multiplies coordinates by phases and may
        permute factors, taking a point out of the patch normalisation the
        rest of the package assumes. Dividing each factor block by its largest
        entry restores it, and is a projective rescaling, so it does not move
        the point: the defining polynomials are homogeneous in each factor
        separately, so scaling a block multiplies each polynomial by a power of
        the scalar and the zero set is unchanged.
        """
        pts = np.array(points, dtype=np.complex128, copy=True)
        for (s, e) in self._blocks:
            block = pts[:, s:e]
            pivot = np.argmax(np.abs(block), axis=-1)
            scale = block[np.arange(len(block)), pivot]
            pts[:, s:e] = block / scale[:, None]
        return pts

    def orbit(self, points):
        """Every image of every point under the group.

        Returns an array of shape ``(|Gamma| * len(points), ncoords)``. The
        identity images come first -- the constructor sorts the identity to the
        front of ``group_matrices`` -- so ``result[:len(points)]`` is the
        original sample.
        """
        return np.concatenate(
            [self._rescale_to_patch(points @ g.T)
             for g in self.group_matrices], axis=0)

    # -- the generator contract -------------------------------------------

    def generate_point_weights(self, n_pw, omega=False,
                               normalize_to_vol_j=True):
        r"""As the parent, but with a Gamma-symmetric point set.

        ``n_pw`` is the number of points **returned**, as in the parent: this
        samples ``n_pw / |Gamma|`` orbits so that the total is ``n_pw``, rather
        than silently returning ``|Gamma|`` times what was asked for.

        Of those, only ``n_pw / |Gamma|`` are independent. The returned
        structure carries an extra ``orbit_index`` field, zero for the original
        sample and ``k`` for the image under the ``k``-th group element, so
        that a caller can take one representative per orbit for integration, or
        split train and validation without putting a point in one and its image
        in the other.

        Weights and volume forms are computed on the base sample and tiled.
        That is ``|Gamma|`` times cheaper than recomputing them, and it makes
        the values on an orbit agree *exactly* rather than to machine
        precision.
        """
        m = len(self.group_matrices)
        if m == 1:
            return super(EquivariantCICYPointGenerator,
                         self).generate_point_weights(
                n_pw, omega=omega, normalize_to_vol_j=normalize_to_vol_j)

        n_base = max(1, int(n_pw) // m)
        base = super(EquivariantCICYPointGenerator,
                     self).generate_point_weights(
            n_base, omega=omega, normalize_to_vol_j=normalize_to_vol_j)

        if self.check_invariance:
            worst_p, worst_o = self.verify_invariance(base['point'])
            if worst_p > 1e-6:
                raise ValueError(
                    'the group does not preserve this hypersurface: '
                    'max |p(g.z)| = %.3e over the sampled points. The '
                    'defining polynomial is probably not Gamma-invariant, in '
                    'which case the quotient does not exist and a metric '
                    'learned here would be a metric on the wrong space.'
                    % worst_p)
            if worst_o > 1e-6:
                raise ValueError(
                    'the group does not preserve the holomorphic volume form: '
                    'det(g) / prod(chi_a) deviates from 1 by %.3e. Gamma is '
                    'then in U(3) but not SU(3), Omega does not descend, and '
                    'X/Gamma is not Calabi-Yau.' % worst_o)

        pts = self.orbit(base['point'])
        dtype = list(base.dtype.descr) + [('orbit_index', np.int32)]
        out = np.zeros(len(pts), dtype=dtype)
        out['point'] = pts
        # Tiled, not recomputed: the measure is Gamma-invariant, so these are
        # equal by construction and tiling makes them equal exactly.
        out['weight'] = np.tile(base['weight'], m)
        if omega:
            out['omega'] = np.tile(base['omega'], m)
        out['orbit_index'] = np.repeat(np.arange(m, dtype=np.int32), n_base)
        return out

    @property
    def vol_quotient(self):
        """Vol(X/Gamma) = Vol(X)/|Gamma|.

        For convenience only; the weights themselves are normalised to the
        volume of the *cover*, because that is what the sample is.
        """
        return (self.get_volume_from_intersections(self.kmoduli)
                / self.gamma_order)
