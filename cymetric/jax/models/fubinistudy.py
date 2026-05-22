"""
Pullbacked Fubini-Study metric implemented as an Equinox Module (JAX).

Faithful line-by-line translation of the TensorFlow implementation.
Uses jax.jit (via eqx.filter_jit) for XLA compilation, analogous to @tf.function.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import itertools as it
from functools import partial
import numpy as np

from cymetric.pointgen.nphelper import generate_monomials, get_levicivita_tensor


class FSModel(eqx.Module):
    r"""FSModel implements all underlying JAX routines for pullbacks
    and computing various loss contributions.

    It is *not* intended for actual training and does not have an explicit
    training step included. It should be used to write your own custom models
    for training of CICYs. Toric hypersurfaces require some extra routines,
    which are implemented here: `cymetric.jax.models.jaxmodels.ToricModel`
    """
    # Geometry / basis data (non-trainable)
    BASIS: dict
    ncoords: int
    nProjective: int
    nfold: int
    n: list          # norm exponents for losses  (plain Python list of floats)
    degrees: jnp.ndarray
    _degrees_list: list   # Python list of ints: degrees[i] as plain Python ints (static under JIT)
    pi: jnp.ndarray
    nhyper: int
    lc: jnp.ndarray
    proj_matrix: dict
    nTransitions: int
    fixed_patches: jnp.ndarray  # may be None for nhyper > 1
    _proj_indices: jnp.ndarray
    slopes: jnp.ndarray

    def __init__(self, BASIS, norm=None):
        r"""A JAX implementation of the pulled back Fubini-Study metric.

        Args:
            BASIS (dict): a dictionary containing all monomials and other
                relevant information from e.g.
                `cymetric.pointgen.pointgen.PointGenerator`.
                All numpy arrays in BASIS will be converted to JAX arrays.
            norm ([5//NLOSS], optional): degree of norm for various losses.
                Defaults to 1 for all but Kaehler norm (2).
        """
        # Convert all numpy arrays in BASIS to JAX complex64 arrays
        new_basis = {}
        for key in BASIS:
            if isinstance(BASIS[key], np.ndarray):
                new_basis[key] = jnp.array(BASIS[key], dtype=jnp.complex64)
            else:
                new_basis[key] = BASIS[key]
        self.BASIS = new_basis

        self.ncoords = len(self.BASIS['DQDZB0'])
        self.nProjective = len(self.BASIS['AMBIENT'])
        self.nfold = int(jnp.real(self.BASIS['NFOLD']))

        if norm is None:
            self.n = [1.0 for _ in range(5)]
            # Default: punish Kähler violation stronger
            self.n[1] = 2.0
        else:
            self.n = [float(ni) for ni in norm]

        # projective degrees: degrees[i] = dim(P^{ambient[i]}) + 1
        ambient_real = jnp.real(self.BASIS['AMBIENT']).astype(jnp.int32)
        self.degrees = jnp.ones_like(ambient_real) + ambient_real   # shape (nProjective,)
        # Plain Python list of ints — static under @eqx.filter_jit (not JAX-traced)
        self._degrees_list = [int(d) + 1 for d in np.real(BASIS['AMBIENT']).tolist()]
        self.pi = jnp.array(np.pi, dtype=jnp.complex64)
        self.nhyper = int(jnp.real(self.BASIS['NHYPER']))

        # ---------- helpers generated at init time ----------
        self.lc = jnp.array(
            get_levicivita_tensor(self.nfold), dtype=jnp.complex64)
        self.proj_matrix = self._generate_proj_matrix()
        self.nTransitions = self._patch_transitions()
        if self.nhyper == 1:
            self.fixed_patches = self._generate_all_patches()
        else:
            self.fixed_patches = None
        self._proj_indices = self._generate_proj_indices()
        self.slopes = self._target_slopes()

    # ------------------------------------------------------------------
    # Initialisation helpers (run once, results stored as attributes)
    # ------------------------------------------------------------------

    def _generate_proj_matrix(self):
        r"""Creates proj_matrix storing ambient-space projection info
        (equivalent of FSModel._generate_proj_matrix in TF).
        """
        proj_matrix = {}
        degrees_np = np.array(self.degrees)
        ncoords = self.ncoords
        for i in range(self.nProjective):
            matrix = np.zeros((degrees_np[i], ncoords), dtype=np.complex64)
            s = int(np.sum(degrees_np[:i]))
            e = int(np.sum(degrees_np[:i + 1]))
            matrix[:, s:e] = np.eye(degrees_np[i], dtype=np.complex64)
            proj_matrix[str(i)] = jnp.array(matrix, dtype=jnp.complex64)
        return proj_matrix

    def _generate_proj_indices(self):
        r"""Makes a 1-D array with the projective-space index for each coord."""
        flat_list = []
        for i, p in enumerate(np.array(self.degrees)):
            for _ in range(p):
                flat_list.append(i)
        return jnp.array(flat_list, dtype=jnp.int32)

    def _generate_all_patches(self):
        r"""Generate all possible patches for CICYs (nhyper == 1)."""
        degrees_np = np.array(self.degrees)
        fixed_patches = []
        for i in range(self.ncoords):
            all_patches = np.array(
                list(it.product(*[
                    [j for j in range(int(np.sum(degrees_np[:k])),
                                     int(np.sum(degrees_np[:k + 1])))
                     if j != i]
                    for k in range(len(degrees_np))
                ], repeat=1)))
            if len(all_patches) == self.nTransitions:
                fixed_patches.append(all_patches)
            else:
                all_patches = np.tile(
                    all_patches,
                    (int(self.nTransitions / len(all_patches)) + 1, 1))
                fixed_patches.append(all_patches[:self.nTransitions])
        fixed_patches = np.array(fixed_patches)
        return jnp.array(fixed_patches, dtype=jnp.int32)

    def _patch_transitions(self):
        r"""Maximum number of patch transitions with the same fixed variables."""
        nTransitions = 0
        degrees_np = np.array(self.degrees)
        for t in generate_monomials(self.nProjective, self.nhyper):
            tmp_deg = [int(d) - t[j] for j, d in enumerate(degrees_np)]
            n = int(np.prod(tmp_deg))
            if n > nTransitions:
                nTransitions = n
        return int(nTransitions)

    def _target_slopes(self):
        r"""Computes the target slopes mu(F_i) for the Volk loss."""
        ks = jnp.eye(len(self.BASIS['KMODULI']), dtype=jnp.complex64)
        nfold = self.nfold
        intnums = self.BASIS['INTNUMS']
        kmoduli = self.BASIS['KMODULI']

        if nfold == 1:
            slope = jnp.einsum('a,xa->x', intnums, ks)
        elif nfold == 2:
            slope = jnp.einsum('ab,a,xb->x', intnums, kmoduli, ks)
        elif nfold == 3:
            slope = jnp.einsum('abc,a,b,xc->x', intnums, kmoduli, kmoduli, ks)
        elif nfold == 4:
            slope = jnp.einsum('abcd,a,b,c,xd->x',
                               intnums, kmoduli, kmoduli, kmoduli, ks)
        elif nfold == 5:
            slope = jnp.einsum('abcde,a,b,c,d,xe->x',
                               intnums, kmoduli, kmoduli, kmoduli, kmoduli, ks)
        else:
            raise NotImplementedError(
                'Only implemented for nfold <= 5.')
        return slope

    # ------------------------------------------------------------------
    # @tf.function methods -> decorated with @jax.jit via eqx.filter_jit
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def _calculate_slope(self, pred, f_a):
        r"""Computes the slopes mu(F_i) at a single Kahler modulus direction."""
        nfold = self.nfold
        if nfold == 1:
            slope = jnp.einsum('xab->x', f_a)
        elif nfold == 2:
            slope = jnp.einsum('xab,xcd,ac,bd->x',
                               pred, f_a, self.lc, self.lc)
        elif nfold == 3:
            slope = jnp.einsum('xab,xcd,xef,ace,bdf->x',
                               pred, pred, f_a, self.lc, self.lc)
        elif nfold == 4:
            slope = jnp.einsum('xab,xcd,xef,xgh,aceg,bdfh->x',
                               pred, pred, pred, f_a, self.lc, self.lc)
        elif nfold == 5:
            slope = jnp.einsum('xab,xcd,xef,xgh,xij,acegi,bdfhj->x',
                               pred, pred, pred, pred, f_a, self.lc, self.lc)
        else:
            raise NotImplementedError('Only implemented for nfold <= 5.')
        inv_factorial = (1. / jnp.exp(jax.scipy.special.gammaln(
            jnp.real(self.BASIS['NFOLD']).astype(jnp.float32) + 1.
        ))).astype(jnp.complex64)
        return inv_factorial * slope

    def __call__(self, input_tensor, training=True, j_elim=None):
        r"""Computes the pullbacked Fubini-Study metric at each point.

        Args:
            input_tensor (jnp.ndarray, [bSize, 2*ncoords], float32): Points.
            training (bool): Unused; kept for API compatibility.
            j_elim (jnp.ndarray, [bSize, nHyper], int64, optional):
                Indices to eliminate. If None uses max|dQ/dz|.

        Returns:
            jnp.ndarray, [bSize, nfold, nfold], complex64.
        """
        return self.fubini_study_pb(input_tensor, j_elim=j_elim)

    @eqx.filter_jit
    def compute_kaehler_loss(self, x):
        r"""Computes Kähler loss.

        Equivalent to FSModel.compute_kaehler_loss in TF.
        Uses jax.vmap + jax.jacobian for batch Jacobians instead of
        tf.GradientTape.batch_jacobian.

        Args:
            x (jnp.ndarray, [bSize, 2*ncoords], float32): Points.

        Returns:
            jnp.ndarray, [bSize], float32.
        """
        pb = self.pullbacks(x)

        # batch_jacobian of real and imaginary parts of model output w.r.t. x
        def model_re_im(x_single):
            pred = self(x_single[None], training=False)[0]   # (nfold, nfold)
            return jnp.real(pred), jnp.imag(pred)

        def jac_re(x_single):
            return jax.jacobian(lambda xs: jnp.real(
                self(xs[None], training=False)[0]))(x_single)

        def jac_im(x_single):
            return jax.jacobian(lambda xs: jnp.imag(
                self(xs[None], training=False)[0]))(x_single)

        # (bSize, nfold, nfold, 2*ncoords)
        gijk_re = jax.vmap(jac_re)(x).astype(jnp.complex64)
        gijk_im = jax.vmap(jac_im)(x).astype(jnp.complex64)

        nc = self.ncoords
        # Reconstruct complex 3-tensor c_ijk  (matches TF formula exactly)
        cijk = 0.5 * (
            gijk_re[:, :, :, :nc]
            + gijk_im[:, :, :, nc:]
            + 1.j * gijk_im[:, :, :, :nc]
            - 1.j * gijk_re[:, :, :, nc:]
        )
        cijk_pb = jnp.einsum('xija,xka->xijk', cijk, pb)
        cijk_pb = cijk_pb - jnp.transpose(cijk_pb, (0, 3, 2, 1))
        cijk_loss = jnp.sum(jnp.abs(cijk_pb) ** self.n[1], axis=(1, 2, 3))
        return cijk_loss

    @eqx.filter_jit
    def fubini_study_pb(self, points, pb=None, j_elim=None, ts=None):
        r"""Computes the pullbacked Fubini-Study metric.

        Equivalent to FSModel.fubini_study_pb in TF.

        Args:
            points (jnp.ndarray, [bSize, 2*ncoords], float32): Points.
            pb (jnp.ndarray, [bSize, nfold, ncoords], complex64, optional):
                Pre-computed pullbacks (overrides j_elim).
            j_elim (jnp.ndarray, [bSize, nHyper], int64, optional):
                Coordinates to eliminate. None → max|dQ/dz|.
            ts (jnp.ndarray, [len(kmoduli)], complex64, optional):
                Kähler parameters. None → BASIS['KMODULI'].

        Returns:
            jnp.ndarray, [bSize, nfold, nfold], complex64.
        """
        if ts is None:
            ts = self.BASIS['KMODULI']

        if self.nProjective > 1:
            s0 = 0
            e0 = self._degrees_list[0]
            cpoints = (points[:, s0:e0]
                       + 1j * points[:, self.ncoords + s0:self.ncoords + e0]).astype(jnp.complex64)
            fs = self._fubini_study_n_metrics(cpoints, n=self._degrees_list[0], t=ts[0])
            pm0 = self.proj_matrix['0']
            fs = jnp.einsum('xij,ia,bj->xab', fs, pm0, pm0.T)
            for i in range(1, self.nProjective):
                s = sum(self._degrees_list[:i])
                e = s + self._degrees_list[i]
                cpoints = (points[:, s:e]
                           + 1j * points[:, self.ncoords + s:self.ncoords + e]).astype(jnp.complex64)
                fs_tmp = self._fubini_study_n_metrics(
                    cpoints, n=self._degrees_list[i], t=ts[i])
                pmi = self.proj_matrix[str(i)]
                fs_tmp = jnp.einsum('xij,ia,bj->xab', fs_tmp, pmi, pmi.T)
                fs = fs + fs_tmp
        else:
            cpoints = (points[:, :self.ncoords]
                       + 1j * points[:, self.ncoords:2 * self.ncoords]).astype(jnp.complex64)
            fs = self._fubini_study_n_metrics(cpoints, t=ts[0])

        if pb is None:
            pb = self.pullbacks(points, j_elim=j_elim)
        fs_pb = jnp.einsum('xai,xij,xbj->xab', pb, fs, jnp.conj(pb))
        return fs_pb

    @eqx.filter_jit
    def _find_max_dQ_coords(self, points):
        r"""Finds for each point the coordinate with largest |dQ/dz|.

        Equivalent to FSModel._find_max_dQ_coords in TF.

        Args:
            points (jnp.ndarray, [bSize, 2*ncoords], float32): Points.

        Returns:
            jnp.ndarray, [bSize, nhyper], int64.
        """
        cpoints = (points[:, :self.ncoords]
                   + 1j * points[:, self.ncoords:]).astype(jnp.complex64)
        available_mask = self._get_inv_one_mask(points).astype(jnp.complex64)

        indices = None
        for i in range(self.nhyper):
            dQdz = self._compute_dQdz(cpoints, i)         # (n_p, ncoords)
            max_idx = jnp.argmax(jnp.abs(dQdz * available_mask), axis=-1)  # (n_p,)
            if i == 0:
                indices = max_idx[:, None]                 # (n_p, 1)
            else:
                indices = jnp.concatenate(
                    [indices, max_idx[:, None]], axis=-1)  # (n_p, i+1)
            available_mask = (available_mask
                              - jax.nn.one_hot(max_idx, self.ncoords,
                                               dtype=jnp.complex64))
        return indices.astype(jnp.int32)

    @eqx.filter_jit
    def pullbacks(self, points, j_elim=None):
        r"""Computes the pullback tensor at each point.

        Equivalent to FSModel.pullbacks in TF.  Uses jax.vmap over points
        for shape-determinism under jit (replaces tf.where + scatter_nd_update).

        Args:
            points (jnp.ndarray, [bSize, 2*ncoords], float32): Points.
            j_elim (jnp.ndarray, [bSize, nHyper], int64, optional):
                Coordinates to eliminate. None → max|dQ/dz|.

        Returns:
            jnp.ndarray, [bSize, nfold, ncoords], complex64.
        """
        if j_elim is None:
            dQdz_indices = self._find_max_dQ_coords(points)
        else:
            dQdz_indices = j_elim

        def _pullback_single(point, dqdz_idx):
            return self._pullback_single_point(point, dqdz_idx)

        return jax.vmap(_pullback_single)(points, dQdz_indices)

    def _pullback_single_point(self, point, dQdz_idx):
        r"""Pullback at a single point (used via vmap).

        Faithful translation of the TF pullbacks logic per-point.

        Args:
            point (jnp.ndarray, [2*ncoords], float32): Single point.
            dQdz_idx (jnp.ndarray, [nhyper], int64): Fixed coord indices.

        Returns:
            jnp.ndarray, [nfold, ncoords], complex64.
        """
        ncoords = self.ncoords
        nfold = self.nfold
        nhyper = self.nhyper

        cpoint = (point[:ncoords] + 1j * point[ncoords:]).astype(jnp.complex64)

        # Build full mask: True for "good" (free) coordinates
        inv_one_mask = ~jnp.isclose(cpoint, 1. + 0j)          # (ncoords,) bool
        full_mask = inv_one_mask.astype(jnp.float32)
        for i in range(nhyper):
            dQdz_mask = -1. * jax.nn.one_hot(dQdz_idx[i], ncoords, dtype=jnp.float32)
            full_mask = full_mask + dQdz_mask
        full_mask = full_mask.astype(bool)                     # (ncoords,), nfold True

        # good_indices: (nfold,) — indices of free coordinates
        good_indices = jnp.nonzero(full_mask, size=nfold)[0].astype(jnp.int32)

        # Initialise pullback matrix (nfold, ncoords)
        pb = jnp.zeros((nfold, ncoords), dtype=jnp.complex64)

        # Set identity block: pb[a, good_indices[a]] = 1
        pb = pb.at[jnp.arange(nfold), good_indices].set(1. + 0j)

        # Compute pia  (= dQ_i/dz_a for each free coord a) and B (= dQ_i/dz_fixed)
        # We collect nhyper rows, building dz_hyper (nhyper, nfold) and B (nhyper, nhyper)
        dz_hyper_rows = []
        B_rows = []

        for i in range(nhyper):
            # --- pia ---
            pia_polys = self.BASIS['DQDZB' + str(i)][good_indices]   # (nfold, n_terms, ncoords)
            pia_factors = self.BASIS['DQDZF' + str(i)][good_indices]  # (nfold, n_terms)
            pia = (cpoint[None, None, :] ** pia_polys)                 # (nfold, n_terms, ncoords)
            pia = jnp.prod(pia, axis=-1)                               # (nfold, n_terms)
            pia = jnp.sum(pia_factors * pia, axis=-1)                  # (nfold,)
            dz_hyper_rows.append(pia)

            # --- pif ---
            pif_polys = self.BASIS['DQDZB' + str(i)][dQdz_idx]        # (nhyper, n_terms, ncoords)
            pif_factors = self.BASIS['DQDZF' + str(i)][dQdz_idx]      # (nhyper, n_terms)
            pif = (cpoint[None, None, :] ** pif_polys)                 # (nhyper, n_terms, ncoords)
            pif = jnp.prod(pif, axis=-1)                               # (nhyper, n_terms)
            pif = jnp.sum(pif_factors * pif, axis=-1)                  # (nhyper,)
            B_rows.append(pif)

        dz_hyper = jnp.stack(dz_hyper_rows, axis=0)   # (nhyper, nfold)
        B = jnp.stack(B_rows, axis=0)                 # (nhyper, nhyper)

        # all_dzdz[a, i] = dz_fixed_i / dx_a  (shape: nfold, nhyper)
        all_dzdz = jnp.einsum('ij,jk->ki',
                              jnp.linalg.inv(B),
                              (-1. + 0j) * dz_hyper)

        # Fill fixed-coordinate columns of pullback
        for i in range(nhyper):
            pb = pb.at[jnp.arange(nfold), dQdz_idx[i]].set(all_dzdz[:, i])

        return pb

    @eqx.filter_jit
    def _get_inv_one_mask(self, points):
        r"""True when z_i != 1+0j (equivalent to TF version)."""
        cpoints = (points[:, :self.ncoords]
                   + 1j * points[:, self.ncoords:]).astype(jnp.complex64)
        return ~jnp.isclose(cpoints, jnp.ones_like(cpoints))

    @eqx.filter_jit
    def _indices_to_mask(self, indices):
        r"""Converts a (bSize, k) index array to a (bSize, ncoords) float mask.
        Equivalent to FSModel._indices_to_mask in TF.
        """
        mask = jax.nn.one_hot(indices, num_classes=self.ncoords,
                              dtype=jnp.float32)   # (bSize, k, ncoords)
        mask = jnp.sum(mask, axis=1)               # (bSize, ncoords)
        return mask

    def _generate_patches(self, fixed, original):
        r"""Generates possible patch transitions for a single point.

        Equivalent to FSModel._generate_patches in TF.

        Args:
            fixed (jnp.ndarray, [nhyper], int64): Fixed coordinate indices.
            original (jnp.ndarray, [nProjective], int64): Current patch indices.

        Returns:
            jnp.ndarray, [nTransitions, nProjective], int64.
        """
        degrees_np = np.array(self.degrees)
        ncoords = self.ncoords
        nProjective = self.nProjective
        nTransitions = self.nTransitions
        proj_indices_np = np.array(self._proj_indices)

        fixed_np = np.array(fixed)
        original_np = np.array(original)

        inv_fixed_mask = np.ones(ncoords, dtype=bool)
        for f in fixed_np:
            inv_fixed_mask[f] = False

        fixed_proj = np.zeros(nProjective, dtype=int)
        for f in fixed_np:
            fixed_proj[proj_indices_np[f]] += 1

        splits = degrees_np.astype(int) - fixed_proj
        all_coords = np.where(inv_fixed_mask)[0]
        products = []
        start = 0
        for s in splits:
            products.append(all_coords[start:start + s])
            start += s
        all_patches = np.array(list(it.product(*products, repeat=1)))
        npatches = len(all_patches)
        if npatches != nTransitions:
            same = np.tile(original_np, nTransitions - npatches)
            same = same.reshape(-1, nProjective)
            all_patches = np.concatenate([all_patches, same], axis=0)
        return jnp.array(all_patches[:nTransitions], dtype=jnp.int32)

    @eqx.filter_jit
    def _fubini_study_n_potentials(self, points, t=None):
        r"""FS Kähler potential on a single projective ambient-space factor.

        Equivalent to FSModel._fubini_study_n_potentials in TF.

        Args:
            points (jnp.ndarray, [bSize, n], complex64): Projective coordinates.
            t (complex, optional): Volume factor. Defaults to 1+0j.

        Returns:
            jnp.ndarray, [bSize], float32.
        """
        if t is None:
            t = jnp.complex64(1. + 0j)
        point_square = jnp.sum(jnp.abs(points) ** 2, axis=-1)
        return (jnp.real(t / self.pi).astype(jnp.float32)
                * jnp.real(jnp.log(point_square)).astype(jnp.float32))

    @eqx.filter_jit
    def _fubini_study_n_metrics(self, points, n=None, t=None):
        r"""FS metric on a single projective ambient-space factor.

        Equivalent to FSModel._fubini_study_n_metrics in TF.

        Args:
            points (jnp.ndarray, [bSize, n], complex64): Projective coordinates.
            n (int, optional): Degree of P^n. Defaults to self.ncoords.
            t (complex, optional): Volume factor. Defaults to 1+0j.

        Returns:
            jnp.ndarray, [bSize, n, n], complex64.
        """
        if n is None:
            n = self.ncoords
        if t is None:
            t = jnp.complex64(1. + 0j)
        point_square = jnp.sum(jnp.abs(points) ** 2, axis=-1).astype(jnp.complex64)
        point_diag = jnp.einsum('x,ij->xij', point_square,
                                jnp.eye(n, dtype=jnp.complex64))
        outer = jnp.einsum('xi,xj->xij', jnp.conj(points), points).astype(jnp.complex64)
        gFS = jnp.einsum('xij,x->xij',
                         (point_diag - outer),
                         point_square ** -2)
        return gFS * t / self.pi

    @eqx.filter_jit
    def _compute_dQdz(self, points, k):
        r"""Computes dQ_k/dz at each point.

        Equivalent to FSModel._compute_dQdz in TF.

        Args:
            points (jnp.ndarray, [bSize, ncoords], complex64): Coordinates.
            k (int): k-th hypersurface.

        Returns:
            jnp.ndarray, [bSize, ncoords], complex64.
        """
        p_exp = points[:, None, None, :]                         # (n_p, 1, 1, ncoords)
        dQdz = p_exp ** self.BASIS['DQDZB' + str(k)]            # (n_p, ncoords, n_terms, ncoords)
        dQdz = jnp.prod(dQdz, axis=-1)                          # (n_p, ncoords, n_terms)
        dQdz = self.BASIS['DQDZF' + str(k)] * dQdz              # (n_p, ncoords, n_terms)
        dQdz = jnp.sum(dQdz, axis=-1)                           # (n_p, ncoords)
        return dQdz

    @eqx.filter_jit
    def _get_patch_coordinates(self, points, patch_mask):
        r"""Transforms points into the patch given by patch_mask.

        Equivalent to FSModel._get_patch_coordinates in TF.

        For each projective space i (coordinates s:e), exactly one entry
        per row in patch_mask is True (the patch coordinate for that space).
        We extract it by summing  points * mask  over that slice, then
        broadcast the resulting scalar across all coordinates of space i.
        This avoids boolean indexing (data-dependent shapes) and is fully
        JIT-compatible.

        Args:
            points (jnp.ndarray, [bSize, ncoords], complex64): Points.
            patch_mask (jnp.ndarray, [bSize, ncoords], bool): Patch indicators.

        Returns:
            jnp.ndarray, [bSize, ncoords], complex64.
        """
        patch_mask_c = patch_mask.astype(jnp.complex64)
        norm_parts = []
        for i in range(self.nProjective):
            s = sum(self._degrees_list[:i])
            e = s + self._degrees_list[i]
            # Exactly one True entry per row in patch_mask[:, s:e]
            # norm_i[x] = the patch coordinate value for projective space i at point x
            norm_i = jnp.sum(points[:, s:e] * patch_mask_c[:, s:e], axis=-1)  # (bSize,)
            # Broadcast to all coords of this projective space
            norm_parts.append(jnp.tile(norm_i[:, None], (1, self._degrees_list[i])))  # (bSize, d_i)
        full_norm = jnp.concatenate(norm_parts, axis=-1)  # (bSize, ncoords)
        return points / full_norm

    @eqx.filter_jit
    def compute_transition_loss(self, points):
        r"""Computes transition loss at each point.

        Equivalent to FSModel.compute_transition_loss in TF.

        Uses jax.vmap + jnp.nonzero(size=nProjective) to extract patch
        indices with a statically-known output shape, enabling full JIT
        compilation (unlike jnp.where(condition) whose output size depends
        on the data).

        Args:
            points (jnp.ndarray, [bSize, 2*ncoords], float32): Points.

        Returns:
            jnp.ndarray, [bSize], float32.
        """
        inv_one_mask = self._get_inv_one_mask(points)   # (bSize, ncoords) bool
        # ~inv_one_mask has exactly nProjective True entries per row (patch coords).
        # Use static-size nonzero to get column indices:
        nP = self.nProjective
        patch_indices = jax.vmap(
            lambda row: jnp.nonzero(~row, size=nP, fill_value=0)[0].astype(jnp.int32)
        )(inv_one_mask)   # (bSize, nProjective)

        current_patch_mask = self._indices_to_mask(patch_indices)
        cpoints = (points[:, :self.ncoords]
                   + 1j * points[:, self.ncoords:]).astype(jnp.complex64)
        fixed = self._find_max_dQ_coords(points)

        if self.nhyper == 1:
            other_patches = self.fixed_patches[fixed[:, 0]]       # (n_p, nTransitions, nProjective)
        else:
            # generate patches per-point (only used for nhyper > 1, runs eagerly)
            n_p = points.shape[0]
            other_patches_list = []
            for xi in range(n_p):
                combined = jnp.concatenate(
                    [fixed[xi], patch_indices[xi]], axis=0)
                op = self._generate_patches(
                    combined[:self.nhyper], combined[self.nhyper:])
                other_patches_list.append(op)
            other_patches = jnp.stack(other_patches_list, axis=0)

        other_patches = other_patches.reshape(-1, self.nProjective)
        other_patch_mask = self._indices_to_mask(other_patches)

        # expand points for all transitions
        exp_points = jnp.repeat(cpoints, self.nTransitions, axis=0)
        patch_points = self._get_patch_coordinates(
            exp_points, other_patch_mask.astype(bool))

        fixed_exp = jnp.tile(fixed, (1, self.nTransitions)).reshape(-1, self.nhyper)
        real_patch_points = jnp.concatenate(
            [jnp.real(patch_points), jnp.imag(patch_points)], axis=-1)
        gj = self(real_patch_points, training=True, j_elim=fixed_exp)
        gi = jnp.repeat(self(points), self.nTransitions, axis=0)
        current_patch_mask_exp = jnp.repeat(
            current_patch_mask, self.nTransitions, axis=0)
        Tij = self.get_transition_matrix(
            patch_points, other_patch_mask, current_patch_mask_exp, fixed_exp)
        all_t_loss = jnp.abs(self.transition_loss_matrices(gj, gi, Tij))
        all_t_loss = jnp.sum(all_t_loss ** self.n[2], axis=(1, 2))
        all_t_loss = all_t_loss.reshape(-1, self.nTransitions)
        all_t_loss = jnp.sum(all_t_loss, axis=-1)
        return all_t_loss / (self.nTransitions * self.nfold ** 2)

    @eqx.filter_jit
    @eqx.filter_jit
    def get_transition_matrix(self, points, i_mask, j_mask, fixed):
        r"""Computes transition matrix between patches i and j.

        Faithful translation of FSModel.get_transition_matrix from TF.

        Rewritten for full JAX JIT compatibility: all shapes are statically
        known at compile time. The same/diff-patch split is replaced by a
        uniform computation over all points, masked with ``jnp.where`` at
        the end (identity for same-patch transitions, computed matrix for
        different-patch transitions).

        Args:
            points (jnp.ndarray, [bSize, ncoords], complex64): Points.
            i_mask (jnp.ndarray, [bSize, ncoords], float32): Patch i mask.
            j_mask (jnp.ndarray, [bSize, ncoords], float32): Patch j mask.
            fixed (jnp.ndarray, [bSize, nhyper], int64): Elimination indices.

        Returns:
            jnp.ndarray, [bSize, nfold, nfold], complex64.
        """
        n_p = fixed.shape[0]
        nfold = self.nfold
        ncoords = self.ncoords
        nP = self.nProjective

        # Same-patch indicator: (n_p,) bool — known shape ✓
        same_patch = jnp.all(i_mask == j_mask, axis=-1)

        # g1_mask / g2_mask: True for free coordinates (not fixed, not patch coord)
        fixed_oh = jnp.sum(
            jax.nn.one_hot(fixed, ncoords, dtype=jnp.float32), axis=-2)  # (n_p, ncoords)
        g1_mask = ~(fixed_oh + i_mask).astype(bool)   # (n_p, ncoords), nfold True/row
        g2_mask = ~(fixed_oh + j_mask).astype(bool)   # (n_p, ncoords), nfold True/row

        # Free-coordinate indices with static output shape (n_p, nfold)
        g1_i = jax.vmap(
            lambda row: jnp.nonzero(row, size=nfold, fill_value=0)[0].astype(jnp.int32)
        )(g1_mask)
        g2_i = jax.vmap(
            lambda row: jnp.nonzero(row, size=nfold, fill_value=0)[0].astype(jnp.int32)
        )(g2_mask)

        # Projective-space index for each g1 free coordinate: (n_p, nfold)
        proj_bc = jnp.broadcast_to(self._proj_indices[None], (n_p, ncoords))
        g1_proj = jnp.take_along_axis(proj_bc, g1_i, axis=-1)   # (n_p, nfold)

        # Patch-coordinate indices from i_mask and j_mask: (n_p, nProjective)
        p1 = jax.vmap(
            lambda row: jnp.nonzero(row.astype(bool), size=nP, fill_value=0)[0].astype(jnp.int32)
        )(i_mask)
        p2 = jax.vmap(
            lambda row: jnp.nonzero(row.astype(bool), size=nP, fill_value=0)[0].astype(jnp.int32)
        )(j_mask)

        # Patch coordinate values and per-projective-space ratios: (n_p, nProjective)
        i_pts = jnp.take_along_axis(points, p1, axis=-1)
        j_pts = jnp.take_along_axis(points, p2, axis=-1)
        ratios = (i_pts / j_pts).astype(jnp.complex64)

        tij = jnp.zeros((n_p, nfold, nfold), dtype=jnp.complex64)

        # ── Mixed-ratio elements (loop over projective spaces) ──────────────
        # For each j: tij[x, i, k] = -points[x, g2_i[x,k]] * ratios[x,j] / points[x, p2[x,j]]
        #   where t_pos[x, i, k] == 1  (i.e. g1_i[x,i]==p2[x,j] and g1_proj[x,i]==j)
        for j in range(nP):
            t_pos_row = (g1_i == p2[:, j:j + 1]).astype(jnp.int32)  # (n_p, nfold)
            t_pos_col = (g1_proj == j).astype(jnp.int32)              # (n_p, nfold)
            t_pos = jnp.einsum('xi,xj->xij', t_pos_row, t_pos_col)   # (n_p, nfold, nfold)

            num_t   = jnp.take_along_axis(points, g2_i, axis=-1)      # (n_p, nfold)
            ratio_t = ratios[:, j]                                     # (n_p,)
            denom_t = jnp.take_along_axis(
                points, p2[:, j:j + 1], axis=-1)[:, 0]                # (n_p,)
            t_vals  = (
                -1. * num_t * ratio_t[:, None] / denom_t[:, None]
            ).astype(jnp.complex64)                                    # (n_p, nfold)

            # Accumulate: tij[x,i,k] += t_vals[x,k] where t_pos[x,i,k]==1
            tij = tij + (t_vals[:, None, :] * t_pos).astype(jnp.complex64)

        # ── Single-ratio (diagonal-like) elements ───────────────────────────
        # For each (x, a, b) where g1_i[x,a] == g2_i[x,b]:
        #   tij[x, a, b] = ratios[x, g1_proj[x,a]]
        c_cond   = (g1_i[:, :, None] == g2_i[:, None, :])              # (n_p, nfold, nfold)
        c_ratios = jnp.take_along_axis(ratios, g1_proj, axis=-1)       # (n_p, nfold)
        tij = jnp.where(c_cond, c_ratios[:, :, None].astype(jnp.complex64), tij)

        # ── Same-patch transitions → identity matrix ────────────────────────
        eye_bc = jnp.broadcast_to(
            jnp.eye(nfold, dtype=jnp.complex64)[None], (n_p, nfold, nfold))
        tij = jnp.where(same_patch[:, None, None], eye_bc, tij)

        return tij


    @eqx.filter_jit
    def transition_loss_matrices(self, gj, gi, Tij):
        r"""g_j - T_{ij} g_i T_{ij}^†.

        Equivalent to FSModel.transition_loss_matrices in TF.
        """
        return gj - jnp.einsum(
            'xij,xjk,xlk->xil', Tij, gi, jnp.conj(Tij))

    @eqx.filter_jit
    def compute_ricci_scalar(self, points, pb=None):
        r"""Computes the Ricci scalar R at each point.

        Equivalent to FSModel.compute_ricci_scalar in TF. Uses nested
        jax.grad/jax.jacobian instead of tf.GradientTape.

        Args:
            points (jnp.ndarray, [bSize, 2*ncoords], float32): Points.
            pb (jnp.ndarray, [bSize, nfold, ncoords], complex64, optional):
                Pre-computed pullbacks.

        Returns:
            jnp.ndarray, [bSize], float32.
        """
        ncoords = self.ncoords

        def log_det_single(x_single):
            pred = self(x_single[None], training=False)[0]        # (nfold, nfold)
            det = jnp.real(jnp.linalg.det(pred))
            return jnp.log(det)

        def di_dg_single(x_single):
            return jax.grad(log_det_single)(x_single)              # (2*ncoords,)

        # d_i d_j log det  — (bSize, 2*ncoords, 2*ncoords)
        didj_dg = jax.vmap(jax.jacobian(di_dg_single))(points).astype(jnp.complex64)

        # Reconstruct complex Ricci tensor (exactly as in TF)
        ricci_ij = (didj_dg[:, :ncoords, :ncoords]
                    + 1j * didj_dg[:, :ncoords, ncoords:]
                    - 1j * didj_dg[:, ncoords:, :ncoords]
                    + didj_dg[:, ncoords:, ncoords:])
        ricci_ij = ricci_ij * 0.25

        prediction = self(points, training=False)
        pred_inv = jnp.linalg.inv(prediction)

        if pb is None:
            pullbacks = self.pullbacks(points)
        else:
            pullbacks = pb

        ricci_scalar = jnp.einsum(
            'xba,xai,xij,xbj->x',
            pred_inv, pullbacks, ricci_ij, jnp.conj(pullbacks))
        return jnp.real(ricci_scalar)

    @eqx.filter_jit
    def compute_ricci_loss(self, points, pb=None):
        r"""|1 - exp(-R)| loss for Ricci scalar.

        Equivalent to FSModel.compute_ricci_loss in TF.
        """
        ricci_scalar = self.compute_ricci_scalar(points, pb)
        return jnp.abs(1. - jnp.exp(-ricci_scalar))
