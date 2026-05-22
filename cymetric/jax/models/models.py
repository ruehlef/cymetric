"""
JAX / Equinox models for learning Calabi-Yau metrics.

Faithful translation of tensorflow/models/models.py.

Key design:
* Each model is an ``equinox.Module`` (analogous to tf.keras.Model).
* ``@eqx.filter_jit`` replaces ``@tf.function`` for XLA compilation.
* ``jax.grad`` / ``jax.jacobian`` + ``jax.vmap`` replace tf.GradientTape.
* ``optax`` replaces tf.keras optimizers.
* Training flags (learn_kaehler, learn_volk, …) are plain Python bools
  (static under jit; changing them requires a new JIT trace, same as
  changing tf.bool Variables before tracing in TF).
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from functools import partial

from .losses import sigma_loss
from .fubinistudy import FSModel
from cymetric.pointgen.nphelper import (
    get_all_patch_degrees, compute_all_w_of_x, get_levicivita_tensor,
)


# ---------------------------------------------------------------------------
# Helper: Hermitian matrix from flat real vector
# (used in multiple models; matches to_hermitian in TF FreeModel)
# ---------------------------------------------------------------------------

def _to_hermitian(x, nfold):
    r"""Return a Hermitian (nfold×nfold) matrix from a flat real vector.

    Faithful translation of FreeModel.to_hermitian in TF.

    Args:
        x (jnp.ndarray, [bSize, nfold**2], float32): Flat prediction.
        nfold (int): CY dimension.

    Returns:
        jnp.ndarray, [bSize, nfold, nfold], complex64.
    """
    t1 = jnp.reshape(x + 0j, (-1, nfold, nfold)).astype(jnp.complex64)
    up = jnp.triu(t1)           # tf.linalg.band_part(t1, 0, -1)
    low = jnp.tril(1j * t1)    # tf.linalg.band_part(1j*t1, -1, 0)
    # diagonal matrix
    diag_vals = jnp.diagonal(t1, axis1=-2, axis2=-1)       # (bSize, nfold)
    diag_mat = jnp.einsum('...i,ij->...ij', diag_vals,
                          jnp.eye(nfold, dtype=jnp.complex64))
    out = up + jnp.swapaxes(up, -2, -1) - diag_mat
    return out + low + jnp.conj(jnp.swapaxes(low, -2, -1))


# ---------------------------------------------------------------------------
# FreeModel
# ---------------------------------------------------------------------------

class FreeModel(FSModel):
    r"""FreeModel: learns g_out = g_NN (a Hermitian tensor).

    Equivalent to tensorflow/models/models.py::FreeModel.

    The NN backbone (``self.model``) is any ``equinox.Module`` that maps
    ``(bSize, 2*ncoords)`` → ``(bSize, nfold**2)`` real-valued outputs.
    Training is done externally via :func:`cymetric.jax.models.helper.train_model`.

    NOTE:
        * ``learn_ricci`` is False by default, matching TF.
        * Loss contributions are weighted by ``self.alpha``.
        * ``self.sigma_loss`` is the MA loss function (closure).
    """
    model: eqx.Module
    NLOSS: int
    alpha: list          # [float] × NLOSS
    learn_kaehler: bool
    learn_transition: bool
    learn_ricci: bool
    learn_ricci_val: bool
    learn_volk: bool
    kappa: float
    gclipping: float
    # sigma_loss closure is not a JAX array — store as a non-leaf attribute
    # by wrapping in a list (equinox treats non-array containers as static)
    _sigma_loss_fn: list   # [callable]
    custom_metrics: list   # [metric objects]

    def __init__(self, nn_model, BASIS, alpha=None, **kwargs):
        r"""
        Args:
            nn_model (eqx.Module): NN mapping (bSize, 2*ncoords) → (bSize, nfold²).
            BASIS (dict): Geometry dictionary from PointGenerator.
            alpha (list of float, optional): Loss weights. Defaults to all 1.
        """
        super(FreeModel, self).__init__(BASIS=BASIS, **kwargs)
        self.model = nn_model
        self.NLOSS = 5

        if alpha is not None:
            self.alpha = [float(a) for a in alpha]
        else:
            self.alpha = [1.0] * self.NLOSS

        self.learn_kaehler = True
        self.learn_transition = True
        self.learn_ricci = False
        self.learn_ricci_val = False
        self.learn_volk = True

        self.kappa = float(jnp.real(BASIS['KAPPA']))
        self.gclipping = 5.0
        self._sigma_loss_fn = [sigma_loss(self.kappa, float(self.nfold))]
        self.custom_metrics = []

    # ------------------------------------------------------------------
    # sigma_loss property (stored as list to keep equinox happy)
    # ------------------------------------------------------------------

    @property
    def sigma_loss(self):
        return self._sigma_loss_fn[0]

    # ------------------------------------------------------------------
    # Forward pass (equivalent to FreeModel.call in TF)
    # ------------------------------------------------------------------

    def __call__(self, input_tensor, training=True, j_elim=None):
        r"""g_out = g_NN (Hermitian).

        Equivalent to FreeModel.call in TF.

        Args:
            input_tensor (jnp.ndarray, [bSize, 2*ncoords], float32): Points.
            training (bool): Passed to NN backbone (for dropout/BN).
            j_elim: Not used in FreeModel; kept for API compatibility.

        Returns:
            jnp.ndarray, [bSize, nfold, nfold], complex64.
        """
        return self.to_hermitian(jax.vmap(self.model)(input_tensor))

    @eqx.filter_jit
    def to_hermitian(self, x):
        r"""Hermitian matrix from flat vector.

        Equivalent to FreeModel.to_hermitian in TF.
        """
        return _to_hermitian(x, self.nfold)

    # ------------------------------------------------------------------
    # Volk loss  (equivalent to FreeModel.compute_volk_loss in TF)
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def compute_volk_loss(self, input_tensor, wo, pred=None):
        r"""Volk loss (integral over batch).

        Equivalent to FreeModel.compute_volk_loss in TF.

        Args:
            input_tensor (jnp.ndarray, [bSize, 2*ncoords], float32): Points.
            wo (jnp.ndarray, [bSize, ≥2], float32): y_train; wo[:,0]=weights,
                wo[:,1]=|Omega|^2.
            pred (jnp.ndarray, [bSize, nfold, nfold], complex64, optional):
                Pre-computed metric prediction.

        Returns:
            jnp.ndarray, [bSize], float32: repeated scalar loss.
        """
        if pred is None:
            pred = self(input_tensor)

        aux_weights = (wo[:, 0] / wo[:, 1]).astype(jnp.complex64)
        nk = len(self.BASIS['KMODULI'])
        aux_weights = jnp.repeat(aux_weights[None, :], nk, axis=0)
        # (nk, bSize)

        ks = jnp.eye(nk, dtype=jnp.complex64)

        # Build actual_slopes via a Python loop (matches TF while_loop logic)
        actual_slopes = None
        for ki in range(nk):
            f_a = self.fubini_study_pb(input_tensor, ts=ks[ki])
            s = self._calculate_slope(pred, f_a)[None, :]     # (1, bSize)
            if actual_slopes is None:
                actual_slopes = s
            else:
                actual_slopes = jnp.concatenate([actual_slopes, s], axis=0)
        # actual_slopes: (nk, bSize)

        actual_slopes = jnp.mean(aux_weights * actual_slopes, axis=-1)
        # actual_slopes: (nk,)

        loss = jnp.mean(jnp.abs(actual_slopes - self.slopes) ** self.n[4])
        return jnp.repeat(loss[None], len(wo), axis=0)

    # ------------------------------------------------------------------
    # Combined loss (used internally by train_step / compute_loss)
    # ------------------------------------------------------------------

    def compute_loss(self, x, y, sample_weight=None):
        r"""Computes the full combined loss for a batch.

        Equivalent to the body of FreeModel.train_step in TF.

        Args:
            x (jnp.ndarray, [bSize, 2*ncoords], float32): Points.
            y (jnp.ndarray, [bSize, ≥2], float32): Labels.
            sample_weight (jnp.ndarray, [bSize], float32, optional).

        Returns:
            jnp.ndarray, [bSize], float32: per-sample loss.
        """
        y_pred = self(x)

        cijk_loss = (self.compute_kaehler_loss(x)
                     if self.learn_kaehler
                     else jnp.zeros(x.shape[0], dtype=jnp.float32))

        t_loss = (self.compute_transition_loss(x)
                  if self.learn_transition
                  else jnp.zeros_like(cijk_loss))

        r_loss = (self.compute_ricci_loss(x)
                  if self.learn_ricci
                  else jnp.zeros_like(cijk_loss))

        volk_loss = (self.compute_volk_loss(x, y, y_pred)
                     if self.learn_volk
                     else jnp.zeros_like(cijk_loss))

        omega = y[:, -1:]
        sigma_loss_cont = self.sigma_loss(omega, y_pred) ** self.n[0]

        total_loss = (self.alpha[0] * sigma_loss_cont
                      + self.alpha[1] * cijk_loss
                      + self.alpha[2] * t_loss
                      + self.alpha[3] * r_loss
                      + self.alpha[4] * volk_loss)

        if sample_weight is not None:
            total_loss = total_loss * sample_weight

        return total_loss, {
            'sigma_loss': sigma_loss_cont,
            'kaehler_loss': cijk_loss,
            'transition_loss': t_loss,
            'ricci_loss': r_loss,
            'volk_loss': volk_loss,
        }

    def save(self, filepath, **kwargs):
        r"""Saves the NN backbone to ``filepath`` using equinox serialisation.

        Equivalent to FreeModel.save in TF (saves only the NN).

        Args:
            filepath (str): destination path.
        """
        eqx.tree_serialise_leaves(filepath, self.model)


# ---------------------------------------------------------------------------
# MultFSModel
# ---------------------------------------------------------------------------

class MultFSModel(FreeModel):
    r"""g_out = g_FS * (1 + g_NN)  (element-wise).

    Equivalent to tensorflow/models/models.py::MultFSModel.
    """

    def __init__(self, *args, **kwargs):
        super(MultFSModel, self).__init__(*args, **kwargs)

    def __call__(self, input_tensor, training=True, j_elim=None):
        r"""g_out_ij = g_FS_ij * (1 + g_NN_ij).

        Equivalent to MultFSModel.call in TF.
        """
        nn_cont = self.to_hermitian(jax.vmap(self.model)(input_tensor))
        fs_cont = self.fubini_study_pb(input_tensor, j_elim=j_elim)
        return fs_cont + jnp.multiply(fs_cont, nn_cont)


# ---------------------------------------------------------------------------
# MatrixFSModel
# ---------------------------------------------------------------------------

class MatrixFSModel(FreeModel):
    r"""g_out = g_FS + g_FS @ g_NN  (matrix multiplication).

    Equivalent to tensorflow/models/models.py::MatrixFSModel.
    """

    def __init__(self, *args, **kwargs):
        super(MatrixFSModel, self).__init__(*args, **kwargs)

    def __call__(self, input_tensor, training=True, j_elim=None):
        r"""g_out_ik = g_FS_ij (delta_jk + g_NN_jk).

        Equivalent to MatrixFSModel.call in TF.
        """
        nn_cont = self.to_hermitian(jax.vmap(self.model)(input_tensor))
        fs_cont = self.fubini_study_pb(input_tensor, j_elim=j_elim)
        return fs_cont + jnp.matmul(fs_cont, nn_cont)


# ---------------------------------------------------------------------------
# AddFSModel
# ---------------------------------------------------------------------------

class AddFSModel(FreeModel):
    r"""g_out = g_FS + g_NN  (element-wise addition).

    Equivalent to tensorflow/models/models.py::AddFSModel.
    """

    def __init__(self, *args, **kwargs):
        super(AddFSModel, self).__init__(*args, **kwargs)

    def __call__(self, input_tensor, training=True, j_elim=None):
        r"""g_out_ij = g_FS_ij + g_NN_ij.

        Equivalent to AddFSModel.call in TF.
        """
        nn_cont = self.to_hermitian(jax.vmap(self.model)(input_tensor))
        fs_cont = self.fubini_study_pb(input_tensor, j_elim=j_elim)
        return fs_cont + nn_cont


# ---------------------------------------------------------------------------
# PhiFSModel
# ---------------------------------------------------------------------------

class PhiFSModel(FreeModel):
    r"""g_out = g_FS + del dbar phi_NN.

    Equivalent to tensorflow/models/models.py::PhiFSModel.

    The NN outputs a scalar phi; the metric correction is computed via
    second-order JAX autodiff (replacing tf.GradientTape nested calls).
    By construction Kähler, so learn_kaehler is disabled.
    """

    def __init__(self, *args, **kwargs):
        super(PhiFSModel, self).__init__(*args, **kwargs)
        # Phi model is automatically Kähler
        self.learn_kaehler = False

    def __call__(self, input_tensor, training=True, j_elim=None):
        r"""g_out_ij = g_FS_ij + partial_i bar_partial_j phi_NN.

        Equivalent to PhiFSModel.call in TF.
        Uses nested jax.jacobian instead of tf.GradientTape.
        """
        ncoords = self.ncoords

        def phi_fn(x_single):
            """Scalar NN output for a single point.

            x_single has shape (n_in,); equinox Linear layers expect 1-D input,
            so we call self.model directly without adding a batch dimension.
            """
            return self.model(x_single)[0]   # scalar: output shape (1,) -> ()

        def d_phi_fn(x_single):
            return jax.grad(phi_fn)(x_single)         # (2*ncoords,)

        # d2_phi: (bSize, 2*ncoords, 2*ncoords)
        dd_phi = jax.vmap(jax.jacobian(d_phi_fn))(input_tensor)

        nc = ncoords
        dx_dx_phi = 0.25 * dd_phi[:, :nc, :nc]
        dx_dy_phi = 0.25 * dd_phi[:, :nc, nc:]
        dy_dx_phi = 0.25 * dd_phi[:, nc:, :nc]
        dy_dy_phi = 0.25 * dd_phi[:, nc:, nc:]
        dd_phi_c = (dx_dx_phi + dy_dy_phi
                    + 1j * (dx_dy_phi - dy_dx_phi)).astype(jnp.complex64)

        pbs = self.pullbacks(input_tensor, j_elim=j_elim)
        dd_phi_pb = jnp.einsum('xai,xij,xbj->xab', pbs, dd_phi_c, jnp.conj(pbs))

        fs_cont = self.fubini_study_pb(input_tensor, pb=pbs, j_elim=j_elim)
        return fs_cont + dd_phi_pb

    def compute_transition_loss(self, points):
        r"""Transition loss for Phi model: phi(lambda^q z) == phi(z).

        Equivalent to PhiFSModel.compute_transition_loss in TF.
        """
        inv_one_mask = self._get_inv_one_mask(points)
        # Each row of ~inv_one_mask has exactly nProjective True entries (patch
        # coords).  jnp.where(condition) used as nonzero requires a static size
        # inside jit; use jax.vmap + jnp.nonzero(size=nP) per row instead.
        nP = self.nProjective
        patch_indices = jax.vmap(
            lambda row: jnp.nonzero(~row, size=nP, fill_value=0)[0].astype(jnp.int32)
        )(inv_one_mask)   # (bSize, nProjective)
        current_patch_mask = self._indices_to_mask(patch_indices)
        fixed = self._find_max_dQ_coords(points)
        cpoints = (points[:, :self.ncoords]
                   + 1j * points[:, self.ncoords:]).astype(jnp.complex64)

        if self.nhyper == 1:
            other_patches = self.fixed_patches[fixed[:, 0]]
        else:
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
        exp_points = jnp.repeat(cpoints, self.nTransitions, axis=0)
        patch_points = self._get_patch_coordinates(
            exp_points, other_patch_mask.astype(bool))
        real_patch_points = jnp.concatenate(
            [jnp.real(patch_points), jnp.imag(patch_points)], axis=-1)
        gj = jax.vmap(self.model)(real_patch_points)
        gi = jnp.repeat(jax.vmap(self.model)(points), self.nTransitions, axis=0)
        all_t_loss = jnp.abs(gi - gj)
        all_t_loss = all_t_loss.reshape(-1, self.nTransitions)
        all_t_loss = jnp.sum(all_t_loss ** self.n[2], axis=-1)
        return all_t_loss / (self.nTransitions * self.nfold ** 2)

    def get_kahler_potential(self, points):
        r"""K_FS + phi_NN.

        Equivalent to PhiFSModel.get_kahler_potential in TF.
        """
        if self.nProjective > 1:
            s0 = 0
            e0 = int(self.degrees[0])
            cpoints = (points[:, s0:e0]
                       + 1j * points[:, self.ncoords + s0: self.ncoords + e0]).astype(jnp.complex64)
            k_fs = self._fubini_study_n_potentials(cpoints, t=self.BASIS['KMODULI'][0])
            for i in range(1, self.nProjective):
                s = int(jnp.sum(self.degrees[:i]))
                e = s + int(self.degrees[i])
                cpoints = (points[:, s:e]
                           + 1j * points[:, self.ncoords + s: self.ncoords + e]).astype(jnp.complex64)
                k_fs = k_fs + self._fubini_study_n_potentials(
                    cpoints, t=self.BASIS['KMODULI'][i])
        else:
            cpoints = (points[:, :self.ncoords]
                       + 1j * points[:, self.ncoords:2 * self.ncoords]).astype(jnp.complex64)
            k_fs = self._fubini_study_n_potentials(
                cpoints, t=self.BASIS['KMODULI'][0])

        k_fs = k_fs + jax.vmap(self.model)(points).reshape(-1)
        return k_fs


# ---------------------------------------------------------------------------
# ToricModel
# ---------------------------------------------------------------------------

class ToricModel(FreeModel):
    r"""Toric CY model (generalisation of FS metric for toric spaces).

    Equivalent to tensorflow/models/models.py::ToricModel.
    """
    # Extra toric attributes (equinox: non-array attrs are static by default)
    kmoduli: jnp.ndarray
    sections: list
    patch_masks_arr: jnp.ndarray   # bool array (nPatches, ncoords)
    glsm_charges: jnp.ndarray
    nPatches: int
    patch_degrees: jnp.ndarray
    transition_coefficients: jnp.ndarray
    transition_degrees: jnp.ndarray

    def __init__(self, *args, toric_data, **kwargs):
        r"""
        Args:
            *args: Positional args forwarded to FreeModel (nn_model, BASIS).
            toric_data (dict): Toric geometry data from sage_lib.
            **kwargs: Keyword args forwarded to FreeModel.
        """
        self.nfold = toric_data['dim_cy']
        self.sections = [
            jnp.array(m, dtype=jnp.complex64)
            for m in toric_data['exps_sections']
        ]
        self.patch_masks_arr = jnp.array(
            np.array(toric_data['patch_masks'], dtype=bool))
        self.glsm_charges = jnp.array(
            np.array(toric_data['glsm_charges']))
        self.nPatches = len(self.patch_masks_arr)
        self.nProjective = len(toric_data['glsm_charges'])
        super(ToricModel, self).__init__(*args, **kwargs)
        self.kmoduli = self.BASIS['KMODULI']
        self.lc = jnp.array(
            get_levicivita_tensor(self.nfold), dtype=jnp.complex64)
        self.slopes = self._target_slopes()

    def __call__(self, input_tensor, training=True, j_elim=None):
        r"""Returns the toric FS metric J = t^alpha J_alpha.

        Equivalent to ToricModel.call in TF.
        """
        return self.fubini_study_pb(input_tensor, j_elim=j_elim)

    def fubini_study_pb(self, points, pb=None, j_elim=None, ts=None):
        r"""Pullbacked toric FS metric.

        Equivalent to ToricModel.fubini_study_pb in TF.
        """
        if ts is None:
            ts = self.BASIS['KMODULI']
        pullbacks = self.pullbacks(points, j_elim=j_elim) if pb is None else pb
        cpoints = (points[:, :self.ncoords]
                   + 1j * points[:, self.ncoords:]).astype(jnp.complex64)
        Js = self._fubini_study_n_metrics(cpoints, n=0, t=ts[0])
        for i in range(1, len(self.kmoduli)):
            Js = Js + self._fubini_study_n_metrics(cpoints, n=i, t=ts[i])
        gFSpb = jnp.einsum('xai,xij,xbj->xab', pullbacks, Js, jnp.conj(pullbacks))
        return gFSpb

    @eqx.filter_jit
    def _fubini_study_n_metrics(self, points, n=None, t=None):
        r"""Toric FS metric contribution g_alpha = d_i dbar_j ln rho_alpha.

        Equivalent to ToricModel._fubini_study_n_metrics in TF.
        """
        if t is None:
            t = jnp.complex64(1. + 0j)
        alpha = 0 if n is None else n
        degrees = self.sections[alpha]
        ms = jnp.prod(
            points[:, None, :] ** degrees[None, :, :], axis=-1)      # (bSize, n_sections)
        mss = ms * jnp.conj(ms)
        kappa_alphas = jnp.sum(mss, axis=-1)                          # (bSize,)
        zizj = points[:, :, None] * jnp.conj(points[:, None, :])     # (bSize, nc, nc)
        J_alphas = 1. / zizj
        J_alphas = jnp.einsum('x,xab->xab',
                              1. / kappa_alphas ** 2, J_alphas)
        coeffs = (jnp.einsum('xa,xb,ai,aj->xij', mss, mss, degrees, degrees)
                  - jnp.einsum('xa,xb,ai,bj->xij', mss, mss, degrees, degrees))
        return J_alphas * coeffs * t / jnp.array(np.pi, dtype=jnp.complex64)

    def _generate_helpers(self):
        """Toric-specific helpers (patch degrees, transition data)."""
        self.nTransitions = int(np.max(
            np.sum(~np.array(self.patch_masks_arr), axis=-2)))
        self.fixed_patches = self._generate_all_patches()
        pm_np = np.array(self.patch_masks_arr)
        gc_np = np.array(self.glsm_charges)
        patch_degrees = get_all_patch_degrees(gc_np, pm_np)
        w_of_x, del_w_of_x, del_w_of_z = compute_all_w_of_x(patch_degrees, pm_np)
        self.patch_degrees = jnp.array(patch_degrees, dtype=jnp.complex64)
        self.transition_coefficients = jnp.array(w_of_x, dtype=jnp.complex64)
        self.transition_degrees = jnp.array(del_w_of_z, dtype=jnp.complex64)
        self.proj_matrix = None
        self._proj_indices = None

    def _generate_all_patches(self):
        """Toric fixed_patches (ncoords, nPatches, nTransitions)."""
        pm_np = np.array(self.patch_masks_arr)
        fixed_patches = np.repeat(
            np.arange(self.nPatches), self.nTransitions)
        fixed_patches = np.tile(fixed_patches, self.ncoords)
        fixed_patches = fixed_patches.reshape(
            self.ncoords, self.nPatches, self.nTransitions)
        for i in range(self.ncoords):
            all_patches = ~pm_np[:, i]
            all_indices = np.where(all_patches)[0]
            fixed_patches[i, all_indices, :len(all_indices)] = (
                all_indices * np.ones((len(all_indices), len(all_indices)),
                                      dtype=int))
        return jnp.array(fixed_patches, dtype=jnp.int32)

    @eqx.filter_jit
    def _get_patch_coordinates(self, points, patch_index):
        r"""Transforms to patch specified by patch_index.

        Equivalent to ToricModel._get_patch_coordinates in TF.
        """
        degrees = self.patch_degrees[patch_index[:, 0]]
        scaled = points[:, None, :] ** degrees
        return jnp.prod(scaled, axis=-1)

    @eqx.filter_jit
    def _mask_to_patch_index(self, mask):
        """Returns patch index in self.patch_masks_arr for each mask row."""
        match = jnp.all(mask[:, None, :] == self.patch_masks_arr[None], axis=-1)
        indices = jnp.where(match)
        return indices[1:]

    @eqx.filter_jit
    def compute_transition_loss(self, points):
        r"""Toric transition loss.

        Equivalent to ToricModel.compute_transition_loss in TF.
        """
        inv_one_mask = self._get_inv_one_mask(points)
        current_patch_mask = ~inv_one_mask
        current_patch_index = self._mask_to_patch_index(
            current_patch_mask).reshape(-1, 1)
        cpoints = (points[:, :self.ncoords]
                   + 1j * points[:, self.ncoords:]).astype(jnp.complex64)
        fixed = self._find_max_dQ_coords(points)
        other_patches = self.fixed_patches[
            jnp.concatenate([fixed, current_patch_index], axis=-1)[:, 0],
            jnp.concatenate([fixed, current_patch_index], axis=-1)[:, 1]]
        other_patch_mask = self.patch_masks_arr[other_patches]
        other_patch_mask = other_patch_mask.reshape(-1, self.ncoords)
        exp_points = jnp.repeat(cpoints, self.nTransitions, axis=0)
        patch_points = self._get_patch_coordinates(
            exp_points, other_patches.reshape(-1, 1))
        fixed_exp = jnp.reshape(
            jnp.tile(fixed, (1, self.nTransitions)), (-1, self.nhyper))
        real_points = jnp.concatenate(
            [jnp.real(patch_points), jnp.imag(patch_points)], axis=-1)
        gj = self(real_points, training=True, j_elim=fixed_exp)
        gi = jnp.repeat(self(points), self.nTransitions, axis=0)
        current_patch_mask_exp = jnp.repeat(
            current_patch_mask, self.nTransitions, axis=0)
        Tij = self.get_transition_matrix(
            patch_points, other_patch_mask, current_patch_mask_exp, fixed_exp)
        all_t_loss = jnp.abs(self.transition_loss_matrices(gj, gi, Tij))
        all_t_loss = jnp.sum(all_t_loss, axis=(1, 2))
        all_t_loss = all_t_loss.reshape(-1, self.nTransitions)
        all_t_loss = jnp.sum(all_t_loss, axis=-1)
        return all_t_loss / (self.nTransitions * self.nfold ** 2)

    @eqx.filter_jit
    def get_transition_matrix(self, points, i_mask, j_mask, fixed):
        r"""Toric transition matrix (uses pre-computed polynomial coefficients).

        Equivalent to ToricModel.get_transition_matrix in TF.
        """
        same_patch = jnp.where(jnp.all(i_mask == j_mask, axis=-1))[0]
        diff_patch = jnp.where(~jnp.all(i_mask == j_mask, axis=-1))[0]
        n_p = fixed.shape[0]
        n_p_red = diff_patch.shape[0]

        i_mask_red = i_mask[diff_patch]
        j_mask_red = j_mask[diff_patch]
        fixed_red = fixed[diff_patch]
        points_red = points[diff_patch]

        i_patch_indices = self._mask_to_patch_index(i_mask_red).reshape(-1, 1)
        j_patch_indices = self._mask_to_patch_index(j_mask_red).reshape(-1, 1)

        tij_indices = jnp.concatenate(
            [fixed_red, i_patch_indices, j_patch_indices], axis=-1)
        tij_degrees = self.transition_degrees[
            tij_indices[:, 0], tij_indices[:, 1], tij_indices[:, 2]]
        tij_coeff = self.transition_coefficients[
            tij_indices[:, 0], tij_indices[:, 1], tij_indices[:, 2]]
        tij_red = jnp.prod(
            points_red[:, None, None, :] ** tij_degrees, axis=-1)
        tij_red = tij_coeff * tij_red
        tij_red = jnp.transpose(tij_red, (0, 2, 1))

        tij_eye = jnp.tile(jnp.eye(self.nfold, dtype=jnp.complex64)[None],
                           (n_p - n_p_red, 1, 1))
        tij_all = jnp.zeros((n_p, self.nfold, self.nfold), dtype=jnp.complex64)
        tij_all = tij_all.at[diff_patch].set(tij_red)
        tij_all = tij_all.at[same_patch].set(tij_eye)
        return tij_all

    @eqx.filter_jit
    def _fubini_study_n_potentials(self, points, n=None, t=None):
        r"""Toric FS Kähler potential for Kahler modulus n.

        Equivalent to PhiFSModelToric._fubini_study_n_potentials in TF.
        """
        if t is None:
            t = jnp.complex64(1. + 0j)
        alpha = 0 if n is None else n
        degrees = self.sections[alpha]
        ms = jnp.prod(
            points[:, None, :] ** degrees[None, :, :], axis=-1)
        mss = ms * jnp.conj(ms)
        kappa_alphas = jnp.sum(mss, axis=-1)
        return (jnp.real(t / np.pi).astype(jnp.float32)
                * jnp.real(jnp.log(kappa_alphas)).astype(jnp.float32))

    def get_kahler_potential(self, points):
        r"""Toric FS Kähler potential K = sum_alpha t_alpha ln rho_alpha.

        Equivalent to PhiFSModelToric.get_kahler_potential in TF.
        """
        cpoints = (points[:, :self.ncoords]
                   + 1j * points[:, self.ncoords:]).astype(jnp.complex64)
        k_fs = self._fubini_study_n_potentials(cpoints, t=self.kmoduli[0])
        for i in range(1, len(self.kmoduli)):
            k_fs = k_fs + self._fubini_study_n_potentials(cpoints, i, t=self.kmoduli[i])
        k_fs = k_fs + jax.vmap(self.model)(points).reshape(-1)
        return k_fs


# ---------------------------------------------------------------------------
# PhiFSModelToric
# ---------------------------------------------------------------------------

class PhiFSModelToric(ToricModel):
    r"""g_out = g_FS' + del dbar phi_NN  (toric version).

    Equivalent to tensorflow/models/models.py::PhiFSModelToric.
    """

    def __init__(self, *args, **kwargs):
        super(PhiFSModelToric, self).__init__(*args, **kwargs)
        self.learn_kaehler = False

    def __call__(self, input_tensor, training=True, j_elim=None):
        r"""g_out_ij = g_FS'_ij + partial_i bar_partial_j phi_NN.

        Equivalent to PhiFSModelToric.call in TF.
        Uses nested jax.jacobian instead of tf.GradientTape.
        """
        ncoords = self.ncoords

        def phi_fn(x_single):
            # x_single is (n_in,); equinox Linear expects 1-D input.
            return self.model(x_single)[0]

        def d_phi_fn(x_single):
            return jax.grad(phi_fn)(x_single)

        dd_phi = jax.vmap(jax.jacobian(d_phi_fn))(input_tensor)

        nc = ncoords
        dx_dx_phi = 0.25 * dd_phi[:, :nc, :nc]
        dx_dy_phi = 0.25 * dd_phi[:, :nc, nc:]
        dy_dx_phi = 0.25 * dd_phi[:, nc:, :nc]
        dy_dy_phi = 0.25 * dd_phi[:, nc:, nc:]
        dd_phi_c = (dx_dx_phi + dy_dy_phi
                    + 1j * (dx_dy_phi - dy_dx_phi)).astype(jnp.complex64)

        pbs = self.pullbacks(input_tensor, j_elim=j_elim)
        dd_phi_pb = jnp.einsum('xai,xij,xbj->xab', pbs, dd_phi_c, jnp.conj(pbs))

        fs_cont = self.fubini_study_pb(input_tensor, pb=pbs, j_elim=j_elim)
        return fs_cont + dd_phi_pb

    def compute_transition_loss(self, points, num_random_scalings=10):
        r"""Toric Phi transition loss: phi(lambda^q z) == phi(z).

        Equivalent to PhiFSModelToric.compute_transition_loss in TF.
        """
        if num_random_scalings is None:
            return super(PhiFSModelToric, self).compute_transition_loss(points)

        cpoints = (points[:, :self.ncoords]
                   + 1j * points[:, self.ncoords:]).astype(jnp.complex64)
        num_pns = self.glsm_charges.shape[0]

        # Random scalings (same logic as TF version)
        key = jax.random.PRNGKey(0)
        key, k1, k2 = jax.random.split(key, 3)
        scale_factor_rand = jax.random.uniform(
            k1, shape=(num_random_scalings, num_pns),
            minval=0.1, maxval=0.9, dtype=jnp.float32).astype(jnp.complex64)
        scale_factor_rand = jnp.repeat(
            scale_factor_rand[:, :, None], self.ncoords, axis=-1)

        lambdas_rand_2 = jax.random.uniform(
            k2, shape=(num_random_scalings, num_pns, 2),
            minval=-1., maxval=1., dtype=jnp.float32)
        lambdas_rand = (lambdas_rand_2[:, :, 0]
                        + 1j * lambdas_rand_2[:, :, 1]).astype(jnp.complex64)
        lambdas_rand = jnp.repeat(lambdas_rand[:, :, None], self.ncoords, axis=-1)
        lambdas_rand = (scale_factor_rand * lambdas_rand
                        / (lambdas_rand * jnp.conj(lambdas_rand)) ** 0.5)
        lambdas_rand = lambdas_rand ** self.glsm_charges
        lambdas_rand = jnp.prod(lambdas_rand, axis=1)   # (num_random_scalings, ncoords)

        scaled_points = jnp.einsum('xi,ai->xai', cpoints, lambdas_rand)
        scaled_points = scaled_points.reshape(-1, self.ncoords)

        real_patch_points = jnp.concatenate(
            [jnp.real(scaled_points), jnp.imag(scaled_points)], axis=-1)
        gj = jax.vmap(self.model)(real_patch_points)
        gi = jnp.repeat(jax.vmap(self.model)(points), num_random_scalings, axis=0)
        all_t_loss = jnp.abs(gi - gj)
        all_t_loss = all_t_loss.reshape(-1, num_random_scalings)
        all_t_loss = jnp.sum(all_t_loss ** self.n[2], axis=-1)
        return all_t_loss / num_random_scalings


# ---------------------------------------------------------------------------
# MatrixFSModelToric
# ---------------------------------------------------------------------------

class MatrixFSModelToric(ToricModel):
    r"""g_out = g_FS' + g_FS' @ g_NN  (toric, matrix multiplication).

    Equivalent to tensorflow/models/models.py::MatrixFSModelToric.
    """

    def __init__(self, *args, **kwargs):
        super(MatrixFSModelToric, self).__init__(*args, **kwargs)

    def __call__(self, input_tensor, training=True, j_elim=None):
        r"""g_out_ik = g_FS'_ij (delta_jk + g_NN_jk).

        Equivalent to MatrixFSModelToric.call in TF.
        """
        nn_cont = self.to_hermitian(jax.vmap(self.model)(input_tensor))
        fs_cont = self.fubini_study_pb(input_tensor, j_elim=j_elim)
        return fs_cont + jnp.matmul(fs_cont, nn_cont)
