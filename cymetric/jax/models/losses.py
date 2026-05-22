"""
Sigma (Monge-Ampere) loss function in JAX.

Faithful translation of the TensorFlow losses.py.
"""
import jax
import jax.numpy as jnp


def sigma_loss(kappa=1., nfold=3., flat=False):
    r"""MA loss.

    Equivalent to tensorflow/models/losses.py::sigma_loss.

    Args:
        kappa (float): inverse volume of the CY given by weights. Defaults to 1.
        nfold (float): dimension of the CY. Defaults to 3.
        flat (bool): True if metric is a flat tensor that needs to be
            assembled into a Hermitian matrix first. Defaults to False.

    Returns:
        function: MA loss function.
    """
    factorial = float(1.)
    nfold_int = int(nfold)
    kappa_f = float(kappa)
    det_factor = float(1.)

    def to_hermitian_vec(x):
        r"""Takes a tensor of shape (-1, nfold**2) and transforms it into
        (-1, nfold, nfold) Hermitian matrix.

        Equivalent to sigma_loss::to_hermitian_vec in TF.

        Args:
            x (jnp.ndarray, [bSize, nfold**2], float32): Flat real input.

        Returns:
            jnp.ndarray, [bSize, nfold, nfold], complex64.
        """
        t1 = jnp.reshape(x + 0j, (-1, nfold_int, nfold_int)).astype(jnp.complex64)
        up = jnp.triu(t1)                       # upper triangular  (TF: band_part(t1, 0, -1))
        low = jnp.tril(1j * t1)                 # lower triangular  (TF: band_part(1j*t1, -1, 0))
        # diagonal matrix from t1  (TF: band_part(t1, 0, 0))
        diag_vals = jnp.diagonal(t1, axis1=-2, axis2=-1)          # (bSize, nfold)
        diag_mat = jnp.einsum('...i,ij->...ij', diag_vals,
                              jnp.eye(nfold_int, dtype=jnp.complex64))
        out = up + jnp.swapaxes(up, -2, -1) - diag_mat
        return out + low + jnp.conj(jnp.swapaxes(low, -2, -1))

    def sigma_integrand_loss_flat(y_true, y_pred):
        r"""MA integrand loss for flat (vector) prediction.

        l = |1 - det(g) / (Omega ^ Omega_bar)|

        Equivalent to sigma_loss::sigma_integrand_loss_flat in TF.

        Args:
            y_true (jnp.ndarray, [bSize, x], float32): Labels; last column is
                Omega ^ Omega_bar.
            y_pred (jnp.ndarray, [bSize, nfold**2], float32): NN flat prediction.

        Returns:
            jnp.ndarray, [bSize], float32.
        """
        g = to_hermitian_vec(y_pred)
        omega_squared = y_true[:, -1]
        det = jnp.real(jnp.linalg.det(g)) * factorial / det_factor
        return jnp.abs(jnp.ones_like(omega_squared) - det / omega_squared / kappa_f)

    def sigma_integrand_loss(y_true, y_pred):
        r"""MA integrand loss for matrix prediction.

        l = |1 - det(g) / (Omega ^ Omega_bar)|

        Equivalent to sigma_loss::sigma_integrand_loss in TF.

        Args:
            y_true (jnp.ndarray, [bSize, x], float32): Labels; last column is
                Omega ^ Omega_bar.
            y_pred (jnp.ndarray, [bSize, nfold, nfold], complex64): Metric.

        Returns:
            jnp.ndarray, [bSize], float32.
        """
        omega_squared = y_true[:, -1]
        det = jnp.real(jnp.linalg.det(y_pred)) * factorial / det_factor
        return jnp.abs(jnp.ones_like(omega_squared) - det / omega_squared / kappa_f)

    if flat:
        return sigma_integrand_loss_flat
    else:
        return sigma_integrand_loss
