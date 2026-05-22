"""
JAX callbacks for Calabi-Yau metric training.

Faithful translation of tensorflow/models/callbacks.py.
Callbacks are plain Python objects (no Keras dependency).
"""
import jax
import jax.numpy as jnp
import numpy as np

from .measures import (
    ricci_measure, sigma_measure,
    kaehler_measure_loss, transition_measure_loss, ricci_scalar_fn,
)


class AlphaCallback:
    """Callback that manipulates the alpha loss-weighting factors.

    Equivalent to tensorflow/models/callbacks.py::AlphaCallback.
    """

    def __init__(self, scheduler):
        """
        Args:
            scheduler (callable): Returns a new alpha list; takes
                (epoch, logs, model.alpha) as arguments.
        """
        self.manipulater = scheduler

    def on_epoch_end(self, epoch, logs=None, model=None):
        r"""Manipulates alpha values.

        Args:
            epoch (int): current epoch.
            logs (dict): training history.
            model: JAX FreeModel (or subclass) instance.
        """
        if model is not None:
            model.alpha = self.manipulater(epoch, logs, model.alpha)

    def on_train_begin(self, logs=None, model=None):
        pass

    def on_train_end(self, logs=None, model=None):
        pass


class KaehlerCallback:
    """Callback that tracks the weighted Kähler measure.

    Equivalent to tensorflow/models/callbacks.py::KaehlerCallback.
    """

    def __init__(self, validation_data, nth=1, bSize=1000, initial=False):
        r"""
        Args:
            validation_data (tuple): (X_val, y_val).
            nth (int): Run every n-th epoch. Defaults to 1.
            bSize (int): Batch size for evaluation. Defaults to 1000.
            initial (bool): If True, run before training. Defaults to False.
        """
        self.X_val = jnp.array(validation_data[0], dtype=jnp.float32)
        self.y_val = jnp.array(validation_data[1], dtype=jnp.float32)
        self.weights = self.y_val[:, -2]
        self.omega = self.y_val[:, -1]
        self.nth = nth
        self.bSize = bSize
        self.initial = initial

    def on_epoch_end(self, epoch, logs=None, model=None):
        r"""Computes Kähler measure.

        Args:
            epoch (int): current epoch.
            logs (dict): training history dict (mutated in-place).
            model: JAX FreeModel (or subclass) instance.
        """
        if epoch % self.nth == 0 and model is not None:
            n_p = len(self.X_val)
            kaehler_losses = []
            for start in range(0, n_p, self.bSize):
                batch = self.X_val[start:start + self.bSize]
                loss = float(kaehler_measure_loss(model, batch))
                kaehler_losses.append(loss)
            # Re-weight last batch (same logic as TF callback)
            last_batch_size = n_p % self.bSize
            if last_batch_size != 0 and len(kaehler_losses) > 0:
                kaehler_losses[-1] *= last_batch_size / self.bSize
            cb_res = float(np.mean(kaehler_losses))
            if logs is not None:
                logs['kaehler_val'] = cb_res
            if cb_res <= 1e-3:
                print(' - Kaehler measure val:    {:.4e}'.format(cb_res))
            else:
                print(' - Kaehler measure val:    {:.4f}'.format(cb_res))

    def on_train_begin(self, logs=None, model=None):
        if self.initial:
            self.on_epoch_end(-1, logs=logs, model=model)

    def on_train_end(self, logs=None, model=None):
        pass


class RicciCallback:
    """Callback that tracks the Ricci measure.

    Equivalent to tensorflow/models/callbacks.py::RicciCallback.
    """

    def __init__(self, validation_data, pullbacks, verbose=0,
                 bSize=1000, nth=1, hlevel=0, initial=False):
        r"""
        Args:
            validation_data (tuple): (X_val, y_val).
            pullbacks (array-like, [n_p, nfold, ncoord]): Pullback tensors.
            verbose (int): Verbosity. Defaults to 0.
            bSize (int): Batch size. Defaults to 1000.
            nth (int): Run every n-th epoch. Defaults to 1.
            hlevel (int): Extra statistics level. Defaults to 0.
            initial (bool): If True, run before training. Defaults to False.
        """
        self.X_val = jnp.array(validation_data[0], dtype=jnp.float32)
        self.y_val = jnp.array(validation_data[1], dtype=jnp.float32)
        self.weights = self.y_val[:, -2]
        self.vol_cy = float(jnp.mean(self.weights))
        self.omega = self.y_val[:, -1]
        self.pullbacks = jnp.array(pullbacks, dtype=jnp.complex64)
        self.verbose = verbose
        self.hlevel = hlevel
        self.nth = nth
        self.bSize = bSize
        self.initial = initial

    def on_epoch_end(self, epoch, logs=None, model=None):
        r"""Computes Ricci measure.

        Args:
            epoch (int): current epoch.
            logs (dict): training history dict (mutated in-place).
            model: JAX FreeModel (or subclass) instance.
        """
        if epoch % self.nth == 0 and model is not None:
            n_p = len(self.X_val)
            nfold = float(model.nfold)
            ricci_scalars_list = []
            dets_list = []
            for start in range(0, n_p, self.bSize):
                end = min(start + self.bSize, n_p)
                X_batch = self.X_val[start:end]
                pb_batch = self.pullbacks[start:end]
                rs_batch, det_batch = ricci_scalar_fn(
                    model, X_batch, pullbacks=pb_batch,
                    verbose=self.verbose, rdet=True)
                ricci_scalars_list.append(np.array(rs_batch))
                dets_list.append(np.array(det_batch))

            ricci_scalars = jnp.array(np.concatenate(ricci_scalars_list))
            dets = jnp.array(np.concatenate(dets_list))
            ricci_scalars = jnp.abs(ricci_scalars)
            det_over_omega = dets / self.omega
            vol_k = float(jnp.mean(det_over_omega * self.weights))
            ricci = ((vol_k ** (1. / nfold)) / self.vol_cy
                     * float(jnp.mean(det_over_omega * ricci_scalars * self.weights)))

            if logs is not None:
                logs['ricci_val'] = ricci
            if self.hlevel > 0:
                logs['ricci_val_mean'] = float(jnp.mean(ricci_scalars))
                if self.hlevel > 1:
                    rs_np = np.array(ricci_scalars)
                    logs['ricci_val_median'] = float(np.median(rs_np))
                    logs['ricci_val_var'] = float(np.var(rs_np))
                    logs['ricci_val_std'] = float(np.std(rs_np))
                    if self.hlevel > 2:
                        dets_np = np.array(dets)
                        logs['ricci_val_dets'] = float(np.sum(dets_np < 0) / len(dets_np))
            if ricci <= 1e-3:
                print(' - Ricci measure val:      {:.4e}'.format(ricci))
            else:
                print(' - Ricci measure val:      {:.4f}'.format(ricci))

    def on_train_begin(self, logs=None, model=None):
        if self.initial:
            self.on_epoch_end(-1, logs=logs, model=model)

    def on_train_end(self, logs=None, model=None):
        pass


class SigmaCallback:
    """Callback that tracks the sigma measure.

    Equivalent to tensorflow/models/callbacks.py::SigmaCallback.
    """

    def __init__(self, validation_data, initial=False):
        r"""
        Args:
            validation_data (tuple): (X_val, y_val).
            initial (bool): If True, run before training. Defaults to False.
        """
        self.X_val = jnp.array(validation_data[0], dtype=jnp.float32)
        self.y_val = jnp.array(validation_data[1], dtype=jnp.float32)
        self.initial = initial

    def on_epoch_end(self, epoch, logs=None, model=None):
        r"""Computes sigma measure.

        Args:
            epoch (int): current epoch.
            logs (dict): training history dict (mutated in-place).
            model: JAX FreeModel (or subclass) instance.
        """
        if model is not None:
            sig = float(sigma_measure(model, self.X_val, self.y_val))
            if logs is not None:
                logs['sigma_val'] = sig
            if sig <= 1e-3:
                print(' - Sigma measure val:      {:.4e}'.format(sig))
            else:
                print(' - Sigma measure val:      {:.4f}'.format(sig))

    def on_train_begin(self, logs=None, model=None):
        if self.initial:
            self.on_epoch_end(-1, logs=logs, model=model)

    def on_train_end(self, logs=None, model=None):
        pass


class TransitionCallback:
    """Callback that tracks the transition measure.

    Equivalent to tensorflow/models/callbacks.py::TransitionCallback.
    """

    def __init__(self, validation_data, initial=False):
        r"""
        Args:
            validation_data (tuple): (X_val, y_val).
            initial (bool): If True, run before training. Defaults to False.
        """
        self.X_val = jnp.array(validation_data[0], dtype=jnp.float32)
        self.y_val = jnp.array(validation_data[1], dtype=jnp.float32)
        self.initial = initial

    def on_epoch_end(self, epoch, logs=None, model=None):
        r"""Computes transition measure.

        Args:
            epoch (int): current epoch.
            logs (dict): training history dict (mutated in-place).
            model: JAX FreeModel (or subclass) instance.
        """
        if model is not None:
            trans = float(transition_measure_loss(model, self.X_val))
            if logs is not None:
                logs['transition_val'] = trans
            if trans <= 1e-3:
                print(' - Transition measure val: {:.4e}'.format(trans))
            else:
                print(' - Transition measure val: {:.4f}'.format(trans))

    def on_train_begin(self, logs=None, model=None):
        if self.initial:
            self.on_epoch_end(-1, logs=logs, model=model)

    def on_train_end(self, logs=None, model=None):
        pass


class VolkCallback:
    r"""Callback that computes the Kähler volume from the metric.

    Equivalent to tensorflow/models/callbacks.py::VolkCallback.
    """

    def __init__(self, validation_data, nfold=3, initial=False):
        r"""
        Args:
            validation_data (tuple): (X_val, y_val).
            nfold (int): CY dimension. Defaults to 3.
            initial (bool): If True, run before training. Defaults to False.
        """
        self.X_val = jnp.array(validation_data[0], dtype=jnp.float32)
        self.y_val = jnp.array(validation_data[1], dtype=jnp.float32)
        self.weights = self.y_val[:, -2]
        self.omega = self.y_val[:, -1]
        self.nfold = float(nfold)
        self.factor = float(1.)
        self.initial = initial

    def on_epoch_end(self, epoch, logs=None, model=None):
        r"""Computes Volk.

        Args:
            epoch (int): current epoch.
            logs (dict): training history dict (mutated in-place).
            model: JAX FreeModel (or subclass) instance.
        """
        if model is not None:
            prediction = model(self.X_val)
            volk = float(self._compute_volk(prediction, self.weights,
                                            self.omega, self.factor))
            if logs is not None:
                logs['volk_val'] = volk
            if volk <= 1e-3:
                print(' - Volk val:               {:.4e}'.format(volk))
            else:
                print(' - Volk val:               {:.4f}'.format(volk))

    def on_train_begin(self, logs=None, model=None):
        if self.initial:
            self.on_epoch_end(-1, logs=logs, model=model)

    def on_train_end(self, logs=None, model=None):
        pass

    @staticmethod
    @jax.jit
    def _compute_volk(pred, weights, omega, factor):
        r"""Vol_K integrated over all points.

        .. math::

            \mathrm{Vol}_K = \frac{1}{N} \sum_p \frac{\det(g)}{|\Omega|^2} w

        Equivalent to VolkCallback.compute_volk in TF.
        """
        det = jnp.real(jnp.linalg.det(pred)) * factor
        return jnp.mean(det * weights / omega)
