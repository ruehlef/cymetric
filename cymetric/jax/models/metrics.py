"""
Custom metrics for JAX Calabi-Yau models.

Faithful translation of tensorflow/models/metrics.py.
These are stateful running-average accumulators (no Keras dependency).
"""
import jax.numpy as jnp


class _RunningMean:
    """Simple running-mean accumulator (replaces tf.keras.metrics.Metric)."""

    def __init__(self, name):
        self.name = name
        self._value = 0.
        self._count = 0.

    # ---------- mirroring TF Metric API ----------

    def update_state(self, values, sample_weight=None):
        loss = values[self._loss_key]
        if sample_weight is not None:
            loss = loss * sample_weight
        new_value = (float(jnp.mean(loss)) - self._value) / (self._count + 1)
        self._value += new_value
        self._count += 1.

    def result(self):
        return self._value

    def reset_state(self):
        self._value = 0.
        self._count = 0.


class SigmaLoss(_RunningMean):
    """Running mean of sigma_loss.

    Equivalent to tensorflow/models/metrics.py::SigmaLoss.
    """
    _loss_key = 'sigma_loss'

    def __init__(self, name='sigma_loss'):
        super().__init__(name=name)


class KaehlerLoss(_RunningMean):
    """Running mean of kaehler_loss.

    Equivalent to tensorflow/models/metrics.py::KaehlerLoss.
    """
    _loss_key = 'kaehler_loss'

    def __init__(self, name='kaehler_loss'):
        super().__init__(name=name)


class TransitionLoss(_RunningMean):
    """Running mean of transition_loss.

    Equivalent to tensorflow/models/metrics.py::TransitionLoss.
    """
    _loss_key = 'transition_loss'

    def __init__(self, name='transition_loss'):
        super().__init__(name=name)


class RicciLoss(_RunningMean):
    """Running mean of ricci_loss.

    Equivalent to tensorflow/models/metrics.py::RicciLoss.
    """
    _loss_key = 'ricci_loss'

    def __init__(self, name='ricci_loss'):
        super().__init__(name=name)


class VolkLoss(_RunningMean):
    """Running mean of volk_loss.

    Equivalent to tensorflow/models/metrics.py::VolkLoss.
    """
    _loss_key = 'volk_loss'

    def __init__(self, name='volk_loss'):
        super().__init__(name=name)


class TotalLoss(_RunningMean):
    """Running mean of total loss.

    Equivalent to tensorflow/models/metrics.py::TotalLoss.
    """
    _loss_key = 'loss'

    def __init__(self, name='loss'):
        super().__init__(name=name)
