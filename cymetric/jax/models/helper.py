"""
Helper functions for JAX Calabi-Yau metric training.

Faithful translation of tensorflow/models/helper.py.
Uses optax for optimization (analogous to tf.keras.optimizers).
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

try:
    import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def prepare_basis(basis, dtype=jnp.complex64):
    r"""Casts each numpy array in BASIS to a JAX array of the given dtype.

    Equivalent to tensorflow/models/helper.py::prepare_basis.

    Args:
        basis (dict or str): Dictionary of geometric data or path to a
            pickle file containing such a dictionary.
        dtype (jax dtype, optional): Target dtype. Defaults to jnp.complex64.

    Returns:
        dict: same keys, but numpy arrays replaced with JAX arrays.
    """
    if isinstance(basis, str):
        import pickle
        with open(basis, 'rb') as f:
            basis = pickle.load(f)

    new_basis = {}
    for key in basis:
        if isinstance(basis[key], np.ndarray):
            new_basis[key] = jnp.array(basis[key], dtype=dtype)
        else:
            new_basis[key] = basis[key]
    return new_basis


# ---------------------------------------------------------------------------
# Per-step loss and gradient computation
# ---------------------------------------------------------------------------

def _make_train_step(optimizer):
    r"""Returns a JIT-compiled single training-step function.

    The returned function differentiates ONLY through the NN backbone
    (``model.model``), matching TF's ``trainable_variables = model.model.trainable_variables``.

    Args:
        optimizer (optax.GradientTransformation): An optax optimizer.

    Returns:
        callable: ``train_step(model, opt_state, x, y, sw)``
    """

    @eqx.filter_jit
    def train_step(model, opt_state, x, y, sample_weight=None, pb=None):
        def loss_fn(nn_model):
            # Reconstruct the full model with the candidate NN
            full_model = eqx.tree_at(lambda m: m.model, model, nn_model)
            total_loss, loss_dict = full_model.compute_loss(x, y, sample_weight, pb=pb)
            return jnp.mean(total_loss), loss_dict

        (mean_loss, loss_dict), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True)(model.model)

        # Replace NaNs with small gradient (matches TF: tf.where(is_nan, 1e-8, g))
        grads = jax.tree_util.tree_map(
            lambda g: jnp.where(jnp.isnan(g), jnp.full_like(g, 1e-8), g),
            grads)

        # Global norm clipping  (matches TF: tf.clip_by_global_norm)
        leaves, treedef = jax.tree_util.tree_flatten(grads)
        global_norm = jnp.sqrt(
            sum(jnp.sum(g ** 2) for g in leaves) + 1e-12)
        clip_factor = jnp.minimum(1.0, model.gclipping / global_norm)
        grads = jax.tree_util.tree_unflatten(
            treedef, [g * clip_factor for g in leaves])

        updates, new_opt_state = optimizer.update(grads, opt_state,
                                                  eqx.filter(model.model, eqx.is_array))
        new_nn = eqx.apply_updates(model.model, updates)
        new_model = eqx.tree_at(lambda m: m.model, model, new_nn)
        return new_model, new_opt_state, mean_loss, loss_dict

    return train_step


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(fsmodel, data, optimizer=None, epochs=50,
                batch_sizes=(64, 10000), verbose=1,
                custom_metrics=None, callbacks=None, sw=False):
    r"""Two-phase training loop for fixing the Kähler class.

    Equivalent to tensorflow/models/helper.py::train_model.

    Phase 1: small batch, volk loss disabled.
    Phase 2: large batch, only MA + volk loss.

    Args:
        fsmodel: Any JAX FreeModel (or subclass) instance.
        data (dict): {'X_train': ndarray, 'y_train': ndarray}.
        optimizer (optax.GradientTransformation, optional): Defaults to Adam.
        epochs (int): Number of training epochs. Defaults to 50.
        batch_sizes (list of int): [small, large]. Defaults to [64, 10000].
        verbose (int): If > 0, prints epoch info. Defaults to 1.
        custom_metrics (list, optional): Metric objects (see metrics.py).
        callbacks (list, optional): Callback objects (see callbacks.py).
        sw (bool): If True, use integration weights as sample weights.

    Returns:
        (model, training_history): updated model and loss history dict.
    """
    if callbacks is None:
        callbacks = []
    if custom_metrics is None:
        custom_metrics = []

    X_train = jnp.array(data['X_train'], dtype=jnp.float32)
    y_train = jnp.array(data['y_train'], dtype=jnp.float32)

    # pullbacks are independent of the network weights, so compute them once
    train_pullbacks = fsmodel.pullbacks(X_train)

    sample_weights = y_train[:, -2] if sw else None

    if optimizer is None:
        optimizer = optax.adam(1e-3)

    # Initialise optimizer state on the NN parameters only
    opt_state = optimizer.init(eqx.filter(fsmodel.model, eqx.is_array))

    training_history = {}

    # Notify callbacks
    for cb in callbacks:
        if hasattr(cb, 'on_train_begin'):
            cb.on_train_begin(logs=training_history, model=fsmodel)

    # Snapshot learning flags
    learn_kaehler = fsmodel.learn_kaehler
    learn_transition = fsmodel.learn_transition
    learn_ricci = fsmodel.learn_ricci
    learn_ricci_val = fsmodel.learn_ricci_val

    training_history['General loss phase'] = []
    training_history['Volume loss phase'] = []
    training_history['epochs'] = list(range(epochs))

    hist1, hist2 = {}, {}
    n_train = len(X_train)

    for epoch in range(epochs):
        if verbose > 0:
            print('\nEpoch {:2d}/{:d}'.format(epoch + 1, epochs))

        # ---- Phase 1: small batch, volk disabled ----
        fsmodel = eqx.tree_at(lambda m: m.learn_kaehler,   fsmodel, learn_kaehler)
        fsmodel = eqx.tree_at(lambda m: m.learn_transition, fsmodel, learn_transition)
        fsmodel = eqx.tree_at(lambda m: m.learn_ricci,     fsmodel, learn_ricci)
        fsmodel = eqx.tree_at(lambda m: m.learn_ricci_val, fsmodel, learn_ricci_val)
        fsmodel = eqx.tree_at(lambda m: m.learn_volk,      fsmodel, False)

        batch_size1 = batch_sizes[0]
        train_step = _make_train_step(optimizer)
        epoch_loss1 = 0.
        num_batches1 = 0
        indices = np.random.permutation(n_train)

        it1 = range(0, n_train, batch_size1)
        if HAS_TQDM and verbose > 0:
            it1 = tqdm.tqdm(it1)
        for start in it1:
            end = min(start + batch_size1, n_train)
            idx = indices[start:end]
            bx, by = X_train[idx], y_train[idx]
            bsw = sample_weights[idx] if sample_weights is not None else None
            fsmodel, opt_state, loss_val, loss_components = train_step(
                fsmodel, opt_state, bx, by, bsw, train_pullbacks[idx])
            epoch_loss1 += loss_val
            num_batches1 += 1
            # Update custom metrics
            if custom_metrics:
                loss_dict = {k: v for k, v in loss_components.items()}
                loss_dict['loss'] = jnp.array([loss_val])
                for m in custom_metrics:
                    m.update_state(loss_dict, bsw)

        avg_loss1 = epoch_loss1 / max(num_batches1, 1)
        training_history['General loss phase'].append(float(avg_loss1))
        for k, v in (loss_components or {}).items():
            if k not in hist1:
                hist1[k] = []
            hist1[k].append(float(jnp.mean(v)))

        # ---- Phase 2: large batch, only MA + volk ----
        fsmodel = eqx.tree_at(lambda m: m.learn_kaehler,   fsmodel, False)
        fsmodel = eqx.tree_at(lambda m: m.learn_transition, fsmodel, False)
        fsmodel = eqx.tree_at(lambda m: m.learn_ricci,     fsmodel, False)
        fsmodel = eqx.tree_at(lambda m: m.learn_ricci_val, fsmodel, False)
        fsmodel = eqx.tree_at(lambda m: m.learn_volk,      fsmodel, True)

        batch_size2 = min(batch_sizes[1], n_train)
        epoch_loss2 = 0.
        num_batches2 = 0
        indices2 = np.random.permutation(n_train)

        it2 = range(0, n_train, batch_size2)
        if HAS_TQDM and verbose > 0:
            it2 = tqdm.tqdm(it2)
        for start in it2:
            end = min(start + batch_size2, n_train)
            idx = indices2[start:end]
            bx, by = X_train[idx], y_train[idx]
            bsw = sample_weights[idx] if sample_weights is not None else None
            fsmodel, opt_state, loss_val, loss_components = train_step(
                fsmodel, opt_state, bx, by, bsw, train_pullbacks[idx])
            epoch_loss2 += loss_val
            num_batches2 += 1
            if custom_metrics:
                loss_dict = {k: v for k, v in loss_components.items()}
                loss_dict['loss'] = jnp.array([loss_val])
                for m in custom_metrics:
                    m.update_state(loss_dict, bsw)

        avg_loss2 = epoch_loss2 / max(num_batches2, 1)
        training_history['Volume loss phase'].append(float(avg_loss2))
        for k, v in (loss_components or {}).items():
            if k not in hist2:
                hist2[k] = []
            hist2[k].append(float(jnp.mean(v)))

        if verbose > 0:
            print(' - General loss phase: {:.6f}'.format(avg_loss1))
            print(' - Volume loss phase:  {:.6f}'.format(avg_loss2))

        # Run epoch-end callbacks.
        # Use a per-epoch dict so callback scalars are accumulated into lists
        # in training_history rather than overwriting each epoch.
        epoch_logs = {}
        for cb in callbacks:
            if hasattr(cb, 'on_epoch_end'):
                cb.on_epoch_end(epoch, logs=epoch_logs, model=fsmodel)
        for k, v in epoch_logs.items():
            if k not in training_history:
                training_history[k] = []
            training_history[k].append(v)

    # Restore flags
    fsmodel = eqx.tree_at(lambda m: m.learn_kaehler,   fsmodel, learn_kaehler)
    fsmodel = eqx.tree_at(lambda m: m.learn_transition, fsmodel, learn_transition)
    fsmodel = eqx.tree_at(lambda m: m.learn_ricci,     fsmodel, learn_ricci)
    fsmodel = eqx.tree_at(lambda m: m.learn_ricci_val, fsmodel, learn_ricci_val)
    fsmodel = eqx.tree_at(lambda m: m.learn_volk,      fsmodel, False)

    # Merge histories (matches TF helper logic)
    for k in set(list(hist1.keys()) + list(hist2.keys())):
        if k in hist2 and (k not in hist1 or max(hist2[k], default=0) != 0):
            training_history[k] = hist2[k]
        else:
            training_history[k] = hist1.get(k, [])

    # Run train-end callbacks
    for cb in callbacks:
        if hasattr(cb, 'on_train_end'):
            cb.on_train_end(logs=training_history, model=fsmodel)

    return fsmodel, training_history
