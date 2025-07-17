""" 
A collection of PyTorch callbacks.
"""
import torch
import numpy as np
from .measures import ricci_measure, sigma_measure, \
    kaehler_measure_loss, transition_measure_loss, ricci_scalar_fn


class BaseCallback:
    """Base class for all callbacks."""
    
    def on_epoch_begin(self, epoch, logs=None, model=None):
        pass
    
    def on_epoch_end(self, epoch, logs=None):
        pass
    
    def on_train_begin(self, logs=None, model=None):
        pass
    
    def on_train_end(self, logs=None, model=None):
        pass


class AlphaCallback(BaseCallback):
    """Callback that allows to manipulate the alpha factors."""
    
    def __init__(self, scheduler):
        """A callback that manipulates the alpha factors.

        Args:
            scheduler (function): A function that returns a list of alpha values
                and takes (int, dict, current_alpha) as args.
        """
        super(AlphaCallback, self).__init__()
        self.scheduler = scheduler

    def on_epoch_end(self, epoch, logs=None):
        r"""Manipulates alpha values according to function `scheduler`.

        Args:
            epoch (int): epoch
            logs (dict, optional): training logs. Defaults to None.
        """
        if hasattr(self, 'model') and hasattr(self.model, 'alpha'):
            new_alpha = self.scheduler(epoch, logs, self.model.alpha)
            if new_alpha is not None:
                self.model.alpha = new_alpha


class KaehlerCallback(BaseCallback):
    """Callback that tracks the weighted Kaehler measure."""
    
    def __init__(self, validation_data, nth=1, bSize=1000, initial=False, device=None):
        r"""A callback which computes the kaehler measure for
        the validation data after every epoch end.

        Args:
            validation_data (tuple(X_val, y_val)): Validation data.
            nth (int, optional): Run every n-th epoch. Defaults to 1.
            bSize (int, optional): Batch size. Defaults to 1000.
            initial (bool, optional): If True does one iteration before training.
                Defaults to False.
            device (torch.device, optional): Device to run on.
        """
        super(KaehlerCallback, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.X_val, self.y_val = validation_data
        self.X_val = torch.tensor(self.X_val, dtype=torch.float32, device=device)
        self.y_val = torch.tensor(self.y_val, dtype=torch.float32, device=device)
        self.weights = self.y_val[:, -2]
        self.omega = self.y_val[:, -1]
        self.nth = nth
        self.bSize = bSize
        self.initial = initial

    def on_epoch_end(self, epoch, logs=None, model=None):
        r"""Computes kaehler measure.

        Args:
            epoch (int): epoch
            logs (dict, optional): training logs. Defaults to None.
            model: The model to evaluate
        """
        if epoch % self.nth == 0 and model is not None:
            n_p = len(self.X_val)
            kaehler_losses = []
            
            # Keep model in train mode for PhiFSModel gradient computation
            model.train()
            for i in range(0, n_p, self.bSize):
                batch = self.X_val[i:i+self.bSize]
                # Ensure batch requires gradients for PhiFSModel
                batch.requires_grad_(True)
                try:
                    loss = model.compute_kaehler_loss(batch)
                    kaehler_losses.append(torch.mean(loss).item())
                except Exception as e:
                    print(f"Warning: Kaehler loss computation failed: {e}")
                    kaehler_losses.append(0.0)
            
            cb_res = np.mean(kaehler_losses)
            if logs is not None:
                if 'kaehler_val' not in logs:
                    logs['kaehler_val'] = [cb_res]
                else:
                    logs['kaehler_val'].append(cb_res)

            if cb_res <= 1e-3:
                print(f' - Kaehler measure val:    {cb_res:.4e}')
            else:
                print(f' - Kaehler measure val:    {cb_res:.4f}')

    def on_train_begin(self, logs=None, model=None):
        r"""Compute Kaehler measure before training as baseline.

        Args:
            logs (dict, optional): training logs. Defaults to None.
            model: The model to evaluate
        """
        if self.initial:
            self.on_epoch_end(-1, logs=logs, model=model)


class SigmaCallback(BaseCallback):
    """Callback that tracks the sigma measure."""
    
    def __init__(self, validation_data, nth=1, bSize=1000, initial=False, device=None):
        r"""A callback which computes the sigma measure for
        the validation data after every epoch end.

        Args:
            validation_data (tuple(X_val, y_val)): Validation data.
            nth (int, optional): Run every n-th epoch. Defaults to 1.
            bSize (int, optional): Batch size. Defaults to 1000.
            initial (bool, optional): If True does one iteration before training.
                Defaults to False.
            device (torch.device, optional): Device to run on.
        """
        super(SigmaCallback, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.X_val, self.y_val = validation_data
        self.X_val = torch.tensor(self.X_val, dtype=torch.float32, device=device)
        self.y_val = torch.tensor(self.y_val, dtype=torch.float32, device=device)
        self.nth = nth
        self.bSize = bSize
        self.initial = initial

    def on_epoch_end(self, epoch, logs=None, model=None):
        r"""Computes sigma measure.

        Args:
            epoch (int): epoch
            logs (dict, optional): training logs. Defaults to None.
            model: The model to evaluate
        """
        if epoch % self.nth == 0 and model is not None:
            n_p = len(self.X_val)
            sigma_losses = []
            
            # Keep model in train mode for PhiFSModel gradient computation
            model.train()
            for i in range(0, n_p, self.bSize):
                batch_x = self.X_val[i:i+self.bSize]
                batch_y = self.y_val[i:i+self.bSize]
                # Ensure batch requires gradients
                batch_x.requires_grad_(True)
                try:
                    y_pred = model(batch_x, training=True)
                    loss = model.sigma_loss(batch_y, y_pred)
                    sigma_losses.append(torch.mean(loss).item())
                except Exception as e:
                    print(f"Warning: Sigma loss computation failed: {e}")
                    sigma_losses.append(0.0)
            
            cb_res = np.mean(sigma_losses)
            if logs is not None:
                if 'sigma_val' not in logs:
                    logs['sigma_val'] = [cb_res]
                else:
                    logs['sigma_val'].append(cb_res)

            print(f' - Sigma measure val:      {cb_res:.4f}')

    def on_train_begin(self, logs=None, model=None):
        r"""Compute sigma measure before training as baseline.

        Args:
            logs (dict, optional): training logs. Defaults to None.
            model: The model to evaluate
        """
        if self.initial:
            self.on_epoch_end(-1, logs=logs, model=model)


class RicciCallback(BaseCallback):
    """Callback that tracks the Ricci measure."""
    
    def __init__(self, validation_data, validation_pullbacks, nth=1, bSize=1000, initial=False, device=None):
        r"""A callback which computes the ricci measure for
        the validation data after every epoch end.

        Args:
            validation_data (tuple(X_val, y_val)): Validation data.
            validation_pullbacks: Precomputed pullbacks for validation data.
            nth (int, optional): Run every n-th epoch. Defaults to 1.
            bSize (int, optional): Batch size. Defaults to 1000.
            initial (bool, optional): If True does one iteration before training.
                Defaults to False.
            device (torch.device, optional): Device to run on.
        """
        super(RicciCallback, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.X_val, self.y_val = validation_data
        self.X_val = torch.tensor(self.X_val, dtype=torch.float32, device=device)
        self.y_val = torch.tensor(self.y_val, dtype=torch.float32, device=device)
        self.val_pullbacks = torch.tensor(validation_pullbacks, dtype=torch.complex64, device=device)
        self.nth = nth
        self.bSize = bSize
        self.initial = initial

    def on_epoch_end(self, epoch, logs=None, model=None):
        r"""Computes ricci measure.

        Args:
            epoch (int): epoch
            logs (dict, optional): training logs. Defaults to None.
            model: The model to evaluate
        """
        if epoch % self.nth == 0 and model is not None:
            n_p = len(self.X_val)
            nfold = float(model.nfold)
            ricci_scalars, dets = [], []
            
            # Process in batches
            model.train()  # Keep in train mode for gradient computation
            for i in range(0, n_p, self.bSize):
                end_idx = min(i + self.bSize, n_p)
                X_batch = self.X_val[i:end_idx].clone().requires_grad_(True)
                pullbacks_batch = self.val_pullbacks[i:end_idx]
                
                try:
                    # Compute Ricci scalar and determinant
                    ricci_scalar_batch, det_batch = self._compute_ricci_scalar(
                        model, X_batch, pullbacks_batch
                    )
                    ricci_scalars.extend(ricci_scalar_batch.detach().cpu().numpy().tolist())
                    dets.extend(det_batch.detach().cpu().numpy().tolist())
                    
                except Exception as e:
                    print(f"Warning: Ricci computation failed for batch {i}: {e}")
                    # Fill with zeros for failed batch
                    batch_size = end_idx - i
                    ricci_scalars.extend([0.0] * batch_size)
                    dets.extend([1.0] * batch_size)  # Use 1.0 to avoid division issues
            
            if len(ricci_scalars) > 0:
                # Convert to tensors
                ricci_scalars = torch.tensor(ricci_scalars, dtype=torch.float32, device=self.device)
                dets = torch.tensor(dets, dtype=torch.float32, device=self.device)
                weights = self.y_val[:len(ricci_scalars), -2]
                omega = self.y_val[:len(ricci_scalars), -1]
                
                # Compute Ricci measure following TensorFlow implementation
                ricci_scalars = torch.abs(ricci_scalars)
                det_over_omega = dets / omega
                det_over_omega = torch.real(det_over_omega)
                vol_cy = torch.mean(weights)
                vol_k = torch.mean(det_over_omega * weights)
                
                ricci_measure = (vol_k**(1/nfold) / vol_cy) * torch.mean(
                    det_over_omega * ricci_scalars * weights
                )
                
                cb_res = ricci_measure.item()
            else:
                cb_res = 0.0
            
            if logs is not None:
                if 'ricci_val' not in logs:
                    logs['ricci_val'] = [cb_res]
                else:
                    logs['ricci_val'].append(cb_res)

            if cb_res <= 1e-3:
                print(f' - Ricci measure val:      {cb_res:.4e}')
            else:
                print(f' - Ricci measure val:      {cb_res:.4f}')
    
    def _compute_ricci_scalar(self, model, points, pullbacks):
        """Compute Ricci scalar following the TensorFlow implementation."""
        ncoords = model.ncoords
        
        # Ensure points require gradients
        points.requires_grad_(True)
        
        # Compute prediction and determinant  
        prediction = model(points, training=True)
        det = torch.real(torch.linalg.det(prediction))
        log_det = torch.log(torch.clamp(det, min=1e-10))  # Clamp to avoid log(0)
        
        # First derivatives of log(det(g))
        di_dg = torch.autograd.grad(
            outputs=log_det.sum(), inputs=points,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        
        # Second derivatives (Hessian) - manual batch jacobian
        batch_size = points.shape[0]
        input_dim = points.shape[1]
        didj_dg = torch.zeros(batch_size, input_dim, input_dim, 
                             device=points.device, dtype=torch.float32)
        
        # Compute Hessian components
        for i in range(input_dim):
            # Create selector for this component
            selector = torch.zeros_like(di_dg)
            selector[:, i] = 1.0
            
            grad_i = torch.autograd.grad(
                outputs=di_dg, inputs=points,
                grad_outputs=selector,
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            didj_dg[:, i, :] = grad_i
        
        # Convert to complex and construct Ricci tensor
        didj_dg = didj_dg.to(torch.complex64)
        
        # Combine derivatives to form complex Ricci tensor
        ricci_ij = didj_dg[:, 0:ncoords, 0:ncoords]
        ricci_ij += 1j * didj_dg[:, 0:ncoords, ncoords:]
        ricci_ij -= 1j * didj_dg[:, ncoords:, 0:ncoords]
        ricci_ij += didj_dg[:, ncoords:, ncoords:]
        ricci_ij *= 0.25
        
        # Compute Ricci scalar: R = g^{ij} R_{ij}
        pred_inv = torch.linalg.inv(prediction)
        ricci_scalar = torch.einsum('xba,xai,xij,xbj->x', 
                                   pred_inv, pullbacks, ricci_ij, torch.conj(pullbacks))
        ricci_scalar = torch.real(ricci_scalar)
        
        return ricci_scalar, det

    def on_train_begin(self, logs=None, model=None):
        r"""Compute ricci measure before training as baseline.

        Args:
            logs (dict, optional): training logs. Defaults to None.
            model: The model to evaluate
        """
        if self.initial:
            self.on_epoch_end(-1, logs=logs, model=model)


class TransitionCallback(BaseCallback):
    """Callback that tracks the transition loss weighted over the CY."""
    
    def __init__(self, validation_data, initial=False):
        r"""A callback which computes the transition measure for
        the validation data after every epoch end.

        Args:
            validation_data (tuple(X_val, y_val)): validation data
            initial (bool, optional): If True does one iteration before training.
                Defaults to False.
        """
        super(TransitionCallback, self).__init__()
        self.X_val, self.y_val = validation_data
        self.X_val = torch.tensor(self.X_val, dtype=torch.float32)
        self.y_val = torch.tensor(self.y_val, dtype=torch.float32)
        self.initial = initial
        
    def on_epoch_end(self, epoch, logs=None, model=None):
        r"""Computes transition measure.

        Args:
            epoch (int): epoch
            logs (dict, optional): training logs. Defaults to None.
            model: The model to evaluate
        """
        if model is None:
            return
            
        try:
            # Move model to correct device and set to train mode for gradients
            device = next(model.parameters()).device
            X_val_device = self.X_val.to(device)
            
            model.train()
            X_val_device.requires_grad_(True)
            
            # Compute transition measure using the proper function
            with torch.enable_grad():
                transition_loss = transition_measure_loss(model, X_val_device)
            
            cb_res = transition_loss.item()
            if logs is not None:
                if 'transition_val' not in logs:
                    logs['transition_val'] = [cb_res]
                else:
                    logs['transition_val'].append(cb_res)
                
            if cb_res <= 1e-3:
                print(' - Transition measure val: {:.4e}'.format(cb_res))
            else:
                print(' - Transition measure val: {:.4f}'.format(cb_res))
                
        except Exception as e:
            print(f" - Transition measure val: Error - {e}")

    def on_train_begin(self, logs=None, model=None):
        r"""Compute transition measure before training as baseline.

        Args:
            logs (dict, optional): training logs. Defaults to None.
            model: The model to evaluate
        """
        if self.initial:
            self.on_epoch_end(-1, logs=logs, model=model)


class VolkCallback(BaseCallback):
    r"""Callback that computes the volume from the metric."""
    
    def __init__(self, validation_data, nfold=3, initial=False):
        r"""A callback which computes Volk of the validation data
        after every epoch end.

        .. math::

            \text{Vol}_K = \int_X \omega^3

        Args:
            validation_data (tuple(X_val, y_val)): validation data
            nfold (int, optional): degree of CY. Defaults to 3.
            initial (bool, optional): If True does one iteration before training.
                Defaults to False.
        """
        super(VolkCallback, self).__init__()
        self.X_val, self.y_val = validation_data
        self.X_val = torch.tensor(self.X_val, dtype=torch.float32)
        self.y_val = torch.tensor(self.y_val, dtype=torch.float32)
        self.weights = self.y_val[:, -2]
        self.omega = self.y_val[:, -1]
        self.nfold = float(nfold)
        # NOTE: Check that convention is consistent with rest of code.
        self.factor = 1.0
        self.initial = initial

    def on_epoch_end(self, epoch, logs=None, model=None):
        r"""Tracks Volk during the training process.

        Args:
            epoch (int): epoch
            logs (dict, optional): training logs. Defaults to None.
            model: The model to evaluate
        """
        if model is None:
            return
            
        try:
            device = next(model.parameters()).device
            X_val_device = self.X_val.to(device)
            X_val_device.requires_grad_(True)
            weights_device = self.weights.to(device)
            omega_device = self.omega.to(device)
            
            model.train()
            prediction = model(X_val_device, training=False)
            volk = self.compute_volk(prediction, weights_device, omega_device, self.factor)

            cb_res = volk.item()
            if logs is not None:
                if 'volume_val' not in logs:
                    logs['volume_val'] = [cb_res]
                else:
                    logs['volume_val'].append(cb_res)

            if cb_res <= 1e-3:
                print(' - Volume val:               {:.4e}'.format(cb_res))
            else:
                print(' - Volume val:               {:.4f}'.format(cb_res))
                
        except Exception as e:
            print(f" - Volume val:               Error - {e}")

    def on_train_begin(self, logs=None, model=None):
        r"""Compute Volume loss before training as baseline.

        Args:
            logs (dict, optional): training logs. Defaults to None.
            model: The model to evaluate
        """
        if self.initial:
            self.on_epoch_end(-1, logs=logs, model=model)

    def compute_volk(self, pred, weights, omega, factor):
        r"""Vol k integrated over all points.

        .. math::

            \text{Vol}_K = \int_X \omega^3 
                = \frac{1}{N} \sum_p \frac{\det(g)}{\Omega \wedge \bar\Omega} w

        Note:
            This is different than the Volk-loss.

        Args:
            pred (torch.tensor([n_p, nfold, nfold], complex)): 
                Metric prediction.
            weights (torch.tensor([n_p], float)): Integration weights.
            omega (torch.tensor([n_p], float)): 
                :math:`\Omega \wedge \bar\Omega`.
            factor (float): Additional prefactors due to conventions.

        Returns:
            torch.tensor([], float): Vol k.
        """
        det = torch.real(torch.linalg.det(pred)) * factor
        volk_pred = torch.mean(det * weights / omega)
        return volk_pred
