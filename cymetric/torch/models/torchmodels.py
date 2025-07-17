""" 
A selection of custom PyTorch models for learning 
Calabi-Yau metrics using neural networks.
"""
import torch
import torch.nn as nn
from cymetric.models.losses import sigma_loss
from cymetric.models.fubinistudy import FSModel
from cymetric.pointgen.nphelper import get_all_patch_degrees, compute_all_w_of_x, get_levicivita_tensor
import numpy as np


class FreeModel(FSModel):
    r"""FreeModel from which all other models inherit.

    The training and validation steps are implemented in this class. All
    other computational routines are inherited from:
    cymetric.models.fubinistudy.FSModel
    
    Example:
        Assume that `BASIS` and `data` have been generated with a point 
        generator.

        >>> import torch
        >>> import numpy as np
        >>> from cymetric.models.torchmodels import FreeModel
        >>> from cymetric.models.torchhelper import prepare_torch_basis
        >>> data = np.load('dataset.npz')
        >>> BASIS = prepare_torch_basis(np.load('basis.pickle', allow_pickle=True))
    
        set up the nn and FreeModel

        >>> nfold = 3
        >>> ncoords = data['X_train'].shape[1]
        >>> nn = torch.nn.Sequential(
        ...     torch.nn.Linear(ncoords, 64),
        ...     torch.nn.GELU(),
        ...     torch.nn.Linear(64, nfold**2),
        ... )
        >>> model = FreeModel(nn, BASIS)

        next we can compile and train

        >>> from cymetric.models.metrics import TotalLoss
        >>> metrics = [TotalLoss()]
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> # Training loop using standard PyTorch patterns
    """
    def __init__(self, torchmodel, BASIS, alpha=None, device=None, **kwargs):
        r"""FreeModel is a PyTorch model predicting CY metrics. 
        
        The output is
            
            .. math:: g_{\text{out}} = g_{\text{NN}}
        
        a hermitian (nfold, nfold) tensor with each float directly predicted
        from the neural network.

        NOTE:
            * The model by default does not train against the ricci loss.
                
                To enable ricci training, set `self.learn_ricci = True`,
                **before** training. For validation data 
                `self.learn_ricci_val = True`,
                can be modified separately.

            * The models loss contributions are

                1. sigma_loss
                2. kaehler loss
                3. transition loss
                4. ricci loss (disabled)
                5. volk loss

            * The different losses are weighted with alpha.

            * The (FB-) norms for each loss are specified with the keyword-arg

                >>> model = FreeModel(nn, BASIS, norm = [1. for _ in range(5)])

            * Set kappa to the kappa value of your training data.

                >>> kappa = np.mean(data['y_train'][:,-2])

        Args:
            torchmodel (torch.nn.Module): the underlying neural network.
            BASIS (dict): a dictionary containing all monomials and other
                relevant information from cymetric.pointgen.pointgen.
            alpha ([5//NLOSS], float): Weighting of each loss contribution.
                Defaults to None, which corresponds to equal weights.
            device (torch.device, optional): Device to run on. If None, auto-detects.
        """
        super(FreeModel, self).__init__(BASIS=BASIS, device=device, **kwargs)
        self.model = torchmodel
        self.NLOSS = 5
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize alpha parameters
        if alpha is not None:
            self.alpha = [torch.tensor(a, dtype=torch.float32, device=self.device, requires_grad=False) for a in alpha]
        else:
            self.alpha = [torch.tensor(1., dtype=torch.float32, device=self.device, requires_grad=False) for _ in range(self.NLOSS)]
        
        self.learn_kaehler = True
        self.learn_transition = True
        self.learn_ricci = False
        self.learn_ricci_val = False
        self.learn_volk = True

        self.custom_metrics = None
        self.kappa = torch.real(torch.tensor(BASIS['KAPPA'], dtype=torch.complex64, device=self.device)).float()
        self.gclipping = float(5.0)
        # add to compile?
        self.sigma_loss = sigma_loss(self.kappa, torch.tensor(self.nfold, dtype=torch.float32, device=self.device))

    def forward(self, input_tensor, training=True, j_elim=None):
        r"""Prediction of the NN.

        .. math:: g_{\text{out}} = g_{\text{NN}}

        The additional arguments are included for inheritance reasons.

        Args:
            input_tensor (torch.tensor([bSize, 2*ncoords], float32)): Points.
            training (bool, optional): Defaults to True.
            j_elim (torch.tensor([bSize, nHyper], int64), optional): 
                Coordinates(s) to be eliminated in the pullbacks.
                Not used in this model. Defaults to None.

        Returns:
            torch.tensor([bSize, nfold, nfold], complex):
                Prediction at each point.
        """
        # Set model to appropriate mode
        if training:
            self.model.train()
        else:
            self.model.eval()
            
        # nn prediction
        nn_output = self.model(input_tensor)
        return self.to_hermitian(nn_output)

    def to_hermitian(self, tensor):
        r"""Converts a real tensor to a hermitian matrix.
        
        Takes a tensor of length (-1,nfold**2) and transforms it
        into a (-1,nfold,nfold) hermitian matrix.
        
        This matches the TensorFlow implementation exactly.
        
        Args:
            tensor (torch.tensor): Real tensor of shape [..., n*n]
            
        Returns:
            torch.tensor: Hermitian tensor of shape [..., n, n]
        """
        batch_size = tensor.shape[0]
        
        # Convert to complex and reshape - matching TF exactly
        t1 = torch.complex(tensor, torch.zeros_like(tensor)).view(batch_size, self.nfold, self.nfold)
        
        # Upper triangular part (including diagonal)
        up = torch.triu(t1)
        
        # Lower triangular part (excluding diagonal) with imaginary factor
        low = torch.tril(1j * t1, diagonal=-1)
        
        # Construct hermitian matrix - matching TF exactly
        # up + up.H - diag(t1) + low + low.H
        out = up + torch.transpose(up, -2, -1) - torch.diag_embed(torch.diagonal(t1, dim1=-2, dim2=-1))
        
        return out + low + torch.transpose(torch.conj(low), -2, -1)

    def compute_loss(self, x, y, sample_weight=None):
        r"""Computes the total loss for training.

        Args:
            x (torch.tensor): Input points
            y (torch.tensor): Target data (weights and omega values)
            sample_weight (torch.tensor, optional): Sample weights

        Returns:
            torch.tensor: Total loss
        """
        # Get predictions
        y_pred = self(x, training=True)
        
        total_loss = 0.0
        
        # Sigma loss (always computed)
        # Pass the full y tensor, the loss function will extract omega
        sigma_loss_val = self.sigma_loss(y, y_pred)
        total_loss += self.alpha[0] * torch.mean(sigma_loss_val)
        
        # Kaehler loss
        if self.learn_kaehler:
            kaehler_loss_val = self.compute_kaehler_loss(x)
            total_loss += self.alpha[1] * torch.mean(kaehler_loss_val)
        
        # Transition loss
        if self.learn_transition:
            transition_loss_val = self.compute_transition_loss(x)
            total_loss += self.alpha[2] * torch.mean(transition_loss_val)
        
        # Ricci loss (usually disabled)
        if self.learn_ricci:
            ricci_loss_val = self.compute_ricci_loss(x)
            total_loss += self.alpha[3] * torch.mean(ricci_loss_val)
        
        # Volk loss
        if self.learn_volk:
            volk_loss_val = self.compute_volk_loss(x, y, y_pred)
            total_loss += self.alpha[4] * torch.mean(volk_loss_val)
        
        return total_loss
    
    def compute_transition_loss(self, points):
        r"""Computes transition loss between patches.
        
        Uses the full FSModel implementation for proper projective transition loss.
        
        Args:
            points (torch.tensor): Input points [batch_size, 2*ncoords]
            
        Returns:
            torch.tensor: Transition loss [batch_size]
        """
        if not points.requires_grad:
            points = points.detach().clone().requires_grad_(True)
        
        # Use the base FSModel's compute_transition_loss implementation
        return super().compute_transition_loss(points)
    
    def compute_ricci_loss(self, x):
        r"""Computes Ricci loss.
        
        Args:
            x (torch.tensor): Input points
            
        Returns:
            torch.tensor: Ricci loss
        """
        return super().compute_ricci_loss(x)
    
    def compute_volk_loss(self, x, y, pred=None):
        r"""Computes volk loss.

        NOTE:
            This is an integral over the batch. Thus batch dependent.

        .. math::

            \mathcal{L}_{\text{vol}_k} = |\int_B g_{\text{FS}} -
                \int_B g_{\text{out}}|_n

        Args:
            x (torch.tensor([bSize, 2*ncoords], torch.float32)): Input points.
            y (torch.tensor([bSize, 2], torch.float32)): Integration weights and omega.
                If None, will be extracted from current batch data.
            pred (torch.tensor([bSize, nfold, nfold], torch.complex64), optional):
                Prediction from `self(x)`. If None will be calculated.

        Returns:
            torch.tensor([bSize], torch.float32): Volk loss.
        """
        if pred is None:
            pred = self(x)
            
        # Extract weights - y[:, 0] contains weights, y[:, 1] contains omega values  
        aux_weights = (y[:, 0] / y[:, 1]).to(dtype=torch.complex64)
        aux_weights = aux_weights.unsqueeze(0).repeat(len(self.BASIS['KMODULI']), 1)
        
        # Create identity matrix for different Kahler moduli
        ks = torch.eye(len(self.BASIS['KMODULI']), dtype=torch.complex64, device=self.device)
        
        # Compute slopes for each Kahler parameter
        actual_slopes = []
        for i in range(len(self.BASIS['KMODULI'])):
            f_a = self.fubini_study_pb(x, ts=ks[i])
            slope = self._calculate_slope([pred, f_a])
            actual_slopes.append(slope)
        
        actual_slopes = torch.stack(actual_slopes, dim=0)
        actual_slopes = torch.mean(aux_weights * actual_slopes, dim=-1)
        
        # Compute loss against target slopes
        loss = torch.mean(torch.abs(actual_slopes - self.slopes.to(self.device))**self.n[4])
        
        # Return the loss repeated for each sample in the batch
        return loss.unsqueeze(0).repeat(len(y))


class MultFSModel(FreeModel):
    r"""MultFSModel inherits from :py:class:`FreeModel`.

    Example:
        Is identical to :py:class:`FreeModel`. Replace the model accordingly.
    """
    def __init__(self, *args, **kwargs):
        r"""MultFSModel is a PyTorch model predicting CY metrics.
        
        The output of this model has the following Ansatz
        
        .. math:: g_{\text{out}} = g_{\text{FS}} (1 + g_{\text{NN}})
        
        with elementwise multiplication and returns a hermitian (nfold, nfold)
        tensor.
        """
        super(MultFSModel, self).__init__(*args, **kwargs)

    def forward(self, input_tensor, training=True, j_elim=None):
        r"""Prediction of the model.

        .. math:: 
        
            g_{\text{out}; ij} = g_{\text{FS}; ij} (1_{ij} + g_{\text{NN}; ij})

        Args:
            input_tensor (torch.tensor([bSize, 2*ncoords], float32)): Points.
            training (bool, optional): Not used. Defaults to True.
            j_elim (torch.tensor([bSize, nHyper], int64), optional): 
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            torch.tensor([bSize, nfold, nfold], complex):
                Prediction at each point.
        """
        # Set model to appropriate mode
        if training:
            self.model.train()
        else:
            self.model.eval()
            
        # nn prediction
        nn_cont = self.to_hermitian(self.model(input_tensor))
        # fs metric
        fs_cont = self.fubini_study_pb(input_tensor, j_elim=j_elim)
        # return g_fs * (1 + g_NN)
        return fs_cont + torch.multiply(fs_cont, nn_cont)


class MatrixFSModel(FreeModel):
    r"""MatrixFSModel inherits from :py:class:`FreeModel`.

    Example:
        Is identical to :py:class:`FreeModel`. Replace the model accordingly.
    """
    def __init__(self, *args, **kwargs):
        r"""MatrixFSModel is a PyTorch model predicting CY metrics.
        
        The output of this model has the following Ansatz
        
        .. math:: g_{\text{out}} = g_{\text{FS}} (1 + g_{\text{NN}})
        
        with matrix multiplication and returns a hermitian (nfold, nfold)
        tensor.
        """
        super(MatrixFSModel, self).__init__(*args, **kwargs)

    def forward(self, input_tensor, training=True, j_elim=None):
        r"""Prediction of the model.

        .. math:: 
        
            g_{\text{out}; ik} = g_{\text{FS}; ij} (1_{jk} + g_{\text{NN}; jk})

        Args:
            input_tensor (torch.tensor([bSize, 2*ncoords], float32)): Points.
            training (bool, optional): Not used. Defaults to True.
            j_elim (torch.tensor([bSize, nHyper], int64), optional): 
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            torch.tensor([bSize, nfold, nfold], complex):
                Prediction at each point.
        """
        # Set model to appropriate mode
        if training:
            self.model.train()
        else:
            self.model.eval()
            
        nn_cont = self.to_hermitian(self.model(input_tensor))
        fs_cont = self.fubini_study_pb(input_tensor, j_elim=j_elim)
        return fs_cont + torch.matmul(fs_cont, nn_cont)


class AddFSModel(FreeModel):
    r"""AddFSModel inherits from :py:class:`FreeModel`.

    Example:
        Is identical to :py:class:`FreeModel`. Replace the model accordingly.
    """
    def __init__(self, *args, **kwargs):
        r"""AddFSModel is a PyTorch model predicting CY metrics.
        
        The output of this model has the following Ansatz
        
        .. math:: g_{\text{out}} = g_{\text{FS}} + g_{\text{NN}}
        
        and returns a hermitian (nfold, nfold) tensor.
        """
        super(AddFSModel, self).__init__(*args, **kwargs)

    def forward(self, input_tensor, training=True, j_elim=None):
        r"""Prediction of the model.

        .. math:: g_{\text{out}; ij} = g_{\text{FS}; ij}  + g_{\text{NN}; ij}

        Args:
            input_tensor (torch.tensor([bSize, 2*ncoords], float32)): Points.
            training (bool, optional): Not used. Defaults to True.
            j_elim (torch.tensor([bSize, nHyper], int64), optional): 
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            torch.tensor([bSize, nfold, nfold], complex64):
                Prediction at each point.
        """
        # Set model to appropriate mode
        if training:
            self.model.train()
        else:
            self.model.eval()
            
        nn_cont = self.to_hermitian(self.model(input_tensor))
        fs_cont = self.fubini_study_pb(input_tensor, j_elim=j_elim)
        return fs_cont + nn_cont


class PhiFSModel(FreeModel):
    r"""PhiFSModel inherits from :py:class:`FreeModel`.

    The PhiModel learns the scalar potential correction to some Kaehler metric
    to make it the Ricci-flat metric. The Kaehler metric is taken to be the 
    Fubini-Study metric.

    Example:
        Is similar to :py:class:`FreeModel`. Replace the nn accordingly.

        >>> nn = torch.nn.Sequential(
        ...     torch.nn.Linear(ncoords, 64),
        ...     torch.nn.GELU(),
        ...     torch.nn.Linear(64, 1),
        ... )
        >>> model = PhiFSModel(nn, BASIS)

    You have to use this model if you want to remain in the same Kaehler class
    specified by the Kaehler moduli.
    """
    def __init__(self, *args, **kwargs):
        r"""PhiFSModel is a PyTorch model predicting CY metrics.
        
        The output of this model has the following Ansatz
        
        .. math:: 
        
            g_{\text{out}} = g_{\text{FS}} + 
                \partial \bar{\partial} \phi_{\text{NN}}
        
        and returns a hermitian (nfold, nfold) tensor. The model is by
        definition Kaehler and thus this loss contribution is by default
        disabled. For similar reasons the Volk loss is also disabled if
        the last layer does not contain a bias. Otherwise it is required
        for successful tracing.
        """
        super(PhiFSModel, self).__init__(*args, **kwargs)
        # automatic in Phi network
        self.learn_kaehler = False

    def forward(self, input_tensor, training=True, j_elim=None):
        r"""Prediction of the model.

        .. math::

            g_{\text{out}; ij} = g_{\text{FS}; ij} + \
                partial_i \bar{\partial}_j \phi_{\text{NN}}

        Args:
            input_tensor (torch.tensor([bSize, 2*ncoords], float32)): Points.
            training (bool, optional): Not used. Defaults to True.
            j_elim (torch.tensor([bSize, nHyper], int64), optional):
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            torch.tensor([bSize, nfold, nfold], complex64):
                Prediction at each point.
        """
        # Ensure input requires gradients for second derivatives
        if not input_tensor.requires_grad:
            input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # NN prediction - disable training to match TF behavior 
        # (batch norm and dropout mix batches, making batch jacobian unreliable)
        self.model.eval()
        phi = self.model(input_tensor)
        
        # First derivative
        d_phi = torch.autograd.grad(
            outputs=phi.sum(),
            inputs=input_tensor,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Second derivatives (Hessian) - batch jacobian equivalent
        batch_size = input_tensor.shape[0]
        dd_phi = torch.zeros(batch_size, input_tensor.shape[1], input_tensor.shape[1], 
                           dtype=torch.float32, device=input_tensor.device)
        
        for i in range(input_tensor.shape[1]):
            grad_outputs = torch.zeros_like(d_phi)
            grad_outputs[:, i] = 1.0
            dd_phi[:, i, :] = torch.autograd.grad(
                outputs=d_phi,
                inputs=input_tensor,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]
        
        # Split into x and y derivatives (matching TF exactly)
        dx_dx_phi = 0.25 * dd_phi[:, :self.ncoords, :self.ncoords]
        dx_dy_phi = 0.25 * dd_phi[:, :self.ncoords, self.ncoords:]
        dy_dx_phi = 0.25 * dd_phi[:, self.ncoords:, :self.ncoords]
        dy_dy_phi = 0.25 * dd_phi[:, self.ncoords:, self.ncoords:]
        
        # Form complex Hessian: ∂∂̄φ = ∂²φ/∂z∂z̄
        dd_phi = torch.complex(dx_dx_phi + dy_dy_phi, dx_dy_phi - dy_dx_phi)
        
        # Apply pullbacks
        pbs = self.pullbacks(input_tensor, j_elim=j_elim)
        dd_phi = torch.einsum('xai,xij,xbj->xab', pbs, dd_phi, torch.conj(pbs))

        # Get Fubini-Study metric
        fs_cont = self.fubini_study_pb(input_tensor, pb=pbs, j_elim=j_elim)
        
        # Return g_fs + ∂∂̄φ
        return fs_cont + dd_phi

    def compute_transition_loss(self, points):
        r"""Computes transition loss at each point. In the case of the Phi model, 
        we demand that φ(λ^q_i z_i) = φ(z_i). This matches the TensorFlow implementation exactly.

        Args:
            points (torch.tensor([bSize, 2*ncoords], float32)): Points.

        Returns:
            torch.tensor([bSize], float32): Transition loss at each point.
        """
        device = points.device
        
        inv_one_mask = self._get_inv_one_mask(points)
        patch_indices = torch.where(~inv_one_mask)[1]
        patch_indices = patch_indices.view(-1, self.nProjective)
        current_patch_mask = self._indices_to_mask(patch_indices)
        
        fixed = self._find_max_dQ_coords(points)
        cpoints = torch.complex(points[:, :self.ncoords], points[:, self.ncoords:])
        
        if self.nhyper == 1:
            other_patches = self.fixed_patches[fixed]
            # Ensure other_patches is a tensor
            if not isinstance(other_patches, torch.Tensor):
                other_patches = torch.tensor(other_patches, device=device)
        else:
            combined = torch.cat((fixed, patch_indices), dim=-1)
            other_patches = self._generate_patches_vec(combined)
        
        other_patches = other_patches.view(-1, self.nProjective)
        other_patch_mask = self._indices_to_mask(other_patches)
        
        # NOTE: This will include same to same patch transitions
        # Match TF exactly: tf.repeat(cpoints, self.nTransitions, axis=-2)
        # For cpoints [batch_size, ncoords], axis=-2 is the batch dimension
        # This should repeat each point nTransitions times
        exp_points = cpoints.repeat_interleave(self.nTransitions, dim=0)
        
        patch_points = self._get_patch_coordinates(
            exp_points, other_patch_mask.bool())
        
        real_patch_points = torch.cat(
            (patch_points.real, patch_points.imag), dim=-1)
        
        # Match TF exactly: training=True flag and tf.repeat behavior
        # Match TF: gi shape [nTransitions, 1], gj shape [nTransitions, 1]
        # TF uses tf.repeat(self.model(points), self.nTransitions, axis=0)
        # But the input to self.model must match the shape of real_patch_points
        # So we need to repeat points to match real_patch_points shape
        batch_size = points.shape[0]
        n_trans = self.nTransitions
        # gi: repeat each model(points) n_trans times, shape [batch_size * n_trans, 1]
        gi = self.model(points)
        gi = gi.repeat_interleave(n_trans, dim=0)
        # gj: model(real_patch_points), shape [batch_size * n_trans, 1]
        gj = self.model(real_patch_points)
        # Ensure shapes match
        assert gi.shape == gj.shape, f"gi shape {gi.shape} != gj shape {gj.shape}"
        all_t_loss = torch.abs(gi - gj)
        all_t_loss = all_t_loss.view(-1, n_trans)
        all_t_loss = torch.sum(all_t_loss ** self.n[2], dim=-1)
        return all_t_loss / (n_trans * self.nfold ** 2)

    def get_kahler_potential(self, points):
        r"""Computes the Kahler potential.

        Args:
            points (torch.tensor([bSize, 2*ncoords], float32)): Points.

        Returns:
            torch.tensor([bSize], float32): Kahler potential.
        """
        if self.nProjective > 1:
            # we go through each ambient space factor and create the Kahler potential.
            cpoints = torch.complex(
                points[:, :self.degrees[0]],
                points[:, self.ncoords:self.ncoords+self.degrees[0]])
            k_fs = self._fubini_study_n_potentials(cpoints, t=self.BASIS['KMODULI'][0])
            for i in range(1, self.nProjective):
                s = torch.sum(self.degrees[:i])
                e = s + self.degrees[i]
                cpoints = torch.complex(points[:, s:e],
                                     points[:, self.ncoords+s:self.ncoords+e])
                k_fs_tmp = self._fubini_study_n_potentials(cpoints, t=self.BASIS['KMODULI'][i])
                k_fs += k_fs_tmp
        else:
            cpoints = torch.complex(
                points[:, :self.ncoords],
                points[:, self.ncoords:2*self.ncoords])
            k_fs = self._fubini_study_n_potentials(cpoints, t=self.BASIS['KMODULI'][0])

        k_fs += self.model(points).view(-1)
        return k_fs


class ToricModel(FreeModel):
    r"""ToricModel is the base class of toric CYs and inherits from
    :py:class:`FreeModel`.

    Example:
        Is similar to :py:class:`FreeModel` but requires additional toric_data.
        This one can be generated with :py:mod:`cymetric.sage.sagelib`.

        >>> #generate toric_data with sage_lib
        >>> import pickle
        >>> toric_data = pickle.load('toric_data.pickle')
        >>> model = ToricModel(nn, BASIS, toric_data=toric_data)

    ToricModel does **not** train the underlying neural network. Instead, it 
    always predicts a generalization of the kaehler metric for toric CYs.
    """
    def __init__(self, *args, **kwargs):
        r"""ToricModel is the equivalent to
        :py:class:`cymetric.models.fubinistudy.FSModel`.

        It will not learn the Ricci-flat metric, but can be used as a baseline
        to compare the neural network against.

        NOTE:
            1. Requires nevertheless a nn in its (kw)args.

            2. Requires `toric_data = toric_data` in its kwargs.
        """
        if 'toric_data' in kwargs.keys():
            self.toric_data = kwargs['toric_data']
            del kwargs['toric_data']
        
        self.nfold = self.toric_data['dim_cy']
        self.sections = [torch.tensor(m, dtype=torch.complex64) for m in self.toric_data['exps_sections']]
        self.patch_masks = np.array(self.toric_data['patch_masks'], dtype=bool)
        self.glsm_charges = np.array(self.toric_data["glsm_charges"])
        self.nPatches = len(self.patch_masks)
        self.nProjective = len(self.toric_data["glsm_charges"])
        
        super(ToricModel, self).__init__(*args, **kwargs)
        
        # Move sections to device
        self.sections = [s.to(self.device) for s in self.sections]
        self.kmoduli = torch.tensor(self.BASIS['KMODULI'], device=self.device)
        self.lc = torch.tensor(get_levicivita_tensor(self.nfold), dtype=torch.complex64, device=self.device)
        self.slopes = self._target_slopes()

    def forward(self, input_tensor, training=True, j_elim=None):
        r"""Computes the equivalent of the pullbacked 
        Fubini-Study metric at each point in input_tensor.

        .. math:: J = t^\alpha J_\alpha

        Args:
            input_tensor (torch.tensor([bSize, 2*ncoords], float32)): Points.
            training (bool, optional): Defaults to True.
            j_elim (torch.tensor([bSize, nHyper], int64), optional): 
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            torch.tensor([bSize, nfold, nfold], complex):
                Prediction at each point.
        """
        # FS prediction
        return self.fubini_study_pb(input_tensor, j_elim=j_elim)

    def fubini_study_pb(self, points, pb=None, j_elim=None, ts=None):
        r"""Computes the pullbacked Fubini-Study metric.

        NOTE:
            The pb argument overwrites j_elim.

        .. math::

            g_{ij} = \frac{1}{\pi} J_i^a \bar{J}_j^b \partial_a 
                \bar{\partial}_b \ln |\vec{z}|^2


        Args:
            points (torch.tensor([bSize, 2*ncoords], float32)): Points.
            pb (torch.tensor([bSize, nfold, ncoords], float32)):
                Pullback at each point. Overwrite j_elim. Defaults to None.
            j_elim (torch.tensor([bSize], int64)): index to be eliminated. 
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.
            ts (torch.tensor([len(kmoduli)], complex64)):
                Kahler parameters. Defaults to the ones specified at time of point generation

        Returns:
            torch.tensor([bSize, nfold, nfold], complex64):
                FS-metric at each point.
        """
        if ts is None:
            ts = self.BASIS['KMODULI']
        # NOTE: Cannot use super since for toric models we have only one toric space, but more than one Kahler modulus
        pullbacks = self.pullbacks(points, j_elim=j_elim) if pb is None else pb
        cpoints = torch.complex(points[:, :self.ncoords], points[:, self.ncoords:])

        Js = self._fubini_study_n_metrics(cpoints, n=0, t=ts[0])
        if len(self.kmoduli) != 1:
            for i in range(1, len(self.kmoduli)):
                Js += self._fubini_study_n_metrics(cpoints, n=i, t=ts[i])

        gFSpb = torch.einsum('xai,xij,xbj->xab', pullbacks, Js, torch.conj(pullbacks))
        return gFSpb

    def _fubini_study_n_metrics(self, points, n=None, t=torch.complex(torch.tensor(1.), torch.tensor(0.))):
        r"""Computes the Fubini-Study equivalent on the ambient space for each
        Kaehler modulus.

        .. math:: g_\alpha = \partial_i \bar\partial_j \ln \rho_\alpha

        Args:
            points (torch.tensor([bSize, ncoords], complex64)): Points.
            n (int, optional): n^th Kahler potential term. Defaults to None.
            t (torch.complex, optional): Volume factor. Defaults to 1+0j.

        Returns:
            torch.tensor([bSize, ncoords, ncoords], complex64): 
                Metric contribution at each point for t_n.
        """
        alpha = 0 if n is None else n 
        degrees = self.sections[alpha]
        ms = torch.pow(points[:, None, :], degrees[None, :, :])
        ms = torch.prod(ms, dim=-1)
        mss = ms * torch.conj(ms)     
        kappa_alphas = torch.sum(mss, dim=-1)
        zizj = points[:, :, None] * torch.conj(points[:, None, :])
        J_alphas = 1.0 / zizj
        J_alphas = torch.einsum('x,xab->xab', 1.0 / (kappa_alphas**2), J_alphas)
        coeffs = torch.einsum('xa,xb,ai,aj->xij', mss, mss, degrees, degrees) - torch.einsum('xa,xb,ai,bj->xij', mss, mss, degrees, degrees)
        return J_alphas * coeffs * t / torch.tensor(np.pi, dtype=torch.complex64, device=self.device)

    def _generate_helpers(self):
        """Additional helper functions."""
        super()._generate_helpers()
        
        self.nTransitions = int(np.max(np.sum(~self.patch_masks, axis=-2)))
        self.fixed_patches = self._generate_all_patches()
        
        # Convert to torch tensors
        patch_degrees = get_all_patch_degrees(self.glsm_charges, self.patch_masks)
        w_of_x, del_w_of_x, del_w_of_z = compute_all_w_of_x(patch_degrees, self.patch_masks)
        
        self.patch_degrees = torch.tensor(patch_degrees, dtype=torch.complex64, device=self.device)
        self.transition_coefficients = torch.tensor(w_of_x, dtype=torch.complex64, device=self.device)
        self.transition_degrees = torch.tensor(del_w_of_z, dtype=torch.complex64, device=self.device)
        self.patch_masks = torch.tensor(self.patch_masks, dtype=torch.bool, device=self.device)

    def _generate_all_patches(self):
        """Torics only have one hypersurface, thus we can generate all patches"""
        # fixed patches will be of shape (ncoords, npatches, nTransitions)
        fixed_patches = np.repeat(np.arange(self.nPatches), self.nTransitions)
        fixed_patches = np.tile(fixed_patches, self.ncoords)
        fixed_patches = fixed_patches.reshape(
            (self.ncoords, self.nPatches, self.nTransitions))
        for i in range(self.ncoords):
            # keep each coordinate fixed and add all patches, where its zero
            all_patches = ~self.patch_masks[:, i]
            all_indices = np.where(all_patches)[0]
            fixed_patches[i, all_indices, 0:len(all_indices)] = all_indices * \
                np.ones((len(all_indices), len(all_indices)), dtype=np.int)
        return torch.tensor(fixed_patches, dtype=torch.int64, device=self.device)


class PhiFSModelToric(ToricModel):
    r"""PhiFSModelToric inherits from :py:class:`ToricModel`.

    The PhiModel learns the scalar potential correction to some Kaehler metric
    to make it the Ricci-flat metric. The Kaehler metric is taken to be a toric 
    equivalent of the Fubini-Study metric. See also :py:class:`PhiFSModel`.

    Example:
        Is similar to :py:class:`FreeModel`. Replace the nn accordingly.

        >>> nn = torch.nn.Sequential(
        ...     torch.nn.Linear(ncoords, 64),
        ...     torch.nn.GELU(),
        ...     torch.nn.Linear(64, 1),
        ... )
        >>> model = PhiFSModelToric(nn, BASIS, toric_data = toric_data)

    You have to use this model if you want to remain in the same Kaehler class
    specified by the Kaehler moduli.
    """
    def __init__(self, *args, **kwargs):
        r"""PhiFSModelToric is a PyTorch model predicting CY metrics.
        
        The output of this model has the following Ansatz
        
        .. math:: 
            
            g_{\text{out}} = g_{\text{FS'}} +
                \partial \bar{\partial} \phi_{\text{NN}}
        
        and returns a hermitian (nfold, nfold) tensor. The model is by
        definition Kaehler and thus this loss contribution is by default
        disabled.
        """
        super(PhiFSModelToric, self).__init__(*args, **kwargs)
        self.learn_kaehler = False

    def forward(self, input_tensor, training=True, j_elim=None):
        r"""Prediction of the model.

        .. math:: 
            g_{\text{out}; ij} = g_{\text{FS'}; ij} +
                \partial_i \bar{\partial}_j \phi_{\text{NN}}

        Args:
            input_tensor (torch.tensor([bSize, 2*ncoords], float32)): Points.
            training (bool, optional): Not used. Defaults to True.
            j_elim (torch.tensor([bSize, nHyper], int64), optional): 
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            torch.tensor([bSize, nfold, nfold], complex64):
                Prediction at each point.
        """
        # Similar implementation to PhiFSModel but using toric metric
        input_tensor.requires_grad_(True)
        
        # Set model to appropriate mode
        if training:
            self.model.train()
        else:
            self.model.eval()
        
        # First derivative
        phi = self.model(input_tensor)
        d_phi = torch.autograd.grad(
            outputs=phi, inputs=input_tensor,
            grad_outputs=torch.ones_like(phi),
            create_graph=True, retain_graph=True
        )[0]
        
        # Second derivatives (Hessian)
        dd_phi = torch.zeros(input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[1], 
                           device=self.device, dtype=input_tensor.dtype)
        for i in range(input_tensor.shape[1]):
            dd_phi[:, i, :] = torch.autograd.grad(
                outputs=d_phi[:, i], inputs=input_tensor,
                grad_outputs=torch.ones_like(d_phi[:, i]),
                create_graph=True, retain_graph=True
            )[0]
        
        # Split into x and y derivatives
        dx_dx_phi = 0.25 * dd_phi[:, :self.ncoords, :self.ncoords]
        dx_dy_phi = 0.25 * dd_phi[:, :self.ncoords, self.ncoords:]
        dy_dx_phi = 0.25 * dd_phi[:, self.ncoords:, :self.ncoords]
        dy_dy_phi = 0.25 * dd_phi[:, self.ncoords:, self.ncoords:]
        
        dd_phi = torch.complex(dx_dx_phi + dy_dy_phi, dx_dy_phi - dy_dx_phi)
        pbs = self.pullbacks(input_tensor, j_elim=j_elim)
        dd_phi = torch.einsum('xai,xij,xbj->xab', pbs, dd_phi, torch.conj(pbs))
        
        # fs metric
        fs_cont = self.fubini_study_pb(input_tensor, pb=pbs, j_elim=j_elim)
        # return g_fs + \del\bar\del\phi
        return fs_cont + dd_phi


class MatrixFSModelToric(ToricModel):
    r"""MatrixFSModelToric inherits from :py:class:`ToricModel`.

    See also: :py:class:`MatrixFSModel` and :py:class:`FreeModel`
    """
    def __init__(self, *args, **kwargs):
        r"""MatrixFSModelToric is a PyTorch model predicting CY metrics.
        
        The output of this model has the following Ansatz
        
        .. math:: g_{\text{out}} = g_{\text{FS'}} (1 + g_{\text{NN}})
        
        with matrix multiplication and returns a hermitian (nfold, nfold)
        tensor.
        """
        super(MatrixFSModelToric, self).__init__(*args, **kwargs)

    def forward(self, input_tensor, training=True, j_elim=None):
        r"""Prediction of the model.

        .. math:: 
        
            g_{\text{out}; ik} = g_{\text{FS}; ij} (1_{jk} + g_{\text{NN}; jk})

        Args:
            input_tensor (torch.tensor([bSize, 2*ncoords], float32)): Points.
            training (bool, optional): Not used. Defaults to True.
            j_elim (torch.tensor([bSize, nHyper], int64), optional): 
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            torch.tensor([bSize, nfold, nfold], complex):
                Prediction at each point.
        """
        # Set model to appropriate mode
        if training:
            self.model.train()
        else:
            self.model.eval()
            
        nn_cont = self.to_hermitian(self.model(input_tensor))
        fs_cont = self.fubini_study_pb(input_tensor, j_elim=j_elim)
        return fs_cont + torch.matmul(fs_cont, nn_cont)
