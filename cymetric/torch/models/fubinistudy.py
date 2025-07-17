""" 
Pullbacked fubini study metric implemented as a PyTorch nn.Module.
"""
import torch
import torch.nn as nn
import itertools as it
from cymetric.pointgen.nphelper import generate_monomials, get_levicivita_tensor
import numpy as np


class FSModel(nn.Module):
    r"""FSModel implements all underlying PyTorch routines for pullbacks 
    and computing various loss contributions.

    It is *not* intended for actual training and does not have an explicit 
    training step included. It should be used to write your own custom models
    for training of CICYs. Toric hypersurfaces require some extra routines,
    which are implemented here: `cymetric.models.torchmodels.ToricModel`
    """
    def __init__(self, BASIS, norm=None, device=None):
        r"""A PyTorch implementation of the pulled back Fubini-Study metric.

        Args:
            BASIS (dict): a dictionary containing all monomials and other
                relevant information from e.g.
                `cymetric.pointgen.pointgen.PointGenerator`
            norm ([5//NLOSS], optional): degree of norm for various losses.
                Defaults to 1 for all but Kaehler norm (2).
            device (torch.device, optional): Device to run computations on.
                If None, uses CUDA if available, else CPU.
        """
        super(FSModel, self).__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.BASIS = BASIS
        self.ncoords = len(self.BASIS['DQDZB0'])
        self.nProjective = len(self.BASIS['AMBIENT'])
        self.nfold = int(torch.real(torch.tensor(self.BASIS['NFOLD'], dtype=torch.complex64)))
        
        if norm is None:
            self.n = [torch.tensor(1., dtype=torch.float32, device=self.device) for _ in range(5)]
            # Default: we want to punish violation of kählerity stronger
            self.n[1] = torch.tensor(2., dtype=torch.float32, device=self.device)
        else:
            self.n = [torch.tensor(n, dtype=torch.float32, device=self.device) for n in norm]
            
        # projective vars
        ambient_np = self.BASIS['AMBIENT'].cpu().numpy() if isinstance(self.BASIS['AMBIENT'], torch.Tensor) else self.BASIS['AMBIENT']
        # Handle complex tensors properly
        if ambient_np.dtype == complex or ambient_np.dtype == np.complex64 or ambient_np.dtype == np.complex128:
            ambient_np = np.real(ambient_np)
        self.degrees = torch.tensor(
            np.ones_like(ambient_np, dtype=np.int32) + ambient_np.astype(np.int32), 
            dtype=torch.int32, device=self.device
        )
        self.pi = torch.tensor(np.pi, dtype=torch.complex64, device=self.device)
        self.nhyper = int(self.BASIS['NHYPER'])
        self._generate_helpers()
        
    def _generate_helpers(self):
        r"""Bunch of helper functions to run during initialization"""
        self.lc = torch.tensor(
            get_levicivita_tensor(self.nfold), 
            dtype=torch.complex64, device=self.device
        )
        self.proj_matrix = self._generate_proj_matrix()
        self.nTransitions = self._patch_transitions()
        if self.nhyper == 1:
            self.fixed_patches = self._generate_all_patches()
        self._proj_indices = self._generate_proj_indices()
        self.slopes = self._target_slopes()

    def _generate_proj_matrix(self):
        r"""PyTorch allows for nice slicing, but we keep this structure for
        compatibility. Here we create `proj_matrix` which stores information 
        about the ambient spaces, so that we can slice via matrix products. 
        See usage in: `self.fubini_study_pb`.
        """
        proj_matrix = {}
        for i in range(self.nProjective):
            matrix = torch.zeros(
                (self.degrees[i], self.ncoords),
                dtype=torch.complex64, device=self.device
            )
            s = torch.sum(self.degrees[:i])
            e = torch.sum(self.degrees[:i+1])
            matrix[:, s:e] = torch.eye(self.degrees[i], dtype=torch.complex64, device=self.device)
            proj_matrix[str(i)] = matrix
        return proj_matrix

    def _generate_proj_indices(self):
        r"""Makes a tensor with corresponding projective index for each variable
        from the ambient space.
        """
        flat_list = []
        for i, p in enumerate(self.degrees):
            for _ in range(p):
                flat_list += [i]
        return torch.tensor(flat_list, dtype=torch.int64, device=self.device)

    def _generate_all_patches(self):
        r"""We generate all possible patches for CICYs. Note for CICYs with
        more than one hypersurface patches are generated on spot.
        """
        fixed_patches = []
        for i in range(self.ncoords):
            all_patches = np.array(
                list(it.product(*[[j for j in range(sum(self.degrees[:k]), sum(self.degrees[:k+1])) if j != i] for k in range(len(self.degrees))], repeat=1)))
            if len(all_patches) == self.nTransitions:
                fixed_patches += [all_patches]
            else:
                # need to pad if there are less than nTransitions.
                all_patches = np.tile(all_patches, (int(self.nTransitions/len(all_patches)) + 1, 1))
                fixed_patches += [all_patches[0:self.nTransitions]]
        fixed_patches = np.array(fixed_patches)
        return torch.tensor(fixed_patches, dtype=torch.int64, device=self.device)

    def _patch_transitions(self):
        r"""Computes the maximum number of patch transitions with same fixed
        variables. This is often not the same number for all patches. In case
        there are less transitions we padd with same to same patches."""
        nTransitions = 0
        for t in generate_monomials(self.nProjective, self.nhyper):
            tmp_deg = [d-t[j] for j, d in enumerate(self.degrees)]
            n = torch.prod(torch.tensor(tmp_deg))
            if n > nTransitions:
                nTransitions = n
        return int(nTransitions)

    def _target_slopes(self):
        ks = torch.eye(len(self.BASIS['KMODULI']), dtype=torch.complex64, device=self.device)
        
        if self.nfold == 1:
            slope = torch.einsum('a, xa->x', 
                               torch.tensor(self.BASIS['INTNUMS'], device=self.device), ks)

        elif self.nfold == 2:
            kmoduli_tensor = self.BASIS['KMODULI'].to(self.device) if isinstance(self.BASIS['KMODULI'], torch.Tensor) else torch.tensor(self.BASIS['KMODULI'], device=self.device)
            intnums_tensor = self.BASIS['INTNUMS'].to(self.device) if isinstance(self.BASIS['INTNUMS'], torch.Tensor) else torch.tensor(self.BASIS['INTNUMS'], device=self.device)
            slope = torch.einsum('ab, a, xb->x', intnums_tensor, kmoduli_tensor, ks)

        elif self.nfold == 3:
            kmoduli_tensor = self.BASIS['KMODULI'].to(self.device) if isinstance(self.BASIS['KMODULI'], torch.Tensor) else torch.tensor(self.BASIS['KMODULI'], device=self.device)
            intnums_tensor = self.BASIS['INTNUMS'].to(self.device) if isinstance(self.BASIS['INTNUMS'], torch.Tensor) else torch.tensor(self.BASIS['INTNUMS'], device=self.device)
            slope = torch.einsum('abc, a, b, xc->x', intnums_tensor, kmoduli_tensor, kmoduli_tensor, ks)
        
        elif self.nfold == 4:
            slope = torch.einsum('abcd, a, b, c, xd->x', 
                               torch.tensor(self.BASIS['INTNUMS'], device=self.device), 
                               torch.tensor(self.BASIS['KMODULI'], device=self.device), 
                               torch.tensor(self.BASIS['KMODULI'], device=self.device), 
                               torch.tensor(self.BASIS['KMODULI'], device=self.device), ks)
        
        elif self.nfold == 5:
            slope = torch.einsum('abcde, a, b, c, d, xe->x', 
                               torch.tensor(self.BASIS['INTNUMS'], device=self.device), 
                               torch.tensor(self.BASIS['KMODULI'], device=self.device), 
                               torch.tensor(self.BASIS['KMODULI'], device=self.device), 
                               torch.tensor(self.BASIS['KMODULI'], device=self.device), 
                               torch.tensor(self.BASIS['KMODULI'], device=self.device), ks)

        else:
            raise NotImplementedError('Only implemented for nfold <= 5. Run the tensor contraction yourself :).')
        
        return slope

    def _calculate_slope(self, args):
        r"""Computes the slopes \mu(F_i) = \int J \wedge J \wegde F_i at the point in Kahler moduli space t_a = 1 for all a
        and for F_i = O_X(0, 0,... , 1, 0, ..., 0), i.e. the flux integers are k_i^a = \delta_{i,a}"""
        pred, f_a = args[0], args[1]
        if self.nfold == 1:
            slope = torch.einsum('xab->x', f_a)
        elif self.nfold == 2:
            slope = torch.einsum('xab,xcd,ac,bd->x',
                              pred, f_a, self.lc, self.lc)
        elif self.nfold == 3:
            slope = torch.einsum('xab,xcd,xef,ace,bdf->x',
                              pred, pred, f_a, self.lc, self.lc)
        elif self.nfold == 4:
            slope = torch.einsum('xab,xcd,xef,xgh,aceg,bdfh->x',
                              pred, pred, pred, f_a, self.lc, self.lc)
        elif self.nfold == 5:
            slope = torch.einsum('xab,xcd,xef,xgh,xij,acegi,bdfhj->x',
                              pred, pred, pred, pred, f_a, self.lc, self.lc)
        else:
            raise NotImplementedError('Only slopes for nfold <= 5')
        
        slope = (1./torch.exp(torch.lgamma(torch.real(torch.tensor(self.BASIS['NFOLD'], dtype=torch.complex64, device=self.device)).float() + 1))) * slope
        return slope

    def forward(self, input_tensor, training=True, j_elim=None):
        r"""Forward method. Computes the pullbacked 
        Fubini-Study metric at each point in input_tensor.

        Args:
            input_tensor (torch.tensor([bSize, 2*ncoords], float)): Points.
            training (bool, optional): Switch between training and eval mode. Not used at the moment
            j_elim (torch.array([bSize], int64)): index to be eliminated.
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            torch.tensor([bSize, nfold, nfold], complex): 
                Pullbacked FS-metric at each point.
        """
        return self.fubini_study_pb(input_tensor, j_elim=j_elim)

    def compute_kaehler_loss(self, x):
        r"""Computes Kähler loss.

        .. math::
            \cal{L}_{\text{dJ}} = \sum_{ijk} ||Re(c_{ijk})||_n + 
                    ||Im(c_{ijk})||_n \\
                \text{with: } c_{ijk} = g_{i\bar{j},k} - g_{k\bar{j},i}

        Args:
            x (torch.tensor([bSize, 2*ncoords], float)): Points.

        Returns:
            torch.tensor([bSize, 1], float): \sum_ijk abs(cijk)**n
        """
        x.requires_grad_(True)
        # Compute the metric tensor
        y_pred = self(x, training=True)
        pb = self.pullbacks(x)
        
        batch_size = x.shape[0]
        nfold = y_pred.shape[1]
        input_dim = x.shape[1]
        
        # Get real and imaginary parts
        gij_re, gij_im = torch.real(y_pred), torch.imag(y_pred)
        
        # Manual batch jacobian computation - more reliable than torch.func
        # Initialize jacobian tensors
        gijk_re = torch.zeros(batch_size, nfold, nfold, input_dim, dtype=torch.float32, device=x.device)
        gijk_im = torch.zeros(batch_size, nfold, nfold, input_dim, dtype=torch.float32, device=x.device)
        
        # Compute jacobian for each output element
        for i in range(nfold):
            for j in range(nfold):
                # Create output selector for this (i,j) component
                output_selector = torch.zeros_like(gij_re)
                output_selector[:, i, j] = 1.0
                
                # Compute gradient for real part
                grad_re = torch.autograd.grad(
                    outputs=gij_re,
                    inputs=x,
                    grad_outputs=output_selector,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]
                gijk_re[:, i, j, :] = grad_re
                
                # Compute gradient for imaginary part  
                grad_im = torch.autograd.grad(
                    outputs=gij_im,
                    inputs=x,
                    grad_outputs=output_selector,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]
                gijk_im[:, i, j, :] = grad_im
        
        # Convert to complex and proceed with original computation
        gijk_re = gijk_re.to(torch.complex64)
        gijk_im = gijk_im.to(torch.complex64)
        
        # Construct the c_ijk tensor
        cijk = 0.5*(gijk_re[:, :, :, :self.ncoords] +
                    gijk_im[:, :, :, self.ncoords:] +
                    1.j*gijk_im[:, :, :, :self.ncoords] -
                    1.j*gijk_re[:, :, :, self.ncoords:])
        
        # Apply pullbacks
        cijk_pb = torch.einsum('xija,xka->xijk', cijk, pb)
        
        # Antisymmetrize: c_ijk - c_kji
        cijk_pb = cijk_pb - torch.transpose(cijk_pb, 1, 3)
        
        # Compute the loss
        cijk_loss = torch.sum(torch.abs(cijk_pb)**self.n[1], dim=[1, 2, 3])
        
        return cijk_loss

    def compute_ricci_scalar(self, points, pb=None):
        r"""Computes the Ricci scalar for each point.

        .. math::

            R = g^{ij} J_i^a \bar{J}_j^b \partial_a \bar{\partial}_b 
                \log \det g

        Args:
            points (torch.tensor([bSize, 2*ncoords], float)): Points.
            pb (torch.tensor([bSize, nfold, ncoords], float), optional):
                Pullback tensor at each point. Defaults to None.

        Returns:
            torch.tensor([bSize], float): R|_p.
        """
        x_vars = points.clone().requires_grad_(True)
        
        # Compute prediction and log determinant
        prediction = self(x_vars, training=True)
        det = torch.real(torch.det(prediction))
        log_det = torch.log(det)
        
        # First derivatives of log(det(g))
        di_dg = torch.autograd.grad(
            outputs=log_det.sum(),
            inputs=x_vars,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Second derivatives (Hessian) of log(det(g))
        batch_size = x_vars.shape[0]
        input_dim = x_vars.shape[1]
        
        didj_dg = torch.zeros(batch_size, input_dim, input_dim, dtype=torch.float32, device=x_vars.device)
        
        for i in range(input_dim):
            # Create a selector for the i-th component of the gradient
            grad_selector = torch.zeros_like(di_dg)
            grad_selector[:, i] = 1.0
            
            # Compute the gradient of the i-th component
            grad_i = torch.autograd.grad(
                outputs=di_dg,
                inputs=x_vars,
                grad_outputs=grad_selector,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            didj_dg[:, i, :] = grad_i
        
        # Convert to complex and construct the complex Hessian
        didj_dg = didj_dg.to(torch.complex64)
        
        # Construct complex second derivative tensor
        ricci_ij = didj_dg[:, :self.ncoords, :self.ncoords]
        ricci_ij += 1j * didj_dg[:, :self.ncoords, self.ncoords:]
        ricci_ij -= 1j * didj_dg[:, self.ncoords:, :self.ncoords]
        ricci_ij += didj_dg[:, self.ncoords:, self.ncoords:]
        ricci_ij *= 0.25
        
        # Get metric inverse and pullbacks
        pred_inv = torch.linalg.inv(prediction)
        if pb is None:
            pullbacks = self.pullbacks(points)
        else:
            pullbacks = pb
        
        # Compute Ricci scalar: g^{ij} J_i^a \bar{J}_j^b \partial_a \bar{\partial}_b \log det g
        ricci_scalar = torch.einsum('xba,xai,xij,xbj->x', 
                                   pred_inv, pullbacks, ricci_ij, torch.conj(pullbacks))
        ricci_scalar = torch.real(ricci_scalar)
        
        return ricci_scalar

    def compute_ricci_loss(self, points, pb=None):
        r"""Computes the absolute value of the Ricci scalar for each point. Since negative
        Ricci scalars are bad, we take a loss of \|1-e^-ricci\|^p. This will exponentially
        punish negative Ricci scalars, and it vanishes for Ricci scalar 0

        .. seealso:: method :py:meth:`.compute_ricci_scalar`.

        Args:
            points (torch.tensor([bSize, 2*ncoords], float)): Points.
            pb (torch.tensor([bSize, nfold, ncoords], float), optional):
                Pullback tensor at each point. Defaults to None.

        Returns:
            torch.tensor([bSize], float): \|R\|_n.
        """
        ricci_scalar = self.compute_ricci_scalar(points, pb)
        
        return torch.abs(1 - torch.exp(-ricci_scalar))

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
            ts = self.BASIS['KMODULI'].detach().clone().to(self.device)
        # TODO: Naming conventions here and in pointgen are different.
        if self.nProjective > 1:
            # we go through each ambient space factor and create fs.
            cpoints = torch.complex(
                points[:, :self.degrees[0]],
                points[:, self.ncoords:self.ncoords+self.degrees[0]])
            fs = self._fubini_study_n_metrics(cpoints, n=self.degrees[0], t=ts[0])
            fs = torch.einsum('xij,ia,bj->xab', fs, self.proj_matrix['0'], torch.transpose(self.proj_matrix['0'], 0, 1))
            for i in range(1, self.nProjective):
                s = torch.sum(self.degrees[:i])
                e = s + self.degrees[i]
                cpoints = torch.complex(points[:, s:e],
                                     points[:, self.ncoords+s:self.ncoords+e])
                fs_tmp = self._fubini_study_n_metrics(
                    cpoints, n=self.degrees[i], t=ts[i])
                fs_tmp = torch.einsum('xij,ia,bj->xab',
                                   fs_tmp, self.proj_matrix[str(i)],
                                   torch.transpose(self.proj_matrix[str(i)], 0, 1))
                fs += fs_tmp
        else:
            cpoints = torch.complex(
                points[:, :self.ncoords],
                points[:, self.ncoords:2*self.ncoords])
            fs = self._fubini_study_n_metrics(cpoints, t=ts[0])

        if pb is None:
            # Disable compiled version for now due to double backward compatibility issues
            pb = self.pullbacks(points, j_elim=j_elim)
        fs_pb = torch.einsum('xai,xij,xbj->xab', pb, fs, torch.conj(pb))
        return fs_pb
    
    def pullbacks(self, points, j_elim=None):
        r"""Computes the pullback tensor at each point.
        
        This is the uncompiled version used for callbacks and debugging.
        For training, use pullbacks_compiled() for better performance.

        .. math::

            J^i_a = \frac{dz_i}{dx_a}

        where x_a are the nfold good coordinates after eliminating j_elim.

        Args:
            points (torch.tensor([bSize, 2*ncoords], float32)): Points.
            j_elim (torch.tensor([bSize, nHyper], int64), optional): 
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            torch.tensor([bSize, nfold, ncoords], complex64): Pullback at each
                point.
        """
        return self._pullbacks_impl(points, j_elim, compiled=False)
    
    def pullbacks_compiled(self, points, j_elim=None):
        r"""Computes the pullback tensor at each point using compiled operations.
        
        This is the optimized version used during training for better performance.

        Args:
            points (torch.tensor([bSize, 2*ncoords], float32)): Points.
            j_elim (torch.tensor([bSize, nHyper], int64), optional): 
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            torch.tensor([bSize, nfold, ncoords], complex64): Pullback at each
                point.
        """
        # For now, create a compiled version if torch.compile is available
        if hasattr(torch, 'compile') and not hasattr(self, '_compiled_pullbacks_impl'):
            try:
                self._compiled_pullbacks_impl = torch.compile(self._pullbacks_impl)
            except:
                # Fallback if compilation fails
                self._compiled_pullbacks_impl = self._pullbacks_impl
        elif not hasattr(self, '_compiled_pullbacks_impl'):
            self._compiled_pullbacks_impl = self._pullbacks_impl
            
        return self._compiled_pullbacks_impl(points, j_elim, compiled=True)
        
    def _pullbacks_impl(self, points, j_elim=None, compiled=True):
        r"""Internal implementation of pullbacks computation.
        
        Args:
            points (torch.tensor([bSize, 2*ncoords], float32)): Points.
            j_elim (torch.tensor([bSize, nHyper], int64), optional): 
                Coordinates(s) to be eliminated in the pullbacks.
            compiled (bool): Whether to use compiled operations for performance.
                
        Returns:
            torch.tensor([bSize, nfold, ncoords], complex64): Pullback at each point.
        """
        batch_size = points.shape[0]
        inv_one_mask = self._get_inv_one_mask(points)
        cpoints = torch.complex(points[:, :self.ncoords], points[:, self.ncoords:])
        
        if j_elim is None:
            dQdz_indices = self._find_max_dQ_coords(points)
        else:
            dQdz_indices = j_elim
            
        full_mask = inv_one_mask.float()
        for i in range(self.nhyper):
            dQdz_mask = -1. * torch.nn.functional.one_hot(dQdz_indices[:, i], self.ncoords).float()
            full_mask = full_mask + dQdz_mask
            
        full_mask = full_mask.bool()
        x_z_indices = torch.where(full_mask)
        good_indices = x_z_indices[1].unsqueeze(1)  # Column indices
        
        pullbacks = torch.zeros((batch_size, self.nfold, self.ncoords), dtype=torch.complex64, device=self.device)
        
        # Create diagonal elements (identity for good coordinates)
        y_indices = torch.arange(self.nfold, device=self.device).unsqueeze(0).repeat(batch_size, 1).flatten().unsqueeze(1)
        batch_indices = x_z_indices[0].unsqueeze(1)  # Batch indices
        
        # Set diagonal elements to 1
        for i in range(self.nfold):
            mask = y_indices.flatten() == i
            if mask.any():
                b_idx = batch_indices[mask].flatten()
                g_idx = good_indices[mask].flatten()
                pullbacks[b_idx, i, g_idx] = 1.0 + 0.0j
        
        # Compute derivative terms
        fixed_indices = dQdz_indices.flatten().unsqueeze(1)
        
        for i in range(self.nhyper):
            # compute p_i\alpha eq (5.24)
            pia_polys = self.BASIS['DQDZB'+str(i)][good_indices.flatten()]
            pia_factors = self.BASIS['DQDZF'+str(i)][good_indices.flatten()]
            
            # Expand cpoints for computation
            pia = cpoints.unsqueeze(1).repeat(1, self.nfold, 1).flatten(0, 1).unsqueeze(1)
            pia = torch.pow(pia, pia_polys.unsqueeze(0))
            pia = torch.prod(pia, dim=-1)
            pia = torch.sum(pia_factors.unsqueeze(0) * pia, dim=-1)
            pia = pia.reshape(-1, 1, self.nfold)
            
            if i == 0:
                dz_hyper = pia
            else:
                dz_hyper = torch.cat((dz_hyper, pia), dim=1)
                
            # compute p_ifixed
            pif_polys = self.BASIS['DQDZB'+str(i)][fixed_indices.flatten()]
            pif_factors = self.BASIS['DQDZF'+str(i)][fixed_indices.flatten()]
            
            pif = cpoints.unsqueeze(1).repeat(1, self.nhyper, 1).flatten(0, 1).unsqueeze(1)
            pif = torch.pow(pif, pif_polys.unsqueeze(0))
            pif = torch.prod(pif, dim=-1)
            pif = torch.sum(pif_factors.unsqueeze(0) * pif, dim=-1)
            pif = pif.reshape(-1, 1, self.nhyper)
            
            if i == 0:
                B = pif
            else:
                B = torch.cat((B, pif), dim=1)
                
        # Solve linear system
        all_dzdz = torch.einsum('xij,xjk->xki', torch.linalg.inv(B), torch.complex(torch.tensor(-1.), torch.tensor(0.)) * dz_hyper)
        
        # Fill at the right positions
        for i in range(self.nhyper):
            fixed_coord_idx = dQdz_indices[:, i]
            for b in range(batch_size):
                for f in range(self.nfold):
                    pullbacks[b, f, fixed_coord_idx[b]] = all_dzdz[b, f, i]
                    
        return pullbacks

    def _find_max_dQ_coords(self, points):
        r"""Finds in each hypersurface the coordinates for which |dQ/dzj|
        is largest.

        Args:
            points (torch.tensor([bSize, 2*ncoords], float32)): Points.

        Returns:
            torch.tensor([bSize, nhyper], int64): max(dQ/dz) index per hyper.
        """
        cpoints = torch.complex(points[:, :self.ncoords], points[:, self.ncoords:])
        available_mask = self._get_inv_one_mask(points).to(torch.complex64)
        
        indices = []
        for i in range(self.nhyper):
            dQdz = self._compute_dQdz(cpoints, i)
            if i == 0:
                max_indices = torch.argmax(torch.abs(dQdz * available_mask), dim=-1)
                indices = max_indices.unsqueeze(1)
            else:
                max_dq = torch.argmax(torch.abs(dQdz * available_mask), dim=-1)
                indices = torch.cat([indices, max_dq.unsqueeze(1)], dim=-1)
            available_mask -= torch.nn.functional.one_hot(indices[:, i], self.ncoords).to(torch.complex64)
        
        return indices

    def _compute_dQdz(self, points, k):
        r"""Computes dQdz at each point.

        Args:
            points (torch.tensor([bSize, ncoords], complex)): vector of coordinates
            k (int): k-th hypersurface

        Returns:
            torch.tensor([bSize, ncoords], complex): dQdz at each point.
        """
        dqdzb_k = self.BASIS['DQDZB'+str(k)].to(self.device) if isinstance(self.BASIS['DQDZB'+str(k)], torch.Tensor) else torch.tensor(self.BASIS['DQDZB'+str(k)], device=self.device)
        dqdzf_k = self.BASIS['DQDZF'+str(k)].to(self.device) if isinstance(self.BASIS['DQDZF'+str(k)], torch.Tensor) else torch.tensor(self.BASIS['DQDZF'+str(k)], device=self.device)
        
        p_exp = points.unsqueeze(1).unsqueeze(1)
        dQdz = torch.pow(p_exp, dqdzb_k)
        dQdz = torch.prod(dQdz, dim=-1)
        dQdz = dqdzf_k * dQdz
        dQdz = torch.sum(dQdz, dim=-1)
        return dQdz

    def _get_inv_one_mask(self, points):
        r"""Computes mask with True when z_i != 1+0.j."""
        cpoints = torch.complex(points[:, :self.ncoords], points[:, self.ncoords:])
        return ~torch.isclose(cpoints, torch.tensor(1.0, dtype=torch.complex64, device=points.device))

    def _fubini_study_n_metrics(self, points, n=None, t=torch.complex(torch.tensor(1.), torch.tensor(0.))):
        r"""Computes the Fubini-Study metric on a single projective
        ambient space factor specified by n.

        Args:
            points (torch.tensor([bSize, ncoords], complex64)): Coordinates of
                the n-th projective space.
            n (int, optional): Degree of P**n. Defaults to None(=self.ncoords).
            t (torch.complex, optional): Volume factor. Defaults to 1+0j.

        Returns:
            torch.tensor([bsize, ncoords, ncoords], complex64): 
                FS-metric in the ambient space coordinates.
        """
        if n is None:
            n = self.ncoords
        point_square = torch.sum(torch.abs(points)**2, dim=-1)
        point_square = point_square.to(torch.complex64)
        point_diag = torch.einsum('x,ij->xij', point_square,
                                torch.eye(n, dtype=torch.complex64, device=self.device))
        outer = torch.einsum('xi,xj->xij', torch.conj(points), points)
        outer = outer.to(torch.complex64)
        gFS = torch.einsum('xij,x->xij', (point_diag - outer), point_square**-2)
        return gFS*t/self.pi

    def _fubini_study_n_potentials(self, points, n=None, t=torch.complex(torch.tensor(1.), torch.tensor(0.))):
        r"""Computes the Fubini-Study potential on a single projective
        ambient space factor specified by n.

        Args:
            points (torch.tensor([bSize, ncoords], complex64)): Coordinates.
            n (int, optional): Degree of P**n. Defaults to None(=self.ncoords).
            t (torch.complex, optional): Volume factor. Defaults to 1+0j.

        Returns:
            torch.tensor([bsize], float32): FS-potential.
        """
        if n is None:
            n = self.ncoords
        point_square = torch.sum(torch.abs(points)**2, dim=-1)
        return torch.real(t/self.pi) * torch.real(torch.log(point_square))

    def _indices_to_mask(self, indices):
        r"""Takes indices ([bSize,nTrue], int) and creates a faux coordinates
        mask. NOTE: the output is *not* of boolean type.
        
        Args:
            indices (torch.tensor): Indices tensor
            
        Returns:
            torch.tensor: Coordinate mask
        """
        mask = torch.nn.functional.one_hot(indices, num_classes=self.ncoords)
        mask = torch.sum(mask, dim=1)
        return mask

    def _generate_patches(self, args):
        r"""Generates possible patch transitions for the patches specified in
        args.
        
        Args:
            args (torch.tensor): Arguments containing fixed and original indices
            
        Returns:
            torch.tensor: All patches
        """
        # Convert to numpy for easier manipulation, then back to torch
        args_np = args.cpu().numpy()
        fixed = args_np[:self.nhyper]
        original = args_np[self.nhyper:]
        
        # Create inverse fixed mask
        fixed_one_hot = np.zeros(self.ncoords, dtype=bool)
        for f in fixed:
            fixed_one_hot[f] = True
        inv_fixed_mask = ~fixed_one_hot
        
        # Get projection indices for fixed coordinates
        proj_indices_np = self._proj_indices.cpu().numpy()
        fixed_proj = np.zeros(self.nProjective, dtype=np.int64)
        for f in fixed:
            fixed_proj[proj_indices_np[f]] += 1
        
        # Calculate splits
        degrees_np = self.degrees.cpu().numpy()
        splits = degrees_np - fixed_proj
        
        # Get all available coordinates
        all_coords = np.arange(self.ncoords)[inv_fixed_mask]
        
        # Generate products manually since we can't use tf.split equivalent easily
        # This is a simplified version - for full implementation, need proper meshgrid
        if len(splits) == 1:
            all_patches = all_coords.reshape(-1, 1)
        else:
            # For multi-projective spaces, this would need more sophisticated logic
            # For now, implement a basic version that works for simple cases
            from itertools import product
            coord_groups = []
            start_idx = 0
            for i, split_size in enumerate(splits):
                if split_size > 0:
                    end_idx = start_idx + split_size
                    if end_idx <= len(all_coords):
                        coord_groups.append(all_coords[start_idx:end_idx])
                        start_idx = end_idx
                    else:
                        coord_groups.append(all_coords[start_idx:])
                else:
                    coord_groups.append([])
            
            if len(coord_groups) > 0 and all(len(g) > 0 for g in coord_groups):
                all_patches = np.array(list(product(*coord_groups)))
            else:
                all_patches = np.array([[original[0]] * self.nProjective])
        
        # Ensure we have the right number of transitions
        if len(all_patches) < self.nTransitions:
            # Pad with same-to-same transitions
            padding_needed = self.nTransitions - len(all_patches)
            same_patches = np.tile(original, (padding_needed, 1))
            all_patches = np.vstack([all_patches, same_patches])
        elif len(all_patches) > self.nTransitions:
            all_patches = all_patches[:self.nTransitions]
        
        return torch.tensor(all_patches, dtype=torch.int64, device=self.device)

    def _generate_patches_vec(self, combined):
        r"""Vectorized version of patch generation.
        
        Args:
            combined (torch.tensor): Combined indices
            
        Returns:
            torch.tensor: Generated patches
        """
        # For each row in combined, generate patches
        batch_size = combined.shape[0]
        all_patches = []
        
        for i in range(batch_size):
            patches = self._generate_patches(combined[i])
            all_patches.append(patches)
        
        return torch.stack(all_patches, dim=0)

    def _get_patch_coordinates(self, points, patch_mask):
        r"""Transforms the coordinates, such that they are in the patch
        given in patch_mask.
        
        Args:
            points (torch.tensor): Complex coordinate points
            patch_mask (torch.tensor): Boolean mask for patch
            
        Returns:
            torch.tensor: Transformed coordinates
        """
        # Get normalization factors from the mask
        norm_factors = points[patch_mask.bool()]
        norm_factors = norm_factors.view(-1, self.nProjective)
        
        # Create full normalization tensor
        full_norm = torch.ones_like(points)
        for i in range(self.nProjective):
            degrees_tensor = torch.ones(self.degrees[i], dtype=torch.complex64, device=self.device)
            tmp_norm = torch.einsum('i,x->xi', degrees_tensor, norm_factors[:, i])
            if i == 0:
                full_norm_combined = tmp_norm
            else:
                full_norm_combined = torch.cat((full_norm_combined, tmp_norm), dim=-1)
        
        return points / full_norm_combined

    def get_transition_matrix(self, points, i_mask, j_mask, fixed):
        r"""Computes transition matrix between patch i and j 
        for each point in points where fixed is the coordinate,
        which is being eliminated.

        This is a faithful translation of the TensorFlow implementation.

        Args:
            points (torch.tensor([bSize, ncoords], complex64)): Complex points.
            i_mask (torch.tensor([bSize, ncoords], bool)): Mask of pi-indices.
            j_mask (torch.tensor([bSize, ncoords], bool)): Mask of pj-indices.
            fixed (torch.tensor([bSize, 1], int64)): Elimination indices.

        Returns:
            torch.tensor([bSize, nfold, nfold], complex64): T_ij on the CY.
        """
        device = points.device
        
        # Find same and different patches - faithful TF translation
        same_patch_bool = torch.all(i_mask == j_mask, dim=-1)
        same_patch = torch.where(same_patch_bool)[0]
        diff_patch = torch.where(~same_patch_bool)[0]
        
        n_p = fixed.shape[0]
        n_p_red = diff_patch.shape[0]
        
        if n_p_red == 0:
            # All same patches, return identity
            return torch.eye(self.nfold, dtype=torch.complex64, device=device).unsqueeze(0).repeat(n_p, 1, 1)
            
        # Reduce non-trivial cases - faithful TF translation
        i_mask_red = i_mask[diff_patch]
        j_mask_red = j_mask[diff_patch]
        fixed_red = fixed[diff_patch]
        points_red = points[diff_patch]
        
        # Get p2 - faithful TF translation
        j_where = torch.where(j_mask_red)
        p2 = j_where[1].view(-1, self.nProjective)
        
        # g1 mask - faithful TF translation
        g1_mask = torch.sum(torch.nn.functional.one_hot(fixed_red, self.ncoords).float(), dim=-2)
        g1_mask = g1_mask + i_mask_red.float()
        g1_mask = ~(g1_mask.bool())
        g1_where = torch.where(g1_mask)
        g1_i = g1_where[1].view(-1, self.nfold)
        
        # g2 mask - faithful TF translation  
        g2_mask = torch.sum(torch.nn.functional.one_hot(fixed_red, self.ncoords).float(), dim=-2)
        g2_mask = g2_mask + j_mask_red.float()
        g2_mask = ~(g2_mask.bool())
        g2_where = torch.where(g2_mask)
        g2_i = g2_where[1].view(-1, self.nfold)
        
        # proj indices - faithful TF translation
        proj_indices = self._proj_indices.repeat(n_p_red, 1)
        g1_proj = proj_indices[g1_mask].view(-1, self.nfold)
        
        # ratios - faithful TF translation
        # TF does: tf.boolean_mask(points_red, i_mask_red) / tf.boolean_mask(points_red, j_mask_red)
        # where points_red are complex coordinates
        i_coords = torch.masked_select(points_red, i_mask_red).view(-1, self.nProjective)
        j_coords = torch.masked_select(points_red, j_mask_red).view(-1, self.nProjective)
        ratios = i_coords / j_coords
        
        # Initialize reduced transition matrix
        tij_red = torch.zeros((n_p_red, self.nfold, self.nfold), dtype=torch.complex64, device=device)
        
        # Fill mixed ratio elements - faithful TF translation
        for j in range(self.nProjective):
            t_pos = torch.einsum('xi,xj->xij',
                               (g1_i == p2[:, j:j+1]).int(),
                               (g1_proj == j).int())
            t_indices = torch.where(t_pos.bool())
            
            if t_indices[0].shape[0] > 0:
                # Get numerator indices - faithful TF translation
                num_indices = g2_i[t_indices[0], t_indices[2]]
                num_point_indices = torch.stack([t_indices[0], num_indices], dim=1)
                # Get complex coordinates directly
                num_tpos = points_red[num_point_indices[:, 0], num_point_indices[:, 1]]
                
                # Get ratios - faithful TF translation
                ratio_indices = t_indices[0]
                ratio_tpos = ratios[ratio_indices, j]
                
                # Get denominator - faithful TF translation
                denom_indices = torch.stack([torch.arange(n_p_red, device=device), p2[:, j]], dim=1)
                # Get complex denominator directly
                denom_tpos = points_red[denom_indices[:, 0], denom_indices[:, 1]]
                denom_tpos = denom_tpos[ratio_indices]
                
                # Compute values - faithful TF translation
                t_values = -1.0 * num_tpos * ratio_tpos / denom_tpos
                
                # Update - faithful TF translation
                tij_red[t_indices[0], t_indices[1], t_indices[2]] = t_values
        
        # Fill single ratio elements - faithful TF translation
        g1_i_reshaped = g1_i.view(-1, 1, self.nfold)
        g2_i_reshaped = g2_i.view(-1, self.nfold, 1)
        c_pos_tensor = torch.where(g1_i_reshaped == g2_i_reshaped)
        
        if c_pos_tensor[0].shape[0] > 0:
            c_indices = g1_proj[c_pos_tensor[0], c_pos_tensor[1]]
            c_ratio_indices = torch.stack([c_pos_tensor[0], c_indices], dim=1)
            c_values = ratios[c_ratio_indices[:, 0], c_ratio_indices[:, 1]]
            
            # Need to switch cols - faithful TF translation
            tij_red[c_pos_tensor[0], c_pos_tensor[2], c_pos_tensor[1]] = c_values
        
        # Fill full matrix - faithful TF translation
        tij_eye = torch.eye(self.nfold, dtype=torch.complex64, device=device).unsqueeze(0).repeat(n_p - n_p_red, 1, 1)
        tij_all = torch.zeros((n_p, self.nfold, self.nfold), dtype=torch.complex64, device=device)
        
        if n_p_red > 0:
            tij_all[diff_patch] = tij_red
        if same_patch.shape[0] > 0:
            tij_all[same_patch] = tij_eye[:same_patch.shape[0]]
            
        return tij_all

    def transition_loss_matrices(self, gj, gi, Tij):
        r"""Computes transition loss matrix between metric
        in patches i and j with transition matrix Tij.

        Args:
            gj (torch.tensor([bSize, nfold, nfold], complex64)):
                Metric in patch j.
            gi (torch.tensor([bSize, nfold, nfold], complex64)):
                Metric in patch i.
            Tij (torch.tensor([bSize, nfold, nfold], complex64)):
                Transition matrix from patch i to patch j.

        Returns:
            torch.tensor([bSize, nfold, nfold], complex64): 
                g_j - T^{ij} g_i T^{ij,†}
        """
        # Compute T^{ij} g_i T^{ij,†}
        # Note: torch.conj() is equivalent to tf.math.conj()
        # torch.transpose() with conjugate=True is equivalent to tf.transpose(..., conjugate=True)
        Tij_conj_transpose = torch.conj(Tij.transpose(-2, -1))
        transformed_gi = torch.einsum('xij,xjk,xkl->xil', Tij, gi, Tij_conj_transpose)
        
        return gj - transformed_gi

    def compute_transition_loss(self, points):
        r"""Computes transition loss at each point.

        .. math::

            \mathcal{L} = \frac{1}{d} \sum_{k,j} 
                ||g^k - T_{jk} \cdot g^j T^\dagger_{jk}||_n

        Args:
            points (torch.tensor([bSize, 2*ncoords], float32)): Points.

        Returns:
            torch.tensor([bSize], float32): Transition loss at each point.
        """
        device = points.device
        batch_size = points.shape[0]
        
        # Faithful translation of TensorFlow implementation
        inv_one_mask = self._get_inv_one_mask(points)
        patch_indices = torch.where(~inv_one_mask)[1]
        patch_indices = patch_indices.view(-1, self.nProjective)
        current_patch_mask = self._indices_to_mask(patch_indices)
        
        cpoints = torch.complex(points[:, :self.ncoords], points[:, self.ncoords:])
        fixed = self._find_max_dQ_coords(points)
        
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
        exp_points = cpoints.repeat_interleave(self.nTransitions, dim=0)
        patch_points = self._get_patch_coordinates(
            exp_points, other_patch_mask.bool())
        
        # Ensure fixed is a tensor before calling repeat_interleave
        if not isinstance(fixed, torch.Tensor):
            fixed = torch.tensor(fixed, device=device, dtype=torch.int64)
        fixed_expanded = fixed.repeat_interleave(self.nTransitions, dim=0).view(-1, self.nhyper)
        
        real_patch_points = torch.cat(
            (patch_points.real, patch_points.imag), dim=-1)
        
        gj = self(real_patch_points, j_elim=fixed_expanded)
        
        # NOTE: We will compute this twice.
        # TODO: disentangle this to save one computation?
        gi = self(points).repeat_interleave(self.nTransitions, dim=0)
        current_patch_mask_expanded = current_patch_mask.repeat_interleave(self.nTransitions, dim=0)
        
        Tij = self.get_transition_matrix(
            patch_points, other_patch_mask.bool(), current_patch_mask_expanded.bool(), fixed_expanded
            )
        
        all_t_loss = torch.abs(self.transition_loss_matrices(gj, gi, Tij))
        all_t_loss = torch.sum(all_t_loss**self.n[2], dim=[1, 2])
        
        # This should now be nTransitions 
        all_t_loss = all_t_loss.view(-1, self.nTransitions)
        all_t_loss = torch.sum(all_t_loss, dim=-1)
        
        return all_t_loss / (self.nTransitions * self.nfold**2)
