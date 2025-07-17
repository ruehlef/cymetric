"""
Sigma loss function in PyTorch.
"""
import torch
import torch.nn as nn


def sigma_loss(kappa=1., nfold=3., flat=False):
    r"""MA loss.

    Args:
        kappa (float): inverse volume of the CY given by weights. Defaults to 1.
        nfold (float): dimension of the CY. Defaults to 3.
        flat (bool): True if metric is a flat tensor and has to be put into
            hermitian matrix first. Defaults to False

    Returns:
        function: MA loss function.
    """
    factorial = float(1.)
    nfold = int(nfold)
    kappa = float(kappa)
    det_factor = float(1.)

    def to_hermitian_vec(x):
        r"""Takes a tensor of length (-1,NFOLD**2) and transforms it
        into a (-1,NFOLD,NFOLD) hermitian matrix.

        Args:
            x (tensor[(-1,NFOLD**2), float]): input tensor

        Returns:
            tensor[(-1,NFOLD,NFOLD), complex]: hermitian matrix
        """
        device = x.device
        t1 = torch.complex(x, torch.zeros_like(x)).view(-1, nfold, nfold)
        
        # Upper triangular part
        up = torch.triu(t1)
        # Lower triangular part (imaginary)
        low = torch.tril(1j * t1, diagonal=-1)
        
        # Create hermitian matrix
        out = up + torch.transpose(up, -2, -1) - torch.diagonal(t1, dim1=-2, dim2=-1).diag_embed()
        return out + low + torch.conj(torch.transpose(low, -2, -1))

    def sigma_integrand_loss_flat(y_true, y_pred):
        r"""Monge-Ampere integrand loss.

        l = |1 - det(g)/ (Omega \wedge \bar{Omega})|

        Args:
            y_true (tensor[(bsize, x), float]): some tensor  
                        with last value being (Omega \wedge \bar{Omega})
            y_pred (tensor[(bsize, 9), float]): NN prediction

        Returns:
            tensor[(bsize, 1), float]: loss for each sample in batch
        """
        g = to_hermitian_vec(y_pred)
        # older tensorflow versions require shape(y_pred) == shape(y_true)
        # then just give it some tensor where omega is the last value.
        omega_squared = y_true[:, -1]
        det = torch.real(torch.det(g)) * factorial / det_factor
        return torch.abs(torch.ones_like(omega_squared) -
                      det / omega_squared / kappa)

    def sigma_integrand_loss(y_true, y_pred):
        r"""Monge-Ampere integrand loss.

        l = |1 - det(g)/ (Omega \wedge \bar{Omega})|

        Args:
            y_true (tensor[(bsize, x), float]): some tensor  
                        with last value being (Omega \wedge \bar{Omega})
            y_pred (tensor[(bsize, 3, 3), complex]): NN prediction

        Returns:
            tensor[(bsize, 1), float]: loss for each sample in batch
        """
        omega_squared = y_true[:, -1]
        det = torch.real(torch.det(y_pred)) * factorial / det_factor
        return torch.abs(torch.ones_like(omega_squared) -
                      det / omega_squared / kappa)

    if flat:
        return sigma_integrand_loss_flat
    else:
        return sigma_integrand_loss


class SigmaLoss(nn.Module):
    r"""PyTorch module version of sigma loss for easier integration."""
    
    def __init__(self, kappa=1., nfold=3., flat=False):
        super(SigmaLoss, self).__init__()
        self.loss_fn = sigma_loss(kappa, nfold, flat)
        
    def forward(self, y_pred, y_true):
        return self.loss_fn(y_true, y_pred)
