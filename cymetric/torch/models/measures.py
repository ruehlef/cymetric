""" 
Various error measures for neural nets 
representing (ricci flat) kaehler metrics.
"""
import torch
import sys


def sigma_measure(model, points, y_true):
    r"""We compute the Monge Ampere equation

    .. math::

        \sigma = 1 / (\text{Vol}_\text{cy} n_p) \sum_i |1 - (\det(g) \text{Vol}_\text{cy})/(|\Omega|^2 \text{Vol}_\text{K})|

    Args:
        model (torch.nn.Module): Any (sub-)class of FSModel.
        points (torch.tensor([n_p,2*ncoord], float32)): NN input
        y_true (torch.tensor([n_p,2], float32)): (weights,  Omega \wedge \bar(\Omega)|_p)

    Returns:
        torch.tensor: sigma measure
    """
    g = model(points)
    weights = y_true[:, -2]
    omega = y_true[:, -1]
    # use gamma series
    det = torch.real(torch.linalg.det(g))  # * factorial / (2**nfold)
    det_over_omega = det / omega
    volume_cy = torch.mean(weights, dim=-1)
    vol_k = torch.mean(det_over_omega * weights, dim=-1)
    ratio = volume_cy / vol_k
    sigma_integrand = torch.abs(torch.ones_like(det_over_omega) - det_over_omega * ratio) * weights
    sigma = torch.mean(sigma_integrand) / volume_cy
    return sigma


def ricci_measure(model, points, y_true, pullbacks=None, verbose=0):
    r"""Computes the Ricci measure for a kaehler metric.

    .. math::

        ||R|| \equiv \frac{\text{Vol}_K^{\frac{1}{\text{nfold}}}}{\text{Vol}_{\text{CY}}}
            \int_X d\text{Vol}_K |R|

    Args:
        model (torch.nn.Module): Any (sub-)class of FSModel.
        points (torch.tensor([n_p,2*ncoord], float32)): NN input
        y_true (torch.tensor([n_p,2], float32)): (weights, 
            Omega \wedge \bar(\Omega)|_p)
        pullbacks (torch.tensor([n_p,nfold,ncoord], complex64)): Pullback tensor
            Defaults to None. Then gets computed.
        verbose (int, optional): if > 0 prints some intermediate
            infos. Defaults to 0.

    Returns:
        torch.tensor: Ricci measure
    """
    nfold = float(model.nfold)
    ncoords = model.ncoords
    weights = y_true[:, -2]
    omega = y_true[:, -1]
    if pullbacks is None:
        pullbacks = model.pullbacks(points)
    # factorial = torch.exp(torch.lgamma(torch.tensor(nfold+1)))
    x_vars = points.clone().requires_grad_(True)
    
    # Compute prediction and determinant
    prediction = model(x_vars)
    det = torch.real(torch.linalg.det(prediction)) * 1.  # factorial / (2**nfold)
    log_det = torch.log(det)
    
    # First derivatives
    di_dg = torch.autograd.grad(
        outputs=log_det.sum(), inputs=x_vars,
        create_graph=True, retain_graph=True
    )[0]
    
    # Second derivatives (Hessian)
    didj_dg = torch.zeros(x_vars.shape[0], x_vars.shape[1], x_vars.shape[1], 
                         device=x_vars.device, dtype=x_vars.dtype)
    for i in range(x_vars.shape[1]):
        didj_dg[:, i, :] = torch.autograd.grad(
            outputs=di_dg[:, i].sum(), inputs=x_vars,
            create_graph=True, retain_graph=True
        )[0]
    
    didj_dg = didj_dg.to(torch.complex64)
    
    # add derivatives together to complex tensor
    ricci_ij = didj_dg[:, 0:ncoords, 0:ncoords]
    ricci_ij += 1j * didj_dg[:, 0:ncoords, ncoords:]
    ricci_ij -= 1j * didj_dg[:, ncoords:, 0:ncoords]
    ricci_ij += didj_dg[:, ncoords:, ncoords:]
    ricci_ij *= 0.25
    pred_inv = torch.linalg.inv(prediction)
    ricci_scalar = torch.einsum('xba,xai,xij,xbj->x', pred_inv, pullbacks,
                             ricci_ij, torch.conj(pullbacks))
    ricci_scalar = torch.abs(torch.real(ricci_scalar))
    if verbose > 0:
        print(f' - Avg ricci scalar is {torch.mean(ricci_scalar).item()}')
        if verbose > 1:
            print(f' - Max ricci scalar is {torch.max(ricci_scalar).item()}')
            print(f' - Min ricci scalar is {torch.min(ricci_scalar).item()}')

    # compute ricci measure
    det_over_omega = det / omega
    volume_cy = torch.mean(weights, dim=-1)
    vol_k = torch.mean(det_over_omega * weights, dim=-1)
    ricci_measure = (vol_k ** (1 / nfold) / volume_cy) * torch.mean(det_over_omega * ricci_scalar * weights, dim=-1)
    return ricci_measure


def ricci_scalar_fn(model, points, pullbacks=None, verbose=0, rdet=True):
    r"""Computes the Ricci scalar for a kaehler metric.

    .. math::
        R = g^{ij} \partial_i \bar{\partial}_j \log \det g

    Args:
        model (torch.nn.Module): Any (sub-)class of FSModel.
        points (torch.tensor([n_p,2*ncoord], float32)): NN input
        pullbacks (torch.tensor([n_p,nfold,ncoord], complex64)): Pullback tensor. Defaults to None. Then gets computed.
        verbose (int, optional): if > 0 prints some intermediate infos. Defaults to 0.
        rdet (bool, optional): if True also returns det. Defaults to True.
            This is a bit hacky, because the output signature changes
            but avoids recomputing the determinant after batching.

    Returns:
        torch.tensor([n_p], float32): Ricci scalar
    """
    ncoords = model.ncoords
    x_vars = points.clone().requires_grad_(True)
    if pullbacks is None:
        pullbacks = model.pullbacks(points)
    
    # Compute prediction and determinant
    prediction = model(x_vars)
    det = torch.real(torch.linalg.det(prediction)) * 1.  # factorial / (2**nfold)
    log_det = torch.log(det)
    
    # First derivatives
    di_dg = torch.autograd.grad(
        outputs=log_det.sum(), inputs=x_vars,
        create_graph=True, retain_graph=True
    )[0]
    
    # Second derivatives (Hessian)
    didj_dg = torch.zeros(x_vars.shape[0], x_vars.shape[1], x_vars.shape[1], 
                         device=x_vars.device, dtype=x_vars.dtype)
    for i in range(x_vars.shape[1]):
        didj_dg[:, i, :] = torch.autograd.grad(
            outputs=di_dg[:, i].sum(), inputs=x_vars,
            create_graph=True, retain_graph=True
        )[0]
    
    didj_dg = didj_dg.to(torch.complex64)
    
    # add derivatives together to complex tensor
    ricci_ij = didj_dg[:, 0:ncoords, 0:ncoords]
    ricci_ij += 1j * didj_dg[:, 0:ncoords, ncoords:]
    ricci_ij -= 1j * didj_dg[:, ncoords:, 0:ncoords]
    ricci_ij += didj_dg[:, ncoords:, ncoords:]
    ricci_ij *= 0.25
    pred_inv = torch.linalg.inv(prediction)
    ricci_scalar = torch.einsum('xba,xai,xij,xbj->x', pred_inv, pullbacks,
                             ricci_ij, torch.conj(pullbacks))
    ricci_scalar = torch.real(ricci_scalar)
    if verbose > 0:
        print(f' - Avg ricci scalar is {torch.mean(ricci_scalar).item()}')
        if verbose > 1:
            print(f' - Max ricci scalar is {torch.max(ricci_scalar).item()}')
            print(f' - Min ricci scalar is {torch.min(ricci_scalar).item()}')
    if rdet:
        return ricci_scalar, det
    else:
        return ricci_scalar


def sigma_measure_loss(model, points, omegas):
    r"""

    Args:
        model (torch.nn.Module): Any (sub-)class of FSModel.
        points (torch.tensor([n_p,2*ncoord], float32)): NN input
        omegas (torch.tensor([n_p], float32)): \|Omega\|^2 for the points provided

    Returns:
        torch.tensor: sigma measure
    """
    return torch.mean(model.sigma_loss(omegas, model(points)))


def kaehler_measure_loss(model, points):
    r"""Computes the Kahler loss measure.

    Args:
        model (torch.nn.Module): Any (sub-)class of FSModel.
        points (torch.tensor([n_p,2*ncoord], float32)): NN input

    Returns:
        torch.tensor: Kahler loss measure
    """
    return torch.mean(model.compute_kaehler_loss(points))


def transition_measure_loss(model, points):
    r"""Computes the Transition loss measure.

    Args:
        model (torch.nn.Module): Any (sub-)class of FSModel.
        points (torch.tensor([n_p,2*ncoord], float32)): NN input

    Returns:
        torch.tensor: Transition loss measure
    """
    return torch.mean(
        model.compute_transition_loss(points.float()))
