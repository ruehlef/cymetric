"""
A bunch of custom metrics for PyTorch models.
These are designed to work with the custom training loop.
"""

import torch
import torch.nn as nn
import numpy as np


class BaseMetric:
    """Base class for all metrics."""
    
    def __init__(self, name='base_metric'):
        self.name = name
        self.reset_state()
    
    def update_state(self, values, sample_weight=None):
        raise NotImplementedError
    
    def result(self):
        raise NotImplementedError
    
    def reset_state(self):
        raise NotImplementedError


class SigmaLoss(BaseMetric):
    """Sigma loss metric for tracking during training."""
    
    def __init__(self, name='sigma_loss'):
        super(SigmaLoss, self).__init__(name=name)
        
    def update_state(self, values, sample_weight=None):
        """
        Args:
            values: dict containing loss values
            sample_weight: sample weights for the validation set (Default: None)
        """
        if isinstance(values, dict) and 'sigma_loss' in values:
            loss = values['sigma_loss']
        else:
            loss = values
            
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().cpu().numpy()
        
        if sample_weight is not None:
            if isinstance(sample_weight, torch.Tensor):
                sample_weight = sample_weight.detach().cpu().numpy()
            loss = loss * sample_weight
            
        new_value = (np.mean(loss) - self.sigma_loss) / (self.count + 1)
        self.sigma_loss += new_value
        self.count += 1

    def result(self):
        return self.sigma_loss

    def reset_state(self):
        self.sigma_loss = 0.0
        self.count = 0


class KaehlerLoss(BaseMetric):
    """Kaehler loss metric for tracking during training."""
    
    def __init__(self, name='kaehler_loss'):
        super(KaehlerLoss, self).__init__(name=name)
        
    def update_state(self, values, sample_weight=None):
        """
        Args:
            values: dict containing loss values or tensor
            sample_weight: sample weights for the validation set (Default: None)
        """
        if isinstance(values, dict) and 'kaehler_loss' in values:
            loss = values['kaehler_loss']
        else:
            loss = values
            
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().cpu().numpy()
        
        if sample_weight is not None:
            if isinstance(sample_weight, torch.Tensor):
                sample_weight = sample_weight.detach().cpu().numpy()
            loss = loss * sample_weight
            
        new_value = (np.mean(loss) - self.kaehler_loss) / (self.count + 1)
        self.kaehler_loss += new_value
        self.count += 1

    def result(self):
        return self.kaehler_loss

    def reset_state(self):
        self.kaehler_loss = 0.0
        self.count = 0


class TransitionLoss(BaseMetric):
    """Transition loss metric for tracking during training."""
    
    def __init__(self, name='transition_loss'):
        super(TransitionLoss, self).__init__(name=name)
        
    def update_state(self, values, sample_weight=None):
        """
        Args:
            values: dict containing loss values or tensor
            sample_weight: sample weights for the validation set (Default: None)
        """
        if isinstance(values, dict) and 'transition_loss' in values:
            loss = values['transition_loss']
        else:
            loss = values
            
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().cpu().numpy()
        
        if sample_weight is not None:
            if isinstance(sample_weight, torch.Tensor):
                sample_weight = sample_weight.detach().cpu().numpy()
            loss = loss * sample_weight
            
        new_value = (np.mean(loss) - self.transition_loss) / (self.count + 1)
        self.transition_loss += new_value
        self.count += 1

    def result(self):
        return self.transition_loss

    def reset_state(self):
        self.transition_loss = 0.0
        self.count = 0


class RicciLoss(BaseMetric):
    """Ricci loss metric for tracking during training."""
    
    def __init__(self, name='ricci_loss'):
        super(RicciLoss, self).__init__(name=name)
        
    def update_state(self, values, sample_weight=None):
        """
        Args:
            values: dict containing loss values or tensor
            sample_weight: sample weights for the validation set (Default: None)
        """
        if isinstance(values, dict) and 'ricci_loss' in values:
            loss = values['ricci_loss']
        else:
            loss = values
            
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().cpu().numpy()
        
        if sample_weight is not None:
            if isinstance(sample_weight, torch.Tensor):
                sample_weight = sample_weight.detach().cpu().numpy()
            loss = loss * sample_weight
            
        new_value = (np.mean(loss) - self.ricci_loss) / (self.count + 1)
        self.ricci_loss += new_value
        self.count += 1

    def result(self):
        return self.ricci_loss

    def reset_state(self):
        self.ricci_loss = 0.0
        self.count = 0


class VolkLoss(BaseMetric):
    """Volume (Volk) loss metric for tracking during training."""
    
    def __init__(self, name='volk_loss'):
        super(VolkLoss, self).__init__(name=name)
        
    def update_state(self, values, sample_weight=None):
        """
        Args:
            values: dict containing loss values or tensor
            sample_weight: sample weights for the validation set (Default: None)
        """
        if isinstance(values, dict) and 'volk_loss' in values:
            loss = values['volk_loss']
        else:
            loss = values
            
        if isinstance(loss, torch.Tensor):
            loss = loss.detach().cpu().numpy()
        
        if sample_weight is not None:
            if isinstance(sample_weight, torch.Tensor):
                sample_weight = sample_weight.detach().cpu().numpy()
            loss = loss * sample_weight
            
        new_value = (np.mean(loss) - self.volk_loss) / (self.count + 1)
        self.volk_loss += new_value
        self.count += 1

    def result(self):
        return self.volk_loss

    def reset_state(self):
        self.volk_loss = 0.0
        self.count = 0


class TotalLoss(BaseMetric):
    """Total loss metric combining all loss components."""
    
    def __init__(self, name='total_loss'):
        super(TotalLoss, self).__init__(name=name)
        
    def update_state(self, values, sample_weight=None):
        """
        Args:
            values: dict containing loss values or tensor
            sample_weight: sample weights for the validation set (Default: None)
        """
        if isinstance(values, dict):
            # Sum all loss components
            loss = 0.0
            for key, value in values.items():
                if 'loss' in key.lower():
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().numpy()
                    loss += np.mean(value)
        else:
            loss = values
            if isinstance(loss, torch.Tensor):
                loss = loss.detach().cpu().numpy()
        
        if sample_weight is not None:
            if isinstance(sample_weight, torch.Tensor):
                sample_weight = sample_weight.detach().cpu().numpy()
            loss = loss * sample_weight
            
        new_value = (np.mean(loss) - self.total_loss) / (self.count + 1)
        self.total_loss += new_value
        self.count += 1

    def result(self):
        return self.total_loss

    def reset_state(self):
        self.total_loss = 0.0
        self.count = 0
