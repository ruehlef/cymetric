""" 
A collection of various helper functions for PyTorch.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import copy
import tqdm


def prepare_basis(basis, dtype=torch.complex64):
    r"""Casts each entry in Basis to dtype.

    Args:
        basis (dict or str): dictionary containing geometric information or path to pickle file
        dtype (torch.dtype, optional): type to cast to. Defaults to torch.complex64.

    Returns:
        dict: with tensors rather than ndarrays
    """
    # Load from file if string path is provided
    if isinstance(basis, str):
        import pickle
        with open(basis, 'rb') as f:
            basis = pickle.load(f)
    
    new_basis = {}
    for key in basis:
        if isinstance(basis[key], np.ndarray):
            new_basis[key] = torch.tensor(basis[key], dtype=dtype)
        else:
            new_basis[key] = basis[key]
    return new_basis


# Fixed training function with proper callbacks and metrics support
def train_model(fsmodel, data, optimizer=None, epochs=50, batch_sizes=[64, 10000], verbose=1, callbacks=[], sw=False, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device
    fsmodel = fsmodel.to(device)
    
    # Convert data to tensors
    X_train = torch.tensor(data['X_train'], dtype=torch.float32, device=device)
    y_train = torch.tensor(data['y_train'], dtype=torch.float32, device=device)
    
    if sw:
        sample_weights = y_train[:, -2]
    else:
        sample_weights = None
        
    if optimizer is None:
        optimizer = torch.optim.Adam(fsmodel.parameters())
    
    training_history = {}
    
    # Initialize callback data storage
    for callback in callbacks:
        if hasattr(callback, 'on_train_begin'):
            callback.on_train_begin(logs=training_history, model=fsmodel)
    
    # Store original learning settings
    learn_kaehler = fsmodel.learn_kaehler
    learn_transition = fsmodel.learn_transition
    learn_ricci = fsmodel.learn_ricci
    learn_ricci_val = fsmodel.learn_ricci_val
        
    training_history['General loss phase'] = []
    training_history['Volume loss phase'] = []
    training_history['epochs'] = list(range(epochs))
    
    for epoch in range(epochs):
        if verbose > 0:
            print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Phase 1: Small batch size, volk loss disabled
        batch_size = batch_sizes[0]
        fsmodel.learn_kaehler = learn_kaehler
        fsmodel.learn_transition = learn_transition
        fsmodel.learn_ricci = learn_ricci
        fsmodel.learn_ricci_val = learn_ricci_val
        fsmodel.learn_volk = False
        
        # Create data loader for phase 1
        dataset1 = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, shuffle=True)
        
        epoch_loss1 = 0.0
        num_batches1 = 0
        
        fsmodel.train()
        for batch_x, batch_y in tqdm.tqdm(dataloader1):
            optimizer.zero_grad()
            loss = fsmodel.compute_loss(batch_x, batch_y, sample_weight=sample_weights)
            loss.backward()
            
            # Gradient clipping
            if hasattr(fsmodel, 'gclipping'):
                torch.nn.utils.clip_grad_norm_(fsmodel.parameters(), fsmodel.gclipping)
            
            optimizer.step()
            epoch_loss1 += loss.item()
            num_batches1 += 1
        
        avg_loss1 = epoch_loss1 / num_batches1
        training_history['General loss phase'].append(avg_loss1)
        
        # Phase 2: Large batch size, only MA and volk loss
        batch_size = min(batch_sizes[1], len(X_train))
        fsmodel.learn_kaehler = False
        fsmodel.learn_transition = False
        fsmodel.learn_ricci = False
        fsmodel.learn_ricci_val = False
        fsmodel.learn_volk = True
        
        # Create data loader for phase 2
        dataset2 = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, shuffle=True)
        
        epoch_loss2 = 0.0
        num_batches2 = 0
        
        for batch_x, batch_y in tqdm.tqdm(dataloader2):
            optimizer.zero_grad()
            loss = fsmodel.compute_loss(batch_x, batch_y, sample_weight=sample_weights)
            loss.backward()
            
            if hasattr(fsmodel, 'gclipping'):
                torch.nn.utils.clip_grad_norm_(fsmodel.parameters(), fsmodel.gclipping)
            
            optimizer.step()
            epoch_loss2 += loss.item()
            num_batches2 += 1
        
        avg_loss2 = epoch_loss2 / num_batches2
        training_history['Volume loss phase'].append(avg_loss2)
        
        if verbose > 0:
            print(f" - General loss phase: {avg_loss1:.6f}")
            print(f" - Volume loss phase:  {avg_loss2:.6f}")
        
        # Run callbacks with proper parameters
        for callback in callbacks:
            callback.on_epoch_end(epoch, logs=training_history, model=fsmodel)
    
    fsmodel.learn_kaehler = learn_kaehler
    fsmodel.learn_transition = learn_transition
    fsmodel.learn_ricci = learn_ricci
    fsmodel.learn_ricci_val = learn_ricci_val
    fsmodel.learn_volk = False

    # Call training end callbacks
    for callback in callbacks:
        callback.on_train_end(logs=training_history, model=fsmodel)
    return fsmodel, training_history


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        if self.restore_best_weights:
            self.best_weights = copy.deepcopy(model.state_dict())
