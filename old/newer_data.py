import torch
import numpy as np
import os
import json
from torch.utils.data import random_split
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, BaseTransform


class FullyConnectedTransform(BaseTransform):
    """Transform to convert a molecular graph to a fully connected graph.
    This is useful for models like PaiNN that typically expect neighbor-based or all-pairs edges."""

    def __init__(self, cutoff=5):
        self.cutoff = cutoff

    def __call__(self, data):
        data = data.clone()  # to avoid modifying the original in-place
        num_nodes = data.x.size(0)

        rows, cols = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                # Add self-loops only if requested
                if i != j or self.add_self_loops:
                    if torch.norm(data.pos[i] - data.pos[j]) <= cutoff:
                        rows.append(i)
                        cols.append(j)

        data.edge_index = torch.tensor([rows, cols], dtype=torch.long)

        return data


class TargetNormalize:
    """Transform to normalize a specific target property."""
    
    def __init__(self, target_idx):
        self.target_idx = target_idx
        self.mean = None
        self.std = None
        
    def fit(self, dataset, target_idx):
        # Extract target property from all samples
        targets = []
        for data in dataset:
            targets.append(data.y[:, target_idx].item())
        targets = np.array(targets)
        
        self.mean = float(targets.mean())
        self.std = float(targets.std())
        
        print(f"Target {target_idx} statistics: mean = {self.mean:.4f}, std = {self.std:.4f}")
        return self
        
    def __call__(self, data):
        # Normalize the target value in-place
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetNormalize transform needs to be fit first")
        data.y[:, self.target_idx] = (data.y[:, self.target_idx] - self.mean) / self.std
        return data


def get_QM9(batch_size=32, subset_until=None, target_idx=0):
    """
    Loads QM9, normalizes a specified target property, 
    and transforms each molecule to a fully connected graph. 
    Returns train/val/test DataLoaders plus the fitted target normalizer.
    """
    # 1) Create and fit the target normalizer
    target_normalizer = TargetNormalize(target_idx)
    temp_dataset = QM9(root='./data/QM9', transform=None)
    if subset_until:
        temp_dataset = temp_dataset[:subset_until]
    target_normalizer.fit(temp_dataset, target_idx)
    
    # 2) Create our composed transform: 
    #    - target normalization
    #    - fully connecting the graph
    transform = Compose([
        target_normalizer,
        FullyConnectedTransform  
    ])
    
    # 3) Load the dataset with the transform
    dataset = QM9(root='./data/QM9', pre_transform=transform, force_reload=True)
    
    # 4) Optionally select a smaller subset for speed
    if subset_until:
        dataset = dataset[:subset_until]
    
    # 5) Randomly split into train/val/test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )

    # 6) Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, target_normalizer
