import torch
import numpy as np
import os
import json
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, BaseTransform

class FullyConnectedTransform(BaseTransform):
    def __init__(self, cutoff=5.0):
        self.cutoff = cutoff
        
    def __call__(self, data):
        data = data.clone()
        num_nodes = data.x.size(0)
        
        rows, cols = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # Exclude self-loops
                    if torch.norm(data.pos[i] - data.pos[j]) <= self.cutoff:
                        rows.append(i)
                        cols.append(j)
        
        data.edge_index = torch.tensor([rows, cols], dtype=torch.long)
        return data

class TargetNormalizePreTransform(BaseTransform):
    def __init__(self, target_idx, stats_file=None):
        self.target_idx = target_idx
        self.stats_file = stats_file or f"target_{target_idx}_stats.json"
        self.mean = None
        self.std = None
        
        # Try to load existing stats
        self._load_stats()
    
    def _compute_stats(self, dataset):
        """Compute statistics on the dataset"""
        targets = []
        for data in dataset:
            targets.append(data.y[:, self.target_idx].item())
        
        self.mean = float(np.mean(targets))
        self.std = float(np.std(targets))
        
        # Save stats to file
        self._save_stats()
        print(f"Computed target {self.target_idx} statistics: mean = {self.mean:.4f}, std = {self.std:.4f}")
    
    def _save_stats(self):
        """Save normalization stats to a JSON file"""
        with open(self.stats_file, 'w') as f:
            json.dump({'mean': self.mean, 'std': self.std}, f)
    
    def _load_stats(self):
        """Try to load stats from file"""
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r') as f:
                stats = json.load(f)
                self.mean = stats['mean']
                self.std = stats['std']
                print(f"Loaded target {self.target_idx} statistics: mean = {self.mean:.4f}, std = {self.std:.4f}")
    
    def __call__(self, data):
        # Normalize the target
        data.y[:, self.target_idx] = (data.y[:, self.target_idx] - self.mean) / self.std
        
        # Store normalization parameters in the data object itself
        # This way they're accessible with each data point
        data.target_mean = self.mean
        data.target_std = self.std
        data.target_idx = self.target_idx
        
        return data
    
    def inverse_transform(self, normalized_value):
        """Convert normalized values back to original scale"""
        return normalized_value * self.std + self.mean

def get_QM9_dataset(target_idx=0, cutoff=5.0, subset_size=None, root='./data/QM9'):
    """
    Load QM9 dataset with pre-computed normalization and fully connected graphs.
    
    Returns:
        - dataset: The pre-transformed dataset
        - normalizer: The fitted normalizer with inverse_transform capability
    """

    normalizer = TargetNormalizePreTransform(target_idx)
    
    # Check if we need to compute statistics
    if normalizer.mean is None or normalizer.std is None:
        # Load dataset without transforms to compute statistics
        temp_dataset = QM9(root=root)
        if subset_size:
            temp_dataset = temp_dataset[:subset_size]
        normalizer._compute_stats(temp_dataset)
    
    pre_transform = Compose([
        normalizer,
        FullyConnectedTransform(cutoff=cutoff)
    ])
    
    # Use force_reload only if necessary (first time or explicit request)
    need_reload = not os.path.exists(os.path.join(root, 'processed', 'pre_transform.pt'))
    
    # Load the dataset with pre-transforms
    dataset = QM9(
        root=root, 
        pre_transform=pre_transform,
        force_reload=need_reload
    )
    
    if subset_size:
        dataset = dataset[:subset_size]
        
    return dataset, normalizer

# Now use the function to get a dataset and create data loaders
def prepare_data_loaders(batch_size=32, target_idx=0, subset_size=None):
    # Get the dataset and normalizer
    dataset, normalizer = get_QM9_dataset(target_idx=target_idx, subset_size=subset_size)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, normalizer


if __name__ == '__main__':
    get_QM9_dataset(target_idx=1)