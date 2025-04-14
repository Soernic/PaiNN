from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeScale, Compose
from torch.utils.data import random_split
import torch



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
            
        import numpy as np
        targets = np.array(targets)
        self.mean = float(targets.mean())
        self.std = float(targets.std())
        print(f"Target {target_idx} statistics: mean = {self.mean:.4f}, std = {self.std:.4f}")
        return self
        
    def __call__(self, data):
        # Normalize the target value
        if self.mean is None or self.std is None:
            raise RuntimeError("TargetNormalize transform needs to be fit first")
            
        data.y[:, self.target_idx] = (data.y[:, self.target_idx] - self.mean) / self.std
        return data


def get_QM9(batch_size=32, subset_until=None, target_idx=0):
    # Create target normalizer and fit it to the data
    target_normalizer = TargetNormalize(target_idx)
    
    # Pre-load a small dataset to fit the normalizer
    temp_dataset = QM9(root='./data/QM9', transform=None)
    if subset_until: 
        temp_dataset = temp_dataset[:subset_until]
    target_normalizer.fit(temp_dataset, target_idx)
    
    # Create the composed transform
    transform = Compose([
        target_normalizer  # Apply target normalization
    ])
    
    # Load the dataset with the transform
    dataset = QM9(root='./data/QM9', transform=transform)
    
    # Select a subset for speed
    dataset = dataset[:subset_until]
    
    # # Split into train/val sets
    # train_size = int(0.8 * len(dataset))
    # val_size = int(0.1 * len(dataset))
    # train_dataset = dataset[:train_size]
    # val_dataset = dataset[train_size:train_size+val_size]
    # test_dataset = dataset[train_size+val_size:]
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )


    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, target_normalizer