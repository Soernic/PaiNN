import os
import torch
from torch.utils.data import Subset
from torch_geometric.datasets import QM9
import os.path as osp
import pickle


class TransformedSubset(torch.utils.data.Dataset):
    """
    Dataset wrapper for applying transforms to a subset
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        item = self.subset[idx]
        if self.transform: 
            item = self.transform(item)
        return item
    
    def __len__(self):
        return len(self.subset)
    
def get_qm9_datasets(normalise=True):
    """
    Load preprocessed (almost) fully connected datasets with consistent splits and normalisation
    """
    
    # Same paths as data_setup
    data_dir = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'data')
    processed_dir = osp.join(data_dir, 'QM9_fully_connected')
    splits_dir = osp.join(data_dir, 'splits')
    norm_dir = osp.join(data_dir, 'normalisers')    

    splits_file = osp.join(splits_dir, 'qm9_random_split_indices.pkl')
    normaliser_file = osp.join(norm_dir, 'qm9_target_normaliser.pt')    

    # Check if necessary files exist
    if not osp.exists(processed_dir) or not osp.exists(splits_file) or not osp.exists(normaliser_file):
        raise FileNotFoundError(
            "Dataset files missing. Please run data_processing/data_setup.py first"
        )    
    
    # Load the processed dataset
    dataset = QM9(
        root=processed_dir,
        pre_transform=None,
        transform=None,
        force_reload=False
    )    

    # .. and split it up
    with open(splits_file, 'rb') as f: 
        split_data = pickle.load(f)

    train_dataset_raw = Subset(dataset, split_data['train'])
    val_dataset_raw = Subset(dataset, split_data['val'])
    test_dataset_raw = Subset(dataset, split_data['test'])

    checkpoint = torch.load(normaliser_file)
    normaliser = checkpoint['normaliser'] if normalise else None

    if normalise: 
        train_dataset = TransformedSubset(train_dataset_raw, normaliser)
        val_dataset = TransformedSubset(val_dataset_raw, normaliser)
        test_dataset = TransformedSubset(test_dataset_raw, normaliser)

    else:
        train_dataset = train_dataset_raw
        val_dataset = val_dataset_raw
        test_dataset = test_dataset_raw

    return train_dataset, val_dataset, test_dataset, normaliser



def ensure_dataset_setup():
    """
    Make sure QM9 is setup so that
    - the graphs are fully connected
    - there is a normaliser file somewhere with statistics to normalise for train set
    - we have a way to denormalise as well
    """
    data_dir = osp.join(osp.dirname(osp.abspath(__file__)), 'data')
    processed_dir = osp.join(data_dir, 'QM9_fully_connected')
    splits_file = osp.join(data_dir, 'splits', 'qm9_random_split_indices.pkl')
    normaliser_file = osp.join(data_dir, 'normalisers', 'qm9_target_normaliser.pt')
    
    if not (osp.exists(processed_dir) and osp.exists(splits_file) and osp.exists(normaliser_file)):
        print("Dataset not set up. Running data setup...")
        from data_processing.data_setup import setup_qm9_dataset
        setup_qm9_dataset()
    else:
        print("Dataset already set up.")


