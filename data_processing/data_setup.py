import os
import torch
from torch.utils.data import random_split
from torch_geometric.datasets import QM9
import os.path as osp
import pickle

from .transforms import FullyConnectedTransform, TargetNormaliser

def setup_qm9_dataset(
        force_reprocess=False,
        cutoff=5.0,
        seed=42
    ):
    """
    Process QM9 dataset creating fully connected molecules up to `cutoff` Å in euclidean distance from atom coordinate. 
    Split into train/val/test and compute normalisation statistics
    """

    data_dir = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'data')
    processed_dir = osp.join(data_dir, 'QM9_fully_connected')
    splits_dir = osp.join(data_dir, 'splits')
    norm_dir = osp.join(data_dir, 'normalisers')    

    # Create directories if they don't exist
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(norm_dir, exist_ok=True)
    
    # Paths for the splits used in train/test/val for future reference so we don't have data leakage etc., and the normaliser instance itself
    splits_file = osp.join(splits_dir, 'qm9_random_split_indices.pkl')
    normaliser_file = osp.join(norm_dir, 'qm9_target_normaliser.pt')    

    # if needed, process the dataset
    if force_reprocess or not osp.exists(processed_dir): # if you specifically tell it, or the file doesn't exist
        print(f'Processing QM9 dataset with fully connected molecules')

        torch.manual_seed(seed)

        fully_connected_dataset = QM9(
            root=processed_dir,
            pre_transform=FullyConnectedTransform(cutoff=cutoff),
            transform=None, #nothing at this stage
            force_reload=True # this is a requirement in order to modify data structure significantly it seems
        )

        print(f'Dataset processed. Total samples: {len(fully_connected_dataset)}')

    else:
        fully_connected_dataset = QM9(
            root=processed_dir,
            pre_transform=None,
            transform=None,
            force_reload=False
        )
        print(f'Loaded processed dataset. Total samples: {len(fully_connected_dataset)}')

    # if we're doing it intentionally or for the first time
    if force_reprocess or not osp.exists(splits_file):
        print(f'Creating dataset splits..')

        torch.manual_seed(seed) # important for data leakage in random_split

        total_size = len(fully_connected_dataset)
        train_size = int(0.8 * total_size) # ~80%
        val_size = int(0.1 * total_size) # ~10%
        test_size = total_size - train_size - val_size # whatever remains

        train_dataset, val_dataset, test_dataset = random_split(
            fully_connected_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed) # so it is consistent across runs
        )

        with open(splits_file, 'wb') as f:
            pickle.dump({
                'train': train_dataset.indices,
                'val': val_dataset.indices,
                'test': test_dataset.indices,
                'seed': seed
            }, f)

        print(f'Splits created and saved: {train_size}/{val_size}/{test_size}')

    else:
        with open(splits_file, 'rb') as f:
            split_data = pickle.load(f)
        print(f'Loaded existing dataset splits.')

    # If needed or first time around, compute statistics
    if force_reprocess or not osp.exists(normaliser_file):
        print(f'Copmuting normalisation statistics..')
        with open(splits_file, 'rb') as f:
            split_data = pickle.load(f)

        # get only train data
        from torch.utils.data import Subset
        train_dataset = Subset(fully_connected_dataset, split_data['train'])

        target_normaliser = TargetNormaliser()
        target_normaliser.fit(train_dataset) # fit it to train data

        torch.save({
            'normaliser': target_normaliser,
            'seed': seed,
            'split': 'random_split_80_10_10'
        }, normaliser_file)

        print('Normalisation statistics computed and saved')

    else:
        print(f'Loaded existing normalisation statistics')

    print(f'QM9 dataset setup complete.\n')



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Set up QM9 dataset for training')
    parser.add_argument('--force_reprocess', action='store_true', help='Force reprocessing of dataset')
    parser.add_argument('--cutoff', type=float, default=5.0, help='Cutoff distance for edges')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    setup_qm9_dataset(force_reprocess=args.force_reprocess, cutoff=args.cutoff, seed=args.seed)