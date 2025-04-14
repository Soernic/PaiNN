import torch
from torch_geometric.transforms import BaseTransform

from pdb import set_trace


class FullyConnectedTransform(BaseTransform):
    """
    Converts any graph to be fully connected up to euclidean range `cutoff` in whatever unit it is in. (Å for QM9)
    """
    def __init__(self, cutoff=5.0):
        self.cutoff = cutoff

    def __call__(self, data):
        data = data.clone() # make copy
        num_nodes = data.x.size(0)

        rows, cols = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j: # xclude self-loops
                    if torch.norm(data.pos[i] - data.pos[j]) <= self.cutoff: # only if they are under cutoff length in Å
                        rows.append(i)
                        cols.append(j)

        data.edge_index = torch.tensor([rows, cols], dtype=torch.long)
        return data
    

class TargetNormaliser(BaseTransform):
    """
    Normalisation class to be saved to disc in order to be used multiple times. 
    We compute statistics once and save the class instance so that we have both normalisation and inverse transform readily avaiable.
    """
    def __init__(self, targets_to_normalise=None):
        self.targets_to_normalise = targets_to_normalise
        self.means = None
        self.stds = None
        self.fitted = False

    def fit(self, dataset):
        num_targets = dataset[0].y.shape[1] # 19 for QM9
        all_targets = torch.zeros((len(dataset), num_targets)) # to be filled in

        print(f'Computing normalisation statistics..')
        for i, data in enumerate(dataset):
            all_targets[i] = data.y
            if i % 10000 == 0:
                print(f'Processed {i}/{len(dataset)} molecules')

        self.means = all_targets.mean(dim=0)
        self.stds = all_targets.std(dim=0)

        self.stds[self.stds < 1e-8] = 1.0 # Corner case if they're too small, assume it is just 1
        
        if self.targets_to_normalise is None: # if no specification, do all of them
            self.targets_to_normalise = list(range(num_targets)) 

        self.fitted = True
        print('Statistics computed.')
        return self
    
    def __call__(self, data):
        data_normalised = data.clone() # make a copy
        
        if not self.fitted:
            raise RuntimeError('You have to fit it to the training set first')
        
        for idx in self.targets_to_normalise:
            data_normalised.y[:, idx] = (data.y[:, idx] - self.means[idx]) / self.stds[idx]

        return data_normalised
    
    def inverse_transform(self, data, target_idx=None):
        """
        data: A tensor of shape [N, k], where k is typically:
           - 1, if you’re only predicting a single target at once, or
           - len(self.targets_to_normalise), if you want to invert all columns you normalised.
        target_idx: If specified, only inverse-transform that one target index from the original 19.
                    This is handy if your model only predicts one column at a time.
        """
        data_denormalised = data.clone()

        if target_idx is not None:
            # Here data[:, 0] is your single predicted column, 
            # and self.means[target_idx], self.stds[target_idx] are the corresponding stats.
            data_denormalised[:, 0] = data[:, 0] * self.stds[target_idx] + self.means[target_idx]

        else:
            # Assume data has exactly one column per item in self.targets_to_normalise
            # so we loop over i = 0..(k-1) while idx is each target from your normalised list.
            for i, idx in enumerate(self.targets_to_normalise):
                data_denormalised[:, i] = data[:, i] * self.stds[idx] + self.means[idx]

        return data_denormalised


