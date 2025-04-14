import torch
from torch_geometric.transforms import BaseTransform


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
    
    def inverse_transform(self, data):
        data_denormalised = data.clone() # make a copy

        for idx in self.targets_to_normalise:
            data_denormalised.y[idx] = data.y[idx]*self.stds[idx] + self.means[idx]

        return data_denormalised



