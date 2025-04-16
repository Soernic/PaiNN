import json
import os
import os.path as osp
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models.painn import PaiNN
from utils.data_utils import ensure_dataset_setup, get_qm9_datasets
from utils.repo_utils import ensure_repo_setup

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pdb import set_trace


class PaiNNTrainer:
    def __init__(
            self, 
            config,
    ):
        # save config for json dumping
        self.config = config

        # ... and now save all the other things for easier access
        print(config)
        self.target_idx = config['target_idx']
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        self.patience = config['patience']
        self.factor = config['factor']
        self.epochs = config['epochs']
        self.test_interval = config['test_interval']
        self.ema_alpha = config['ema_alpha']
        self.batch_size = config['batch_size']
        self.model_save_path = config['model_save_path']
        self.device = config['device']
        self.use_tensorboard = config['use_tensorboard']

        # Initialise "empty" model
        self.model = PaiNN().to(self.device)

        self.get_data() # dataloaders etc.
        self.setup_learning() # optimiser, scheduler
        self.setup_dir() # run directory and checkpoint paths
        self.setup_writer() # tensorboard stuff


    def setup_learning(self):
        self.optimiser = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimiser, mode='min', factor=self.factor, patience=self.patience)

    def setup_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = f"runs/prop_{self.target_idx}_{timestamp}"
        os.makedirs(self.run_dir, exist_ok=True)
        self.checkpoint_path = f'{self.run_dir}/best_model.pt'

    def setup_writer(self):
        if self.use_tensorboard and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(self.run_dir)
        else:
            print(f'Not using tensorboard..')


    def get_data(self):
        # Raw
        self.train_dataset, self.val_dataset, self.test_dataset, self.normaliser = get_qm9_datasets(normalise=True)

        # Dataloaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_samples = 0

        for data in tqdm(self.train_loader):
            data = data.to(self.device)
            self.optimiser.zero_grad()
            out = self.model(data)
            target = data.y[:, self.target_idx][:, None]
            loss = F.mse_loss(out, target)
            loss.backward()
            self.optimiser.step()

            batch_size = data.y.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size


        return total_loss / num_samples


    def evaluate(self, set: str):

        dataloader = None
        if set == 'validation': 
            dataloader = self.val_loader
        elif set == 'test':
            dataloader = self.test_loader
        else:
            raise RuntimeError('set not recognised. Valid set argument choices are [validation, test]')

        self.model.eval()
        mse_total = 0.0
        mae_total = 0.0
        num_samples = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                out = self.model(data)
                target = data.y[:, self.target_idx][:, None]
                
                all_preds.append(out)
                all_targets.append(target)

                mae_total += F.l1_loss(out, target).item() * data.y.size(0)
                mse_total += F.mse_loss(out, target).item() * data.y.size(0)
                num_samples += data.y.size(0)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = {
            'mae_normalised': mae_total / num_samples,
            'mse_normalised': mse_total / num_samples,
            'rmse_normalised': np.sqrt(mse_total / num_samples) 
        }
        
        if self.normaliser:
            preds_original = self.normaliser.inverse_transform(all_preds, target_idx=self.target_idx)
            targets_original = self.normaliser.inverse_transform(all_targets, target_idx=self.target_idx)

            # unnormalised mae and rmse
            metrics['mae'] = F.l1_loss(preds_original, targets_original).item()
            metrics['rmse'] = torch.sqrt(F.mse_loss(preds_original, targets_original)).item()

            metrics['preds_original'] = preds_original.cpu().numpy().flatten().tolist()
            metrics['targets_original'] = targets_original.cpu().numpy().flatten().tolist()

            # print(f'preds_original.shape: {preds_original.shape}')
            # print(f'targets_original.shape: {targets_original.shape}')

        return metrics


    def train(self):

        # Write config used for this run to file
        with open(f'{self.run_dir}/config.json', 'w') as f: 
            json.dump(self.config, f, indent=2)

        # best_val_loss = float('inf')
        best_val_ema_loss = float('inf')
        val_ema_loss = None # initialise EMA trakcing

        results = {
            'train_losses': [],
            'val_losses': [],
            'val_ema_losses': [],
            'learning_rates': [],
            'best_epoch': 0,
            'best_val_mae': float('inf'),
            'best_val_ema_mae': float('inf'),
            'best_val_mae_unnormalised': float('inf'),
            'test_metrics':  {}
        }

        print(f'Starting training for target property: {self.target_idx}')
        print(f'Model checkpoints and logs will be saved to: {self.run_dir}')
        print(f'Using EMA validation tracking with alpha:{self.ema_alpha}')

        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate('validation')
            val_loss = val_metrics['mae_normalised']

            # First epoch
            if val_ema_loss is None: 
                val_ema_loss = val_loss

            # Otherwise use ema update rule: alpha * ema + (1 - alpha) * new_val
            else: 
                val_ema_loss = self.ema_alpha * val_ema_loss + (1 - self.ema_alpha) * val_loss
            
            self.scheduler.step(val_ema_loss)
            current_lr = self.optimiser.param_groups[0]['lr']

            results['train_losses'].append(train_loss)
            results['val_losses'].append(val_loss)
            results['val_ema_losses'].append(val_ema_loss)
            results['learning_rates'].append(current_lr)

            if self.writer: 
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Loss/val_ema', val_ema_loss, epoch)
                self.writer.add_scalar('Loss/val_unnormalised', val_metrics.get('mae', 0), epoch)
                self.writer.add_scalar('LearningRate', current_lr, epoch)
        
            # Print progress
            print(f'Epoch: {epoch:3d}, Train Loss: {train_loss:.4f}, '
                f'Val MAE: {val_loss:.4f}, Val EMA: {val_ema_loss:.4f}, LR: {current_lr:.6f}')
            
            if val_ema_loss < best_val_ema_loss: 
                best_val_ema_loss = val_ema_loss
                best_val_loss = val_loss
                results['best_epoch'] = epoch
                results['best_val_mae'] = val_loss
                results['best_val_ema_mae'] = val_ema_loss
                results['best_val_mae_unnormalised'] = val_metrics.get('mae', 0)

                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f'New best model saved with validation EMA MAE: {val_ema_loss:.5f}')

                if self.normaliser: 
                    print(f'Unnormalised MAE: {val_metrics.get("mae", 0):.5f}')


            # Run test evaluation periodically
            if epoch % self.test_interval == 0 or epoch == self.epochs - 1: 
                test_metrics = self.evaluate('test')
                results['test_metrics'][f'epoch_{epoch}'] = {
                    'mae': test_metrics.get('mae', 0),
                    'mae_normalised': test_metrics['mae_normalised']
                }

                # Plot predictions
                if self.normaliser:
                    plot_path = f"{self.run_dir}/predictions_epoch_{epoch}.png"
                    self.plot_predictions(test_metrics, plot_path)
                    print(f"Test MAE: {test_metrics.get('mae', 0):.4f}")
                    print(f"Prediction plot saved to {plot_path}")

        print(f'\nLoading best model for final evaluation..')
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        final_test_metrics = self.evaluate('test')
        
        # Save final test metrics
        results["final_test_metrics"] = {
            "mae": final_test_metrics.get('mae', 0),
            "mae_normalised": final_test_metrics['mae_normalised']
        }
        results["final_test_mae"] = final_test_metrics.get('mae', 0)
        
        # Generate final prediction plot
        final_plot_path = f"{self.run_dir}/final_predictions.png"
        self.plot_predictions(final_test_metrics, final_plot_path)
        
        # Save all results
        with open(f"{self.run_dir}/results.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {k: v for k, v in results.items() 
                                if k not in ['preds_original', 'targets_original']}
            json.dump(serializable_results, f, indent=2)
        
        # Print final performance summary
        property_names = {
            0: ('Dipole moment (μ)', 'D'),
            1: ('Isotropic polarizability (α)', 'a₀³'),
            2: ('HOMO energy', 'meV'),
            3: ('LUMO energy', 'meV'),
            4: ('HOMO-LUMO gap', 'meV'),
            5: ('Electronic spatial extent ⟨R²⟩', 'a₀²'),
            6: ('ZPVE', 'meV'),
            7: ('U₀', 'meV'),
            8: ('U', 'meV'),
            9: ('H', 'meV'),
            10: ('G', 'meV'),
            11: ('Heat capacity', 'cal/mol·K')
        }
        prop_name, unit = property_names.get(self.target_idx, (f'Property {self.target_idx}', ''))
        
        print(f"\n=== Final Results for {prop_name} ===")
        print(f"Best model saved at epoch {results['best_epoch']}")
        print(f"Final Test MAE: {results['final_test_mae']:.4f} {unit}")
        print(f"Final prediction plot saved to {final_plot_path}")
        
        # Close tensorboard writer if used
        if self.writer:
            self.writer.close()
        
        return results

    def plot_predictions(self, metrics, save_path):
        property_names = {
            0: ('Dipole moment (μ)', 'D'),
            1: ('Isotropic polarizability (α)', 'a₀³'),
            2: ('HOMO energy', 'meV'),
            3: ('LUMO energy', 'meV'),
            4: ('HOMO-LUMO gap', 'meV'),
            5: ('Electronic spatial extent ⟨R²⟩', 'a₀²'),
            6: ('ZPVE', 'meV'),
            7: ('U₀', 'meV'),
            8: ('U', 'meV'),
            9: ('H', 'meV'),
            10: ('G', 'meV'),
            11: ('Heat capacity', 'cal/mol·K')
        }

        prop_name, unit = property_names.get(self.target_idx, (f'Property {self.target_idx}', ''))


        preds = metrics['preds_original']
        targets = metrics['targets_original']
        
        plt.figure(figsize=(8, 8))
        plt.scatter(targets, preds, alpha=0.5)
        plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
        plt.xlabel(f'True {prop_name} ({unit})')
        plt.ylabel(f'Predicted {prop_name} ({unit})')
        plt.title(f'Model Performance on {prop_name}\nMAE: {metrics["mae"]:.4f} {unit}')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


                
    

if __name__ == '__main__':

    ensure_repo_setup() # checkpoints, runs, and plots folders setup
    ensure_dataset_setup() # dataset is properly configured and available as it shoul dbe
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # if torch.backends.mps.is_available():
    #     device = torch.device('mps')

    # print(device)
    # Configuration for current run
    config = {
        'target_idx': 0,
        'lr': 0.0001,
        'weight_decay': 0.0001,
        'patience': 5,
        'factor': 0.5,
        'epochs': 200,
        'test_interval': 10,
        'ema_alpha': 0.9,
        'batch_size': 100,
        'model_save_path': 'best_model.pt',
        'device': device,
        'use_tensorboard': True
    }

    trainer = PaiNNTrainer(
        config=config,
    )

    trainer.train()