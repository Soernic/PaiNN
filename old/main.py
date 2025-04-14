import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Tuple, Optional, List, Callable
import argparse

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser('Benchmark PaiNN')
    parser.add_argument('--target', type=int, help='Targets 0-19 to benchmark')

    return parser.parse_args()

# Create necessary directories
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('runs', exist_ok=True)


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    target_idx: int,
    device: torch.device
) -> float:
    """Train model for one epoch and return average loss."""
    model.train()
    total_loss = 0
    num_samples = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data)
        target = data.y[:, target_idx].unsqueeze(1)
        
        # Compute loss and update weights
        loss = F.mse_loss(out, target)  # Train with MSE
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        batch_size = data.y.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size
    
    return total_loss / num_samples


def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    target_idx: int,
    device: torch.device,
    normalizer=None
) -> Dict[str, float]:
    """Evaluate model on a dataset and return metrics."""
    model.eval()
    mse_total = 0.0
    mae_total = 0.0
    num_samples = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data)
            target = data.y[:, target_idx].unsqueeze(1)
            
            # Store predictions and targets
            all_preds.append(out)
            all_targets.append(target)
            
            # Compute losses
            mae_total += F.l1_loss(out, target).item() * data.y.size(0)
            mse_total += F.mse_loss(out, target).item() * data.y.size(0)
            num_samples += data.y.size(0)
    
    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = {
        'mae_normalized': mae_total / num_samples,
        'mse_normalized': mse_total / num_samples,
        'rmse_normalized': np.sqrt(mse_total / num_samples),
    }
    
    # Add unnormalized metrics if normalizer is provided
    if normalizer:
        # Option 1: Denormalize predictions and targets first
        preds_original = normalizer.inverse_transform(all_preds)
        targets_original = normalizer.inverse_transform(all_targets)
        
        # Calculate unnormalized metrics
        metrics['mae'] = F.l1_loss(preds_original, targets_original).item()
        metrics['rmse'] = torch.sqrt(F.mse_loss(preds_original, targets_original)).item()
        
        # Store min/max values for analysis
        metrics['pred_min'] = float(preds_original.min())
        metrics['pred_max'] = float(preds_original.max())
        metrics['target_min'] = float(targets_original.min())
        metrics['target_max'] = float(targets_original.max())
        
        # For visualization
        metrics['preds_original'] = preds_original.cpu().numpy().flatten().tolist()
        metrics['targets_original'] = targets_original.cpu().numpy().flatten().tolist()
    
    return metrics


def plot_predictions(
    metrics: Dict[str, float],
    target_idx: int,
    save_path: str = None
):
    """Generate and save a prediction scatter plot."""
    # QM9 property information
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
    
    prop_name, unit = property_names.get(target_idx, (f'Property {target_idx}', ''))
    
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


def create_run_directory(target_idx: int) -> str:
    """Create a timestamped run directory for experiment tracking."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/prop_{target_idx}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def train_and_evaluate(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    target_idx: int,
    normalizer,
    epochs: int = 200,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    patience: int = 5,
    test_interval: int = 10,
    device: torch.device = None,
    use_tensorboard: bool = True,
    ema_alpha: float = 0.9  # EMA smoothing factor
) -> Dict:
    """Main training and evaluation function with EMA validation tracking."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    # Create run directory and experiment tracking 
    run_dir = create_run_directory(target_idx)
    checkpoint_path = f"{run_dir}/best_model.pt"
    config = {
        "target_idx": target_idx,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "epochs": epochs,
        "test_interval": test_interval,
        "ema_alpha": ema_alpha,
        "time_started": datetime.now().isoformat(),
        "normalizer_mean": float(normalizer.mean),
        "normalizer_std": float(normalizer.std)
    }
    
    # Save configuration
    with open(f"{run_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Setup optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.8,
        patience=patience,
    )
    
    # Setup tensorboard if requested and available
    writer = None
    if use_tensorboard and TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(run_dir)
    
    # Training loop
    best_val_loss = float('inf')
    best_val_ema_loss = float('inf')
    val_ema_loss = None  # Initialize EMA tracking
    
    results = {
        "train_losses": [],
        "val_losses": [],
        "val_ema_losses": [],  # Add EMA tracking
        "learning_rates": [],
        "best_epoch": 0,
        "best_val_mae": float('inf'),
        "best_val_ema_mae": float('inf'),
        "best_val_mae_unnormalized": float('inf'),
        "test_metrics": {}
    }
    
    print(f"Starting training for target property {target_idx}")
    print(f"Model checkpoints and logs will be saved to: {run_dir}")
    print(f"Using EMA validation tracking with alpha={ema_alpha}")
    
    for epoch in range(epochs):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, target_idx, device)
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, target_idx, device, normalizer)
        val_loss = val_metrics['mae_normalized']  # Use MAE for validation
        
        # Calculate EMA of validation loss
        if val_ema_loss is None:
            val_ema_loss = val_loss  # First epoch
        else:
            val_ema_loss = ema_alpha * val_ema_loss + (1 - ema_alpha) * val_loss
        
        # Update learning rate based on EMA validation loss
        scheduler.step(val_ema_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        results["train_losses"].append(train_loss)
        results["val_losses"].append(val_loss)
        results["val_ema_losses"].append(val_ema_loss)
        results["learning_rates"].append(current_lr)
        
        # Log to tensorboard if available
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Loss/val_ema', val_ema_loss, epoch)
            writer.add_scalar('Loss/val_unnormalized', val_metrics.get('mae', 0), epoch)
            writer.add_scalar('LearningRate', current_lr, epoch)
        
        # Print progress
        print(f'Epoch: {epoch:3d}, Train Loss: {train_loss:.4f}, '
              f'Val MAE: {val_loss:.4f}, Val EMA: {val_ema_loss:.4f}, LR: {current_lr:.6f}')
        
        # Save best model based on EMA validation loss
        if val_ema_loss < best_val_ema_loss:
            best_val_ema_loss = val_ema_loss
            best_val_loss = val_loss
            results["best_epoch"] = epoch
            results["best_val_mae"] = val_loss
            results["best_val_ema_mae"] = val_ema_loss
            results["best_val_mae_unnormalized"] = val_metrics.get('mae', 0)
            
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved with validation EMA MAE: {val_ema_loss:.4f}")
            
            if normalizer:
                print(f"Unnormalized MAE: {val_metrics.get('mae', 0):.4f}")
        
        # Run test evaluation periodically
        if epoch % test_interval == 0 or epoch == epochs - 1:
            test_metrics = evaluate(model, test_loader, target_idx, device, normalizer)
            
            # Save test metrics
            results["test_metrics"][f"epoch_{epoch}"] = {
                "mae": test_metrics.get('mae', 0),
                "mae_normalized": test_metrics['mae_normalized']
            }
            
            # Plot predictions
            if normalizer:
                plot_path = f"{run_dir}/predictions_epoch_{epoch}.png"
                plot_predictions(test_metrics, target_idx, plot_path)
                print(f"Test MAE: {test_metrics.get('mae', 0):.4f}")
                print(f"Prediction plot saved to {plot_path}")
    
    # Final evaluation on test set with best model
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(checkpoint_path))
    final_test_metrics = evaluate(model, test_loader, target_idx, device, normalizer)
    
    # Save final test metrics
    results["final_test_metrics"] = {
        "mae": final_test_metrics.get('mae', 0),
        "mae_normalized": final_test_metrics['mae_normalized']
    }
    results["final_test_mae"] = final_test_metrics.get('mae', 0)
    
    # Generate final prediction plot
    final_plot_path = f"{run_dir}/final_predictions.png"
    plot_predictions(final_test_metrics, target_idx, final_plot_path)
    
    # Save all results
    with open(f"{run_dir}/results.json", 'w') as f:
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
    prop_name, unit = property_names.get(target_idx, (f'Property {target_idx}', ''))
    
    print(f"\n=== Final Results for {prop_name} ===")
    print(f"Best model saved at epoch {results['best_epoch']}")
    print(f"Final Test MAE: {results['final_test_mae']:.4f} {unit}")
    print(f"Final prediction plot saved to {final_plot_path}")
    
    # Close tensorboard writer if used
    if writer:
        writer.close()
    
    return results


if __name__ == '__main__':
    from torch_geometric.datasets import QM9
    from torch_geometric.loader import DataLoader
    # Import your model and data preparation

    args = parse_args()
    target_idx = args.target
    
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Task selection
    # target_idx = 0  # Dipole moment
    
    # Load data (using your improved data loading function)
    from old.data import prepare_data_loaders
    train_loader, val_loader, test_loader, normalizer = prepare_data_loaders(
        batch_size=100, 
        subset_size=None,
        target_idx=target_idx
    )

    # Initialize model
    from models.painn import PaiNN
    model = PaiNN()
    
    # Train model
    results = train_and_evaluate(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        target_idx=target_idx,
        normalizer=normalizer,
        epochs=400,
        lr=0.001,
        weight_decay=0.0001,
        patience=5,
        test_interval=10,
        device=device,
        use_tensorboard=True,
        ema_alpha=0.9  # EMA smoothing factor
    )