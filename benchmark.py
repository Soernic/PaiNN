import torch
import torch.nn as nn
import torch.nn.functional as F
from painn import PaiNN
import matplotlib.pyplot as plt
import numpy as np




def benchmark_model(model_path, test_loader, target_idx, target_normalizer, device):
    # Load the best model
    model = PaiNN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Tracking metrics
    mse_total = 0.0
    mae_total = 0.0
    sample_count = 0
    
    # Original-scale predictions and targets for visualization
    all_preds_original = []
    all_targets_original = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device) 
            
            # Get normalized predictions
            pred_normalized = model(data)
            targets_normalized = data.y[:, target_idx].unsqueeze(1)
            
            # Calculate normalized errors
            mse = F.mse_loss(pred_normalized, targets_normalized)
            mae = F.l1_loss(pred_normalized, targets_normalized)
            
            # Store for calculation
            # batch_size = targets_normalized.size(0)
            batch_size = data.y.size(0)
            mse_total += mse.item() * batch_size
            mae_total += mae.item() * batch_size
            sample_count += batch_size
            
            # Convert to original scale for visualization
            pred_original = pred_normalized * target_normalizer.std + target_normalizer.mean
            targets_original = targets_normalized * target_normalizer.std + target_normalizer.mean
            

            all_preds_original.extend(pred_original.cpu().numpy().flatten())
            all_targets_original.extend(targets_original.cpu().numpy().flatten())
    
        # Debug information
        print(f"Normalized predictions - min: {min(pred_normalized.cpu().numpy().flatten())}, max: {max(pred_normalized.cpu().numpy().flatten())}")
        print(f"Normalized targets - min: {min(targets_normalized.cpu().numpy().flatten())}, max: {max(targets_normalized.cpu().numpy().flatten())}")
        print(f"Original predictions - min: {min(all_preds_original)}, max: {max(all_preds_original)}")
        print(f"Original targets - min: {min(all_targets_original)}, max: {max(all_targets_original)}")
        print(f"Prediction std: {np.std(all_preds_original)}, Target std: {np.std(all_targets_original)}")



    # Calculate final metrics
    normalized_mse = mse_total / sample_count
    normalized_mae = mae_total / sample_count
    
    # Convert to original scale
    original_rmse = (normalized_mse ** 0.5) * target_normalizer.std
    original_mae = normalized_mae * target_normalizer.std
    
    # Get property name and unit for nice reporting
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
    
    # Print results
    print(f"\n=== Benchmark Results for {prop_name} ===")
    print(f"Test set size: {sample_count} molecules")
    print(f"MAE: {original_mae:.4f} {unit}")
    print(f"RMSE: {original_rmse:.4f} {unit}")
    
    # Optional: visualize prediction vs actual
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.scatter(all_targets_original, all_preds_original, alpha=0.5)
        plt.plot([min(all_targets_original), max(all_targets_original)], 
                 [min(all_targets_original), max(all_targets_original)], 
                 'r--')
        plt.xlabel(f'True {prop_name} ({unit})')
        plt.ylabel(f'Predicted {prop_name} ({unit})')
        plt.title(f'PAINN Model Performance on {prop_name}')
        plt.savefig(f'prediction_plot_{target_idx}.png')
        print(f"Prediction plot saved as prediction_plot_{target_idx}.png")
    except ImportError:
        print("Matplotlib not available for visualization")
    
    return {
        'mae': original_mae,
        'rmse': original_rmse,
        'normalized_mse': normalized_mse,
        'normalized_mae': normalized_mae,
        'property': prop_name,
        'unit': unit
    }


if __name__ == '__main__':
    from data import get_QM9
    target_idx = 1

    best_model = f'best_model_prop_{target_idx}.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train, val, test, normaliser = get_QM9(batch_size=256, subset_until=5000, target_idx=target_idx)

    benchmark_model(best_model, test, target_idx, normaliser, device)
