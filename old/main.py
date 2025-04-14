import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt



def main(
        train_loader,
        val_loader,
        test_loader,
        target_idx,
        model,
        optimiser,
        scheduler,  # Added scheduler
        epochs,
        target_normalizer,  # Optional for reporting unnormalized metrics
        model_save_path
    ):
    
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        num_samples = 0
        
        for data in train_loader:
            
            # Move to gpu
            data = data.to(device)
            
            optimiser.zero_grad()
            # Forward pass
            out = model(data)
            # Get target (focusing on a single property)
            target = data.y[:, target_idx].unsqueeze(1)
            # Compute loss
            loss = F.mse_loss(out, target)
            # Backward pass
            loss.backward()
            # Update parameters
            optimiser.step()
            
            # Add batch loss
            batch_size = data.y.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size
        
        # Compute average loss properly
        train_loss = total_loss / num_samples
        
        # Validation
        model.eval()
        val_total_loss = 0
        mae_total_loss = 0.0
        val_num_samples = 0
        
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                target = data.y[:, target_idx].unsqueeze(1)
                val_total_loss += F.mse_loss(out, target).item() * data.y.size(0)
                mae_total_loss += F.l1_loss(out, target).item() * data.y.size(0) # count the mae
                val_num_samples += data.y.size(0)
        
        val_loss = val_total_loss / val_num_samples
        mae_loss = mae_total_loss / val_num_samples
        
        # Update the scheduler
        if scheduler:
            scheduler.step(val_loss)
            current_lr = optimiser.param_groups[0]['lr']
            print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}')
        else:
            print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_mae_loss = mae_loss
            torch.save(model.state_dict(), 'checkpoints' + model_save_path)
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
            
            # Report the unnormalized error if normalizer is provided
            if target_normalizer:
                # Convert normalized MSE to unnormalized units
                unnormalized_rmse = (best_val_loss ** 0.5) * target_normalizer.std
                unnormalized_mae = best_mae_loss * target_normalizer.std
                print(f"Unnormalized MAE: {unnormalized_mae:.4f}")
                print(f"Unnormalized RMSE: {unnormalized_rmse:.4f}")

            if epoch % 20 == 0:
                model.eval()
                test_total_loss = 0.0
                test_num_samples = 0

                all_preds_original = []
                all_targets_original = []

                with torch.no_grad():
                    for data in test_loader:
                        data = data.to(device)
                        out = model(data)
                        target = data.y[:, target_idx].unsqueeze(1)
                        test_total_loss += F.l1_loss(out, target).item() * data.y.size(0)
                        test_num_samples += data.y.size(0)

                        pred_original = out * target_normalizer.std + target_normalizer.mean
                        target_original = target * target_normalizer.std + target_normalizer.mean

                        all_preds_original.extend(pred_original.cpu().numpy().flatten())
                        all_targets_original.extend(target_original.cpu().numpy().flatten())

                    # Debug information
                    print(f"Normalized predictions - min: {min(out.cpu().numpy().flatten()):.4f}, max: {max(out.cpu().numpy().flatten()):.4f}")
                    print(f"Normalized targets -     min: {min(target.cpu().numpy().flatten()):.4f}, max: {max(target.cpu().numpy().flatten()):.4f}")
                    print(f"Original predictions -   min: {min(all_preds_original):.4f}, max: {max(all_preds_original):.4f}")
                    print(f"Original targets -       min: {min(all_targets_original):.4f}, max: {max(all_targets_original):.4f}")
                    # print(f"Prediction std: {np.std(all_preds_original)}, Target std: {np.std(all_targets_original)}")




                test_loss = test_total_loss / test_num_samples
                unnormalised_test_mae = test_loss*target_normalizer.std
                # print(f"Unnormalized TEST MAE: {unnormalised_test_mae:.4f}")

                
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
                print(f"Test set size: {test_num_samples} molecules")
                print(f"MAE: {unnormalised_test_mae:.4f} {unit}")
                # print(f"RMSE: {original_rmse:.4f} {unit}")
                
                plt.figure(figsize=(8, 8))
                plt.scatter(all_targets_original, all_preds_original, alpha=0.5)
                plt.plot([min(all_targets_original), max(all_targets_original)], 
                        [min(all_targets_original), max(all_targets_original)], 
                        'r--')
                plt.xlabel(f'True {prop_name} ({unit})')
                plt.ylabel(f'Predicted {prop_name} ({unit})')
                plt.title(f'PAINN Model Performance on {prop_name}')
                plt.savefig(f'plots/prediction_plot_{target_idx}.png')
                print(f"Prediction plot saved as prediction_plot_{target_idx}.png")


if __name__ == '__main__':

    from data import get_QM9
    from painn import PaiNN
    from benchmark import benchmark_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    task_list = {
        0: 'mu',
        1: 'alpha',
        2: 'eps_homo',
        3: 'eps_lumo',
        4: 'delta_eps'
        # continue this
    }

    target_idx = 0  # alpha (scalar)
    train_loader, val_loader, test_loader, target_normalizer = get_QM9(
        batch_size=100, 
        subset_until=5000,
        target_idx=target_idx
    )

    model = PaiNN().to(device)

    # Setup optimizer with weight decay for better regularization
    optimiser = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    # Setup learning rate scheduler that reduces LR when validation loss plateaus
    scheduler = ReduceLROnPlateau(
        optimiser, 
        mode='min',
        factor=0.8,  # Multiply LR by this factor when reducing
        patience=5,   # Wait for this many epochs before reducing LR
    )

    epochs = 200
    model_save_path = f'best_model_prop_{target_idx}.pt'

    main(
        train_loader,
        val_loader,
        test_loader,
        target_idx,
        model,
        optimiser,
        scheduler,
        epochs,
        target_normalizer,
        model_save_path
    )

    # results = benchmark_model(
    #     model_save_path,
    #     test_loader,
    #     target_idx,
    #     target_normalizer,
    #     device
    # )

