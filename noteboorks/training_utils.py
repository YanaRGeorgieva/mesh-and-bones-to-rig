import os
import torch
import torch.nn as nn
import torch.nn.init as init
from torch_geometric.loader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.losses.combined_loss import combined_loss
from src.models.mesh_bones_to_rig import MeshBonesToRigNet

def initialize_module_xavier_uniform(m):
    """
    Initialize the weights of a module using Xavier uniform initialization.
    """
    # Initialize standard Linear layers.
    if isinstance(m, nn.Linear):
        # Xavier uniform initialization works well for many activations.
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.1)
    # For modules from torch_geometric that use an internal linear layer:
    elif hasattr(m, 'lin') and isinstance(m.lin, nn.Linear):
        init.xavier_uniform_(m.lin.weight)
        if m.lin.bias is not None:
            nn.init.constant_(m.lin.bias, 0.1)

def train_model(model, train_dataset, hyperparams, run_title, validate_dataset=None, save_checkpoint=False, checkpoint_dir=None):
    """
    Train the model.
    Will save the best model and the final model only if we have a validation set and save_checkpoint is True with a valid checkpoint_dir.
    """
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # If the model is not provided, instantiate it with the given parameters.
    if model is None:
        model = MeshBonesToRigNet(hyperparams['mesh_encoder_in_channels'],
                              hyperparams['mesh_encoder_hidden_channels'],
                              hyperparams['mesh_encoder_out_channels'],
                              hyperparams['mesh_encoder_kernel_size'],
                              hyperparams['mesh_encoder_num_layers'],
                              hyperparams['mesh_encoder_dim'],
                              hyperparams['bone_encoder_in_channels'],
                              hyperparams['bone_encoder_hidden_channels'],
                              hyperparams['bone_encoder_out_channels'],
                              hyperparams['bone_encoder_num_layers'],
                              hyperparams['fusion_common_dim'],
                              hyperparams['fusion_top_k'],
                              hyperparams['fusion_alpha'],
                              hyperparams['fusion_alpha_learnable'],
                              hyperparams['refinement_gamma'],
                              hyperparams['with_refinement']
                            )
    # Move the model to the device
    model.to(device)

    # Apply the initialization function to the model.
    if hyperparams['initialization'] == 'xavier_uniform_':
        model.apply(initialize_module_xavier_uniform)

    # Set the optimizer
    if hyperparams['optimizer'].lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    elif hyperparams['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=hyperparams['learning_rate'])
    else:
        raise ValueError(f"Optimizer {hyperparams['optimizer']} not supported")

    # Create the train loader
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)

    # Create the validate loader
    if validate_dataset is not None:
        validate_loader = DataLoader(validate_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
    else:
        validate_loader = None

    # Create a tensorboard writer
    writer = SummaryWriter(run_title)
    # Metrics dictionary
    metrics = {
        'final_train_loss': 0.0,  # Update this with final loss
        'final_val_loss': 0.0,  # Update this with final loss
        'best_train_loss': 0.0,    # Update this with best loss
        'best_val_loss': 0.0  # Update this with best validation loss
    }
    # Add hyperparameters
    writer.add_hparams(hyperparams, metrics)

    # Add detailed text description
    writer.add_text('Experiment Description',
        f"""
        Training Settings:
        - Mesh: {hyperparams['mesh_name']}
        - Epochs: {hyperparams['num_epochs']}
        - Learning Rate: {hyperparams['learning_rate']}
        - Batch Size: {hyperparams['batch_size']}
        - Initialization: {hyperparams['initialization']}

        Model Architecture:
        - Mesh Encoder: {hyperparams['mesh_encoder_hidden_channels']} hidden, {hyperparams['mesh_encoder_out_channels']} out, {hyperparams['mesh_encoder_num_layers']} layers
        - Bone Encoder: {hyperparams['bone_encoder_hidden_channels']} hidden, {hyperparams['bone_encoder_out_channels']} out, {hyperparams['bone_encoder_num_layers']} layers
        - Fusion: top-k={hyperparams['fusion_top_k']} alpha={hyperparams['fusion_alpha']} alpha_learnable={hyperparams['fusion_alpha_learnable']}
        - Refinement: {'Enabled' if hyperparams['with_refinement'] else 'Disabled'} gamma={hyperparams['refinement_gamma']}
        - Loss Settings:
            - Skinning Loss: {hyperparams['lambda_skin']}
            - Geodesic Loss: {hyperparams['lambda_geo']}
                - Geodesic Alpha: {hyperparams['lambda_geo_alpha']}
            - Smoothness Loss: {hyperparams['lambda_smooth']}
        """,
        0)

    # Initialize tracking variables
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    final_train_loss = 0.0
    final_val_loss = 0.0

    if save_checkpoint:
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir must be provided if save_checkpoint is True")
        else:
            os.makedirs(checkpoint_dir, exist_ok=True)
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")


    # Training loop
    for epoch in range(hyperparams['num_epochs']):
        total_train_loss = train_epoch(model,
                                    train_loader,
                                    optimizer,
                                    device,
                                    hyperparams['lambda_skin'],
                                    hyperparams['lambda_geo'],
                                    hyperparams['lambda_smooth'],
                                    hyperparams['lambda_geo_alpha'],
                                    epoch,
                                    writer)
        print(f"Epoch {epoch+1}/{hyperparams['num_epochs']} - Train Loss: {total_train_loss:.4f}")

        if validate_loader is not None:
            validate_loss = validate_epoch(model,
                                            validate_loader,
                                            device,
                                            epoch,
                                            hyperparams['lambda_skin'],
                                            hyperparams['lambda_geo'],
                                            hyperparams['lambda_smooth'],
                                            hyperparams['lambda_geo_alpha'],
                                            writer)
            print(f"Epoch {epoch+1}/{hyperparams['num_epochs']} - Validate Loss: {validate_loss:.4f}")
            if save_checkpoint:
                # Save checkpoint every 10 epochs
                if (epoch + 1) % 10 == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": validate_loss
                    }, checkpoint_path)
                    print(f"Checkpoint saved at {checkpoint_path}")
                # Save the best model (lowest validation loss)
                if validate_loss < best_val_loss:
                    best_val_loss = validate_loss
                    torch.save({
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": validate_loss
                    }, best_model_path)
                    print(f"Best model updated with val_loss {validate_loss:.4f} at epoch {epoch+1}")
            # Log current losses
            writer.add_scalar('Validate/Loss', validate_loss, epoch)
            writer.add_scalar('Validate/Best_Loss', best_val_loss, epoch)
            # Update final loss each epoch
            final_val_loss = validate_loss

        # Track best loss
        if total_train_loss < best_train_loss:
            best_train_loss = total_train_loss

        # Update final loss each epoch
        final_train_loss = total_train_loss

        # Log current losses
        writer.add_scalar('Training/Loss', total_train_loss, epoch)
        writer.add_scalar('Training/Best_Loss', best_train_loss, epoch)

    # After training, update the hparams metrics
    if validate_loader is not None:
        metrics = {
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'best_train_loss': best_train_loss,
            'best_val_loss': best_val_loss
        }
    else:
        metrics = {
            'final_train_loss': final_train_loss,
            'best_train_loss': best_train_loss
        }

    # Add final hyperparameters and metrics
    writer.add_hparams(hyperparams, metrics)

    # Close the writer when done
    writer.close()

    return model

def train_epoch(model, dataloader, optimizer, device, epoch, lambda_skin=1.0, lambda_geo=1.0, lambda_smooth=1.0, lambda_geo_alpha=1.0, writer=None):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        batch.to(device)
        optimizer.zero_grad()
        predicted_weights = model(batch["vertices"],
                                  batch["edge_index_geodesic"],
                                  batch["edge_attr_geodesic"],
                                  batch["vertex_neighbors"],
                                  batch["vertex_adj"],
                                  batch["vertex_normals"],
                                  batch["bone_features"],
                                  batch["bone_adj"],
                                  batch["volumetric_geodesic"],
                                  batch["surface_geodesic"]
                                )
        loss_skin, loss_geo, loss_smooth, loss = combined_loss(predicted_weights,
                             batch["target_skin_weights"],
                             batch["volumetric_geodesic"],
                             batch["vertex_adj"],
                             alpha=lambda_geo_alpha,
                             lambda_skin=lambda_skin,
                             lambda_geo=lambda_geo,
                             lambda_smooth=lambda_smooth
                            )
        if writer:
            # Log the loss to tensorboard
            writer.add_scalar('Train/Loss', loss, epoch)
            writer.add_scalar('Train/Skinning loss', loss_skin, epoch)
            writer.add_scalar('Train/Geodesic loss', loss_geo, epoch)
            writer.add_scalar('Train/Smoothness loss', loss_smooth, epoch)

        loss.backward()

        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def validate_epoch(model, dataloader, device, epoch, lambda_skin=1.0, lambda_geo=1.0, lambda_smooth=1.0, lambda_geo_alpha=1.0, writer=None):
    """
    Validate the model for one epoch.
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch.to(device)
            predicted_weights = model(batch["vertices"],
                                      batch["edge_index_geodesic"],
                                      batch["edge_attr_geodesic"],
                                      batch["vertex_neighbors"],
                                      batch["vertex_adj"],
                                      batch["vertex_normals"],
                                      batch["bone_features"],
                                      batch["bone_adj"],
                                      batch["volumetric_geodesic"],
                                      batch["surface_geodesic"]
                                    )
            loss_skin, loss_geo, loss_smooth, loss = combined_loss(predicted_weights,
                             batch["target_skin_weights"],
                             batch["volumetric_geodesic"],
                             batch["vertex_adj"],
                             alpha=lambda_geo_alpha,
                             lambda_skin=lambda_skin,
                             lambda_geo=lambda_geo,
                             lambda_smooth=lambda_smooth
                            )
            if writer:
                # Log the loss to tensorboard
                writer.add_scalar('Validate/Loss', loss, epoch)
                writer.add_scalar('Validate/Skinning loss', loss_skin, epoch)
                writer.add_scalar('Validate/Geodesic loss', loss_geo, epoch)
                writer.add_scalar('Validate/Smoothness loss', loss_smooth, epoch)

            running_loss += loss.item()
    return running_loss / len(dataloader)

def test_model(model, dataloader, device, lambda_skin=1.0, lambda_geo=1.0, lambda_smooth=1.0, lambda_geo_alpha=1.0):
    """
    Test the model.
    """
    return validate_epoch(model, dataloader, device, 0, lambda_skin, lambda_geo, lambda_smooth, lambda_geo_alpha)