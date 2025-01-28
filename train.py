import torch
from torch.utils.data import DataLoader
from callbacks import Callback
from utils.learning_rate import poly_lr_scheduler
from utils.metrics import tabular_print, fast_hist, per_class_iou
from utils.model_analysis import setup_model_device
from utils.visualization import visualize_segmentation_results

from typing import Optional, Tuple
import tqdm as tqdm


def train_segmentation_model(
    current_epoch: int,
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    initial_learning_rate: float,
    max_iterations: int,
    learning_rate_decay_power: float = 0.9,
    learning_rate_decay_step: float = 1.0,
    device: str = 'cpu',
    training_callbacks: list[Callback] = []
) -> torch.nn.Module:
    """Trains a segmentation model for one epoch."""
    for callback in training_callbacks:
        callback.on_train_begin()

    model.train()
    epoch_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    
    # Training loop
    progress_bar = tqdm(enumerate(data_loader), 
                       total=len(data_loader), 
                       desc=f'Epoch {current_epoch + 1}', 
                       leave=False)
    
    for batch_idx, (images, labels) in progress_bar:
        # Calculate current training iteration
        global_iteration = current_epoch * len(data_loader) + batch_idx
        
        # Update learning rate if needed
        if global_iteration % learning_rate_decay_step == 0 and global_iteration <= max_iterations:
            poly_lr_scheduler(
                optimizer, 
                initial_learning_rate,
                global_iteration, 
                learning_rate_decay_step,
                max_iterations,
                learning_rate_decay_power
            )

        # Prepare data
        images = images.to(device)
        labels = labels.to(device).squeeze(1)  # Remove channel dimension

        # Forward pass and loss computation
        optimizer.zero_grad()
        model_outputs = model(images)
        batch_loss = compute_total_loss(model_outputs, labels, loss_function)
        
        # Backward pass
        batch_loss.backward()
        optimizer.step()
        
        # Update metrics
        epoch_loss += batch_loss.item()
        main_output = get_main_output(model_outputs)
        predictions = main_output.argmax(dim=1)
        
        # Compute accuracy
        batch_pixels = labels.numel()
        batch_correct = predictions.eq(labels).sum().item()
        correct_pixels += batch_correct
        total_pixels += batch_pixels

        # Update callbacks
        accuracy = 100.0 * correct_pixels / total_pixels
        update_training_metrics(training_callbacks, batch_idx, batch_loss.item(), accuracy)

    # Compute final metrics
    average_loss = epoch_loss / len(data_loader)
    final_accuracy = 100.0 * correct_pixels / total_pixels
    
    print(f'Epoch {current_epoch + 1} Results:')
    print(f'Average Loss: {average_loss:.6f}')
    print(f'Pixel Accuracy: {final_accuracy:.2f}%')

    # Final callback updates
    for callback in training_callbacks:
        callback.on_epoch_end(current_epoch, {
            'train_loss': average_loss,
            'train_accuracy': final_accuracy
        })

    return model


def compute_total_loss(
    model_outputs: torch.Tensor | Tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    loss_function: torch.nn.Module
) -> torch.Tensor:
    """Computes the total loss including auxiliary outputs if present."""
    if isinstance(model_outputs, tuple):
        main_output, aux1, aux2 = model_outputs
        total_loss = loss_function(main_output, labels)
        if aux1 is not None:
            total_loss += loss_function(aux1, labels)
        if aux2 is not None:
            total_loss += loss_function(aux2, labels)
        return total_loss
    return loss_function(model_outputs, labels)


def get_main_output(
    model_outputs: torch.Tensor | Tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """Extracts the main output from model outputs."""
    if isinstance(model_outputs, tuple):
        return model_outputs[0]
    return model_outputs


def update_training_metrics(
    callbacks: list[Callback],
    batch_index: int,
    loss: float,
    accuracy: float
) -> None:
    """Updates training metrics through callbacks."""
    for callback in callbacks:
        callback.on_batch_end(batch_index, {
            'train_loss': loss,
            'train_accuracy': accuracy,
        })

