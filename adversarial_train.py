import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from callbacks import Callback
from utils.learning_rate import poly_lr_scheduler
from utils.metrics import tabular_print, fast_hist, per_class_iou
from utils.model_analysis import setup_model_device
from utils.visualization import visualize_segmentation_results
from eval import val_GTA5
from typing import Tuple, Optional

try:
    from IPython import get_ipython
    if get_ipython():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except:
    from tqdm import tqdm


def train_generator_step(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    gen_optimizer: torch.optim.Optimizer,
    source_data: Tuple[torch.Tensor, torch.Tensor],
    target_data: Tuple[torch.Tensor, torch.Tensor],
    gen_loss: torch.nn.Module,
    disc_loss: torch.nn.Module,
    lambda_adv: float,
    device: str,
    iterations: int
) -> Tuple[float, float, float, torch.Tensor, int]:
    """Performs a single generator training step."""
    source_image, source_label = source_data
    target_image, _ = target_data
    
    # Move data to device
    source_image = source_image.to(device)
    source_label = source_label.to(device).squeeze(1)
    target_image = target_image.to(device)
    
    # Disable discriminator gradients during generator training
    for param in discriminator.parameters():
        param.requires_grad = False
        
    # Train on source domain
    source_output = generator(source_image)
    if isinstance(source_output, tuple):
        source_loss = gen_loss(source_output[0], source_label)
        source_loss += gen_loss(source_output[1], source_label)
        source_loss += gen_loss(source_output[2], source_label)
        source_features = source_output[0]
    else:
        source_loss = gen_loss(source_output, source_label)
        source_features = source_output

    # Normalize source loss
    source_loss = source_loss / iterations
    
    # Train on target domain
    target_output = generator(target_image)
    if isinstance(target_output, tuple):
        target_feature = target_output[0]
    else:
        target_feature = target_output
    
    # Compute adversarial loss
    pred_target = discriminator(F.softmax(target_feature, dim=1))
    source_mask = torch.ones(pred_target.size()).to(device)
    adv_loss = lambda_adv * disc_loss(pred_target, source_mask)
    adv_loss = adv_loss / iterations
    
    # Compute accuracy metrics
    predicted = source_features.argmax(dim=1)
    correct = predicted.eq(source_label).sum().item()
    total = source_label.numel()
    
    return source_loss, adv_loss, source_features, correct, total


def train_discriminator_step(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    source_features: torch.Tensor,
    target_feature: torch.Tensor,
    disc_loss: torch.nn.Module,
    device: str,
    iterations: int
) -> Tuple[float, float]:
    """Performs a single discriminator training step."""
    # Enable discriminator gradients
    for param in discriminator.parameters():
        param.requires_grad = True
    
    # Detach features to avoid gradient flow to generator
    source_features = source_features.detach()
    target_feature = target_feature.detach()
    
    # Train on source domain
    pred_source = discriminator(F.softmax(source_features, dim=1))
    source_mask = torch.ones(pred_source.size()).to(device)
    source_loss = disc_loss(pred_source, source_mask)
    source_loss = source_loss / iterations
    source_loss.backward()
    
    # Train on target domain
    pred_target = discriminator(F.softmax(target_feature, dim=1))
    target_mask = torch.zeros(pred_target.size()).to(device)
    target_loss = disc_loss(pred_target, target_mask)
    target_loss = target_loss / iterations
    target_loss.backward()
    
    return source_loss.item(), target_loss.item()


def adversarial_train(
    iterations: int,
    epochs: int,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    gen_optimizer: torch.optim.Optimizer,
    disc_optimizer: torch.optim.Optimizer,
    source_dataloader: DataLoader,
    target_dataloader: DataLoader,
    gen_loss: torch.nn.Module,
    disc_loss: torch.nn.Module,
    lambda_: float,
    gen_init_lr: float,
    gen_power: float,
    dis_power: float,
    dis_init_lr: float,
    lr_decay_iter: float,
    num_classes: int,
    class_names: list[str],
    val_loader: DataLoader,
    do_validation: int = 1,
    device: str = 'cpu',
    callbacks: list[Callback] = []
) -> None:
    """
    Trains a generator and discriminator in an adversarial setting for domain adaptation.
    Based on: https://openaccess.thecvf.com/content_cvpr_2018/papers/Tsai_Learning_to_Adapt_CVPR_2018_paper.pdf
    """
    best_miou = 0.0
    max_iter = epochs * iterations

    for epoch in range(epochs):
        # Initialize callbacks
        for callback in callbacks:
            callback.on_train_begin()

        # Initialize metrics
        running_metrics = {
            'gen_source_loss': 0.0,
            'adversarial_loss': 0.0,
            'disc_source_loss': 0.0,
            'disc_target_loss': 0.0,
            'generator_correct': 0,
            'generator_total': 0
        }

        # Set models to training mode
        generator.train()
        discriminator.train()

        # Update discriminator learning rate
        dis_lr = poly_lr_scheduler(
            disc_optimizer, dis_init_lr, epoch, 
            lr_decay_iter, epochs, dis_power
        )

        # Training loop
        for i in tqdm(range(iterations), total=iterations, desc=f'Epoch {epoch}'):
            # Reset gradients
            gen_optimizer.zero_grad()
            disc_optimizer.zero_grad()

            # Update generator learning rate
            current_iter = epoch * iterations + i
            if current_iter % lr_decay_iter == 0 and current_iter <= max_iter:
                gen_lr = poly_lr_scheduler(
                    gen_optimizer, gen_init_lr, current_iter,
                    lr_decay_iter, max_iter, gen_power
                )

            # Get batch data
            source_data = (next(iter(source_dataloader))[0].to(device),
                         next(iter(source_dataloader))[1].to(device))
            target_data = (next(iter(target_dataloader))[0].to(device),
                         next(iter(target_dataloader))[1].to(device))

            # Train generator
            gen_source_loss, adv_loss, source_features, correct, total = train_generator_step(
                generator, discriminator, gen_optimizer,
                source_data, target_data, gen_loss, disc_loss,
                lambda_, device, iterations
            )

            # Train discriminator
            disc_source_loss, disc_target_loss = train_discriminator_step(
                generator, discriminator, disc_optimizer,
                source_features, source_features, disc_loss,
                device, iterations
            )

            # Update metrics
            running_metrics['gen_source_loss'] += gen_source_loss
            running_metrics['adversarial_loss'] += adv_loss
            running_metrics['disc_source_loss'] += disc_source_loss
            running_metrics['disc_target_loss'] += disc_target_loss
            running_metrics['generator_correct'] += correct
            running_metrics['generator_total'] += total

            # Update callbacks
            for callback in callbacks:
                callback.on_batch_end(i, {
                    'loss_gen_source': gen_source_loss,
                    'loss_adversarial': adv_loss,
                    'loss_disc_source': disc_source_loss,
                    'loss_disc_target': disc_target_loss
                })

        # Compute epoch metrics
        epoch_metrics = compute_epoch_metrics(running_metrics, iterations)
        
        # Print epoch results
        print(f'Epoch Results {epoch}')
        tabular_print({
            **epoch_metrics,
            'dis_lr': dis_lr if dis_lr else -1,
            'gen_lr': gen_lr if gen_lr else -1,
        })

        # Update epoch end callbacks
        for callback in callbacks:
            callback.on_epoch_end(epoch, {
                'dis_lr': dis_lr if dis_lr else -1,
                'gen_lr': gen_lr if gen_lr else -1,
                'Generator_Accuracy': epoch_metrics['Generator_Accuracy']
            })

        # Validation phase
        if epoch % do_validation == 0 and do_validation != 0:
            validation_miou = validate_model(
                epoch, generator, discriminator, val_loader,
                num_classes, class_names, callbacks, device
            )
            
            # Save best model
            if validation_miou > best_miou:
                best_miou = validation_miou
                save_models(generator, discriminator, epoch, validation_miou)

    # Finish training
    for callback in callbacks:
        callback.on_train_end()


def compute_epoch_metrics(metrics: dict, iterations: int) -> dict:
    """Computes final metrics for the epoch."""
    return {
        'Generator_Loss': metrics['gen_source_loss'] / iterations,
        'Adversarial_Loss': metrics['adversarial_loss'] / iterations,
        'Discriminator_Source_Loss': metrics['disc_source_loss'] / iterations,
        'Discriminator_Target_Loss': metrics['disc_target_loss'] / iterations,
        'Generator_Accuracy': 100.0 * metrics['generator_correct'] / max(metrics['generator_total'], 1)
    }


def validate_model(
    epoch: int,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    val_loader: DataLoader,
    num_classes: int,
    class_names: list[str],
    callbacks: list[Callback],
    device: str
) -> float:
    """Performs validation and returns the mean IoU score."""
    print('-' * 50, 'Validation', '-' * 50)
    validation_miou, _ = val_GTA5(
        epoch, generator, val_loader,
        num_classes, class_names, callbacks,
        device=device
    )
    print('-' * 100)
    return validation_miou


def save_models(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    epoch: int,
    miou: float
) -> None:
    """Saves the models with their current performance metrics."""
    torch.save(generator.state_dict(), 'best_generator.pth')
    torch.save(discriminator.state_dict(), 'best_discriminator.pth')
    print(f'Best Model Saved at Epoch {epoch} with mIoU: {miou:.4f}')