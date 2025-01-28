import warnings
warnings.filterwarnings("ignore")

import yaml
import argparse
from collections import namedtuple
import numpy as np
import torch
from torch.utils.data import DataLoader

# Local imports
from configuration import (
    setup_augmentations,
    setup_datasets,
    setup_model_components,
    setup_training_arguments
)

from callbacks import WandBCallback


import torch
import numpy as np
import random

def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all random number generators.
    
    This function sets the seed for PyTorch operations (both CPU and CUDA),
    NumPy operations, and Python's built-in random number generator.
    
    Args:
        seed: Integer value to use as random seed
    """
    # Set Python's random seed
    random.seed(seed)
    
    # Set NumPy's random seed
    np.random.seed(seed)
    
    # Set PyTorch's random seed
    torch.manual_seed(seed)
    
    # If CUDA is available, set the seed for CUDA operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        
        # Additional settings for better reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get the current learning rate from an optimizer.
    
    Args:
        optimizer: PyTorch optimizer instance
        
    Returns:
        float: Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    filename: str,
    is_best: bool = False
) -> None:
    """
    Save model checkpoint to file.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch number
        filename: Path to save the checkpoint
        is_best: If True, also save as best model
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # Save the regular checkpoint
    torch.save(checkpoint, filename)
    
    # If this is the best model so far, save it as best model
    if is_best:
        best_filename = filename.rsplit('.', 1)[0] + '_best.pth'
        torch.save(checkpoint, best_filename)

class TrainingOrchestrator:
    """
    Orchestrates the training process for semantic segmentation models.
    This class handles configuration, model setup, and training execution
    for both standard training and domain adaptation scenarios.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the training orchestrator with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.callbacks = []
        
    def _load_config(self, config_path: str) -> namedtuple:
        """Loads and validates the configuration file."""
        try:
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
                return namedtuple('Config', config_dict.keys())(*config_dict.values())
        except FileNotFoundError:
            raise FileNotFoundError(
                f'Configuration file not found at {config_path}. '
                'Please provide the correct path to the config file.'
            )

    def setup_training_environment(self, args: argparse.Namespace) -> None:
        """
        Sets up the training environment including datasets, models, and callbacks.
        
        Args:
            args: Command line arguments
        """
        # Set random seeds for reproducibility
        set_random_seed(args.seed)
        
        # Setup data loaders
        self.train_loader, self.val_loader, self.gta5_loader = setup_datasets(
            self.config,
            use_augmentation=args.augmented
        )
        
        # Initialize callbacks
        if args.wandb:
            self._setup_wandb_callback()
            
        # Setup model and training components
        if args.domain_adaptation:
            self._setup_domain_adaptation_components(args.model)
        else:
            self._setup_standard_training_components(args.model, args.dataset)

    def _setup_wandb_callback(self) -> None:
        """Configures and initializes the Weights & Biases callback."""
        callbacks_cfg = self.config.callbacks.logging.wandb
        self.callbacks.append(
            WandBCallback(
                project_name=callbacks_cfg.project_name,
                run_name=callbacks_cfg.run_name,
                config=self.config._asdict(),
                note=callbacks_cfg.note
            )
        )

    def _setup_domain_adaptation_components(self, model_name: str) -> None:
        """
        Sets up components needed for domain adaptation training.
        
        Args:
            model_name: Name of the model architecture to use
        """
        self.generator_config, self.discriminator_config = setup_model_components(
            self.config,
            is_adversarial=True,
            model_name=model_name
        )
        
        # Unpack configurations
        (self.generator, self.gen_optimizer, self.gen_loss, self.gen_hparams) = self.generator_config
        (self.discriminator, self.disc_optimizer, self.disc_loss, self.disc_hparams) = self.discriminator_config

    def _setup_standard_training_components(self, model_name: str, dataset: str) -> None:
        """
        Sets up components needed for standard semantic segmentation training.
        
        Args:
            model_name: Name of the model architecture to use
            dataset: Name of the dataset to use for training
        """
        if dataset == 'gta5':
            print('Training model on GTA5 dataset and validating on Cityscapes dataset')
            self.train_loader = self.gta5_loader
            
        self.model_config = setup_model_components(
            self.config,
            is_adversarial=False,
            model_name=model_name
        )
        self.model, self.optimizer, self.criterion, self.model_hparams = self.model_config

    def train(self) -> None:
        """Executes the training process based on the configured settings."""
        if hasattr(self, 'generator'):
            self._run_domain_adaptation_training()
        else:
            self._run_standard_training()

    def _run_domain_adaptation_training(self) -> None:
        """Executes domain adaptation training using adversarial approach."""
        from train import adversarial_train
        
        training_cfg = self.config.training.domain_adaptation
        
        adversarial_train(
            iterations=training_cfg.iterations,
            epochs=training_cfg.epochs,
            lambda_=training_cfg.lambda_,
            generator=self.generator,
            discriminator=self.discriminator,
            generator_optimizer=self.gen_optimizer,
            discriminator_optimizer=self.disc_optimizer,
            generator_loss=self.gen_loss,
            discriminator_loss=self.disc_loss,
            source_dataloader=self.gta5_loader,
            target_dataloader=self.train_loader,
            gen_init_lr=self.gen_hparams['gen_init_lr'],
            dis_init_lr=self.disc_hparams['dis_init_lr'],
            lr_decay_iter=training_cfg.lr_decay_iter,
            gen_power=self.gen_hparams['gen_power'],
            dis_power=self.disc_hparams['dis_power'],
            num_classes=training_cfg.num_classes,
            class_names=self.config.meta.class_names,
            val_loader=self.val_loader,
            do_validation=training_cfg.do_validation,
            when_print=training_cfg.when_print,
            callbacks=self.callbacks,
            device=self.config.device
        )

    def _run_standard_training(self) -> None:
        """Executes standard semantic segmentation training."""
        from train import train
        from eval import val
        
        training_cfg = self.config.training.segmentation
        max_iter = training_cfg.epochs * len(self.train_loader)
        
        for epoch in range(training_cfg.epochs):
            # Training phase
            train(
                model=self.model,
                optimizer=self.optimizer,
                criterion=self.criterion,
                train_loader=self.train_loader,
                epoch=epoch,
                init_lr=self.model_hparams['init_lr'],
                lr_decay_iter=training_cfg.lr_decay_iter,
                power=self.model_hparams['power'],
                max_iter=max_iter,
                callbacks=self.callbacks,
                device=self.config.device
            )
            
            # Validation phase
            val(
                epoch=epoch,
                model=self.model,
                val_loader=self.val_loader,
                num_classes=training_cfg.num_classes,
                class_names=self.config.meta.class_names,
                device=self.config.device,
                callbacks=self.callbacks
            )

def main():
    """Main entry point for the training orchestrator."""
    args = setup_training_arguments()
    orchestrator = TrainingOrchestrator(args.config)
    orchestrator.setup_training_environment(args)
    orchestrator.train()

if __name__ == '__main__':
    main()