import argparse
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models.deeplabv2 import deeplabv2
from models.bisenet import build_bisenet
from models.adversarial.model import DomainDiscriminator, TinyDomainDiscriminator
from datasets.cityscapes import CityScapes
from datasets.gta5 import GTA5
from utils.learning_rate import poly_lr_scheduler
from utils.metrics import tabular_print, fast_hist, per_class_iou
from utils.model_analysis import setup_model_device, IntRangeTransformer
from utils.visualization import visualize_segmentation_results



def setup_training_arguments() -> argparse.Namespace:
    """Sets up and parses command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description='Semantic Segmentation and Domain Adaptation Training'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to the configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='cityscapes',
        choices=['cityscapes', 'gta5'],
        help='Dataset to use for training (default: cityscapes)'
    )
    
    parser.add_argument(
        '--augmented',
        action='store_true',
        help='Enable data augmentation for GTA5 dataset'
    )
    
    parser.add_argument(
        '--domain_adaptation',
        action='store_true',
        help='Enable domain adaptation training'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='bisenet',
        choices=['deeplab', 'bisenet'],
        help='Model architecture for segmentation (default: bisenet)'
    )
    
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    return parser.parse_args()

def setup_augmentations(config: Any, name: str) -> transforms.Transform:
    """Creates augmentation transforms based on configuration."""
    aug_config = config.augmentation.get(name, {})
    
    if name == 'GaussianBlur':
        kernel_size = [int(i) for i in aug_config['kernel_size'].split(',')]
        sigma = [float(i) for i in aug_config['sigma'].split(',')]
        return transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
        
    elif name == 'RandomHorizontalFlip':
        return transforms.RandomHorizontalFlip(p=aug_config['p'])
        
    elif name == 'ColorJitter':
        return transforms.ColorJitter(
            brightness=aug_config['brightness'],
            contrast=aug_config['contrast'],
            saturation=aug_config['saturation'],
            hue=aug_config['hue']
        )
    
    return None

def create_augmentation_pipeline(config: Any, probability: float) -> transforms.RandomApply:
    """Creates a complete augmentation pipeline."""
    augmentations = []
    for aug_name in config.augmentation.keys():
        aug = setup_augmentations(config, aug_name)
        if aug is not None:
            augmentations.append(aug)
    
    return transforms.RandomApply(augmentations, p=probability)

def setup_datasets(
    config: Any,
    use_augmentation: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Sets up dataset loaders for both Cityscapes and GTA5.
    
    Returns:
        Tuple containing (train_loader, val_loader, gta5_loader)
    """
    # Setup Cityscapes transforms
    cityscapes_cfg = config.data.cityscapes
    cityscapes_size = [int(i) for i in cityscapes_cfg.image_size.split(',')]
    
    cityscapes_input_transform = transforms.Compose([
        transforms.Resize(cityscapes_size, antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    cityscapes_target_transform = transforms.Compose([
        transforms.Resize(cityscapes_size, antialias=True),
        IntRangeTransformer(min_val=0, max_val=cityscapes_cfg.num_classes)
    ])
    
    # Setup GTA5 transforms
    gta5_cfg = config.data.gta5_modified
    gta5_size = [int(i) for i in gta5_cfg.image_size.split(',')]
    
    gta5_transforms = [
        transforms.Resize(gta5_size),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    if use_augmentation:
        gta5_transforms.insert(0, create_augmentation_pipeline(config, 0.5))
    
    gta5_input_transform = transforms.Compose(gta5_transforms)
    gta5_target_transform = transforms.Compose([
        transforms.Resize(gta5_size)
    ])
    
    # Create datasets
    train_dataset = CityScapes(
        cityscapes_cfg.segmentation_train_dir,
        cityscapes_cfg.images_train_dir,
        cityscapes_input_transform,
        cityscapes_target_transform
    )
    
    val_dataset = CityScapes(
        cityscapes_cfg.segmentation_val_dir,
        cityscapes_cfg.images_val_dir,
        cityscapes_input_transform,
        cityscapes_target_transform
    )
    
    gta5_dataset = GTA5(
        gta5_cfg.images_dir,
        gta5_cfg.segmentation_dir,
        gta5_input_transform,
        gta5_target_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cityscapes_cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cityscapes_cfg.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cityscapes_cfg.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=cityscapes_cfg.num_workers
    )
    
    gta5_loader = DataLoader(
        gta5_dataset,
        batch_size=gta5_cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=gta5_cfg.num_workers
    )