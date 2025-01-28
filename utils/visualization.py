import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union

# Cityscapes color palette for semantic segmentation
CITYSCAPES_PALETTE = {
    0: [128, 64, 128],   # road
    1: [244, 35, 232],   # sidewalk
    2: [70, 70, 70],     # building
    3: [102, 102, 156],  # wall
    4: [190, 153, 153],  # fence
    5: [153, 153, 153],  # pole
    6: [250, 170, 30],   # traffic light
    7: [220, 220, 0],    # traffic sign
    8: [107, 142, 35],   # vegetation
    9: [152, 251, 152],  # terrain
    10: [70, 130, 180],  # sky
    11: [220, 20, 60],   # person
    12: [255, 0, 0],     # rider
    13: [0, 0, 142],     # car
    14: [0, 0, 70],      # truck
    15: [0, 60, 100],    # bus
    16: [0, 80, 100],    # train
    17: [0, 0, 230],     # motorcycle
    18: [119, 11, 32],   # bicycle
}

def apply_cityscapes_color_map(
    segmentation_map: np.ndarray,
    color_palette: Dict[int, List[int]] = CITYSCAPES_PALETTE
) -> np.ndarray:
    """Applies the Cityscapes color palette to a segmentation map.
    
    Args:
        segmentation_map: Input segmentation map
        color_palette: Dictionary mapping class indices to RGB colors
    
    Returns:
        np.ndarray: Colored segmentation map
    """
    h, w = segmentation_map.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    for key, color in color_palette.items():
        color_image[segmentation_map == key] = color
    return color_image

def rescale_for_display(input_tensor: torch.Tensor) -> torch.Tensor:
    """Rescales a tensor to [0, 1] range for visualization.
    
    Args:
        input_tensor: Input tensor to rescale
    
    Returns:
        torch.Tensor: Rescaled tensor
    """
    min_val = input_tensor.min()
    max_val = input_tensor.max()
    return (input_tensor - min_val) / (max_val - min_val)

def visualize_segmentation_results(
    inputs_list: List[torch.Tensor],
    targets_list: List[torch.Tensor],
    predictions: List[torch.Tensor],
    num_batches: int = 5
) -> None:
    """Visualizes segmentation results showing input, ground truth, and prediction.
    
    Args:
        inputs_list: List of input image batches
        targets_list: List of target segmentation maps
        predictions: List of predicted segmentation maps
        num_batches: Number of batches to visualize
    """
    num_batches = min(num_batches, len(inputs_list))
    fig, axes = plt.subplots(nrows=num_batches, ncols=3, figsize=(18, num_batches * 6))
    
    for idx in range(num_batches):
        ax = axes[idx] if num_batches > 1 else axes
        
        # Process input image
        input_tensor = inputs_list[idx][0]
        input_tensor = rescale_for_display(input_tensor)
        input_img = to_pil_image(input_tensor)
        
        # Plot input image
        ax[0].imshow(input_img)
        ax[0].set_title('Input Image')
        ax[0].axis('off')

        # Process and plot ground truth
        target_img = targets_list[idx][0].squeeze(0).numpy()
        colored_target = apply_cityscapes_color_map(target_img)
        ax[1].imshow(colored_target)
        ax[1].set_title('Ground Truth')
        ax[1].axis('off')

        # Process and plot prediction
        prediction_img = predictions[idx][0].squeeze(0).numpy()
        colored_prediction = apply_cityscapes_color_map(prediction_img)
        ax[2].imshow(colored_prediction)
        ax[2].set_title('Prediction')
        ax[2].axis('off')

    plt.tight_layout()
    plt.show()