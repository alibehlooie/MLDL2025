from typing import List, Dict, Optional, Any
from torch.utils.tensorboard import SummaryWriter
import wandb
from dataclasses import dataclass
from utils.learning_rate import poly_lr_scheduler
from utils.metrics import tabular_print, fast_hist, per_class_iou
from utils.model_analysis import setup_model_device
from utils.visualization import visualize_segmentation_results


@dataclass
class TrainingMetrics:
    """Container for tracking training metrics during model training."""
    generator_loss: float = 0.0
    discriminator_loss: float = 0.0
    adversarial_loss: float = 0.0
    generator_accuracy: float = 0.0
    current_gen_lr: Optional[float] = None
    current_disc_lr: Optional[float] = None


class Callback:
    """
    Base class for all callbacks. Callbacks are used to perform actions at specific points 
    during training, validation, and testing phases.
    
    This class defines the interface that all callback classes should follow.
    Subclasses should override these methods to implement specific logging,
    monitoring, or intervention behaviors.
    """
    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        """Called at the start of training."""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of each training epoch."""
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of each training batch."""
        pass

    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        """Called at the end of training."""
        pass

    def on_validation_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of each validation batch."""
        pass

    def on_validation_begin(self, logs: Optional[Dict] = None) -> None:
        """Called at the start of validation."""
        pass

    def on_validation_end(self, logs: Optional[Dict] = None) -> None:
        """Called at the end of validation."""
        pass

    def on_test_begin(self, logs: Optional[Dict] = None) -> None:
        """Called at the start of testing."""
        pass

    def on_test_end(self, logs: Optional[Dict] = None) -> None:
        """Called at the end of testing."""
        pass


class TensorBoardCallback(Callback):
    """
    Callback for logging metrics to TensorBoard.
    
    This callback automatically logs all metrics passed to it during training
    and validation phases to TensorBoard, enabling real-time monitoring and
    visualization of the training process.
    """
    def __init__(self, log_dir: str = './logs'):
        """
        Initialize TensorBoard writer.
        
        Args:
            log_dir: Directory where TensorBoard logs will be stored
        """
        self.writer = SummaryWriter(log_dir)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Log metrics at the end of each epoch."""
        if logs:
            for key, value in logs.items():
                self.writer.add_scalar(f'epoch/{key}', value, epoch)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Log metrics at the end of each batch."""
        if logs:
            for key, value in logs.items():
                self.writer.add_scalar(f'batch/{key}', value, batch)
    
    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        """Clean up by closing the TensorBoard writer."""
        self.writer.close()


class WandBCallback(Callback):
    """
    Callback for logging metrics to Weights & Biases (wandb).
    
    This callback handles real-time logging of metrics, hyperparameters,
    and other training information to the wandb platform for experiment
    tracking and visualization.
    """
    def __init__(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        note: str = ''
    ):
        """
        Initialize wandb logging.
        
        Args:
            project_name: Name of the wandb project
            run_name: Optional name for this specific run
            config: Optional dictionary of configuration parameters to log
            note: Optional note to attach to this run
        """
        self._wandb_ = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            notes=note
        )
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Log metrics after each batch."""
        if logs:
            self._wandb_.log({**logs})
            
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Log metrics after each epoch."""
        if logs:
            self._wandb_.log({**logs})

    def on_validation_end(
        self,
        logs: Optional[Dict] = None,
        data: Optional[List[Any]] = None
    ) -> None:
        """
        Log validation metrics and per-class performance data.
        
        Args:
            logs: Dictionary of validation metrics
            data: Optional per-class performance data for detailed logging
        """
        if logs:
            self._wandb_.log(logs)
        if data:
            self._wandb_.log({"per class mIoU": wandb.Table(data=data)})

    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        """Clean up by finishing the wandb run."""
        print('Training finished. Terminating wandb logger.')
        self._wandb_.finish()


def update_training_callbacks(
    callbacks: List[Callback],
    iteration: int,
    metrics: TrainingMetrics
) -> None:
    """
    Updates all callbacks with the current training metrics.
    
    This function handles the routing of metrics to different callback types,
    ensuring proper formatting and logging for each callback implementation.
    
    Args:
        callbacks: List of callback instances to update
        iteration: Current training iteration
        metrics: TrainingMetrics instance containing current statistics
    """
    metric_dict = {
        'generator_loss': metrics.generator_loss,
        'discriminator_loss': metrics.discriminator_loss,
        'adversarial_loss': metrics.adversarial_loss,
        'generator_accuracy': metrics.generator_accuracy
    }
    
    # Add learning rates if available
    if metrics.current_gen_lr is not None:
        metric_dict['generator_lr'] = metrics.current_gen_lr
    if metrics.current_disc_lr is not None:
        metric_dict['discriminator_lr'] = metrics.current_disc_lr
        
    for callback in callbacks:
        if isinstance(callback, WandBCallback):
            # WandB handles dictionary logging directly
            callback.on_batch_end(iteration, metric_dict)
        elif isinstance(callback, TensorBoardCallback):
            # TensorBoard requires separate scalar logging
            for key, value in metric_dict.items():
                if value is not None:
                    callback.writer.add_scalar(f'batch/{key}', value, iteration)
        else:
            # Default callback handling
            callback.on_batch_end(iteration, metric_dict)


def update_validation_callbacks(
    callbacks: List[Callback],
    metrics: Dict[str, Any],
    class_data: Optional[List[Any]] = None
) -> None:
    """
    Updates callbacks with validation results.
    
    This function handles the logging of validation metrics and per-class
    performance data to the appropriate callback implementations.
    
    Args:
        callbacks: List of callback instances to update
        metrics: Dictionary containing validation metrics
        class_data: Optional per-class performance data for detailed logging
    """
    for callback in callbacks:
        if isinstance(callback, WandBCallback) and class_data is not None:
            callback.on_validation_end(logs=metrics, data=class_data)
        else:
            callback.on_validation_end(logs=metrics)