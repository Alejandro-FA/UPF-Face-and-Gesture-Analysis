import torch
from torch import nn
import torchinfo
from .train import Trainer
from .evaluation_results import EvaluationResults
from typing import Optional


def training_summary(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    trainer: Trainer,
    validation_results: Optional[EvaluationResults] = None,
) -> str:  
    """Build a performance summary report for future reference.

    This function generates a summary report that includes information about
    the model, optimizer, trainer, and validation results. The summary can be
    printed to the screen or written to a file.

    Args:
        model (nn.Module): The model from which we want to get an architecture summary.
        optimizer (torch.optim.Optimizer): The optimizer used during training.
        trainer (Trainer): The trainer instance used to train the model.
        validation_results (Optional[BasicResults]): The evaluation results obtained during testing.
            Used to record the performance of the model. (default: None)

    Returns:
        str: The summary content.
    """
    batch, _ = next(iter(trainer.train_data_loader))
    model_stats = torchinfo.summary(model, input_size=batch.shape, device=trainer.device, verbose=0)

    results = validation_results.average(per_epoch=True).as_dict() if validation_results is not None else None
    results = f"Validation results: {results}\n" if results is not None else ""
    summary = (
        results
        + f"Loss function used: {trainer.evaluation.loss_criterion}\n"
        + f"Epochs: {trainer.epochs}\n"
        + f"Optimizer: {optimizer}\n"
        + str(model_stats)
    )

    return summary


def get_torch_device(use_gpu: bool = True, debug: bool = False) -> torch.device:
    """Obtains a torch device in which to perform computations

    Args:
        use_gpu (bool, optional): Use GPU if available. Defaults to True.
        debug (bool, optional): Whether to print debug information or not. Defaults to False.

    Returns:
        torch.device: Device in which to perform computations
    """    
    device = torch.device(
        'cuda:0' if use_gpu and torch.cuda.is_available() else
        'mps' if use_gpu and torch.backends.mps.is_available() else
        'cpu'
    )
    if debug: print("Device selected:", device)
    return device


def get_model_params(model: nn.Module) -> int:
    """Computes the number of trainable parameters of a model.

    Args:
        model (nn.Module): Model to evaluate.

    Returns:
        int: Number of parameters of the model
    """  
    params = 0
    for p in model.parameters():
        params += p.numel()
    return params
