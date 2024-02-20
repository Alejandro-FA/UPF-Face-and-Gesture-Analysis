import numpy as np
import scipy
import matplotlib.pyplot as plt
from .evaluation_results import BasicResults, AccuracyResults
from .train import Trainer
from typing import Literal


class Plotter:
    """
    A class for plotting training and validation results.

    Args:
        training_results (BasicResults): The training results.
        validation_results (BasicResults): The validation results.
        num_epochs (int): The number of epochs used for training.
    """
    def __init__(self, trainer: Trainer, training_results: BasicResults, validation_results: BasicResults) -> None:
        self.training_results = training_results
        self.validation_results = validation_results
        self.num_epochs = trainer.epochs

    
    def plot_evaluation_per_batch(self, figsize: tuple[int, int]=(10, 5)) -> plt.Figure:
        """
        Plot the evaluation (loss and accuracy) at each step/batch during training.

        Args:
            figsize (tuple[int, int], optional): The figure size. Defaults to (10, 5).

        Returns:
            matplotlib.figure.Figure: The plotted figure.
        """
        num_subplots = 2 if isinstance(self.training_results, AccuracyResults) else 1
        fig, axes = plt.subplots(1, num_subplots, figsize=figsize)
        plt.suptitle("Training loss and accuracy evolution at each step / batch", fontsize=14, fontweight="bold")

        losses = self.training_results.loss
        window = min(100, len(losses))

        # Plot loss evolution with the training dataset per batch
        axes[0].plot(losses, label="Real losses", color="#68CDFF")
        
        # Plot smoothed losses evolution with the training dataset per batch
        smoothed_losses = scipy.signal.savgol_filter(losses, window, 5)
        axes[0].plot(smoothed_losses, label="Smoothed losses", color="blue")
        
        axes[0].set_title("Training loss at each step")
        axes[0].set_xlabel("Steps (number of forward passes)")
        axes[0].set_ylabel("Loss")
        axes[0].set_ylim((0, None))
        axes[0].legend()
        axes[0].grid()
        
        if num_subplots == 2:
            accuracies = self.training_results.accuracy

            # Plot loss evolution with the training dataset per batch
            axes[1].plot(accuracies, label="Real accuracies", color="#68CDFF")
            
            # Plot smoothed losses evolution with the training dataset per batch
            smoothed_accuracies = scipy.signal.savgol_filter(accuracies, window, 5)
            axes[1].plot(smoothed_accuracies, label="Smoothed accuracies", color="blue")
            
            axes[1].set_xlim(None)
            axes[1].set_title("Training accuracy at each step")
            axes[1].set_xlabel("Steps (number of forward passes)")
            axes[1].set_ylabel("Accuracy (%)")
            axes[1].set_ylim((0, 100))
            axes[1].legend()
            axes[1].grid()

        return fig
    

    def plot_train_validation_loss(self, figsize: tuple[int, int]=(10, 5)) -> plt.Figure:
        """
        Plot the training and validation loss.

        Args:
            figsize (tuple[int, int], optional): The figure size. Defaults to (10, 5).

        Returns:
            matplotlib.figure.Figure: The plotted figure.
        """
        train_losses = self.training_results.average(self.num_epochs).as_dict()["loss"]
        validation_losses = self.validation_results.average(self.num_epochs).as_dict()["loss"]
        return self.__plot_train_validation_result(train_losses, validation_losses, "loss", figsize)
    

    def plot_train_validation_accuracy(self, figsize: tuple[int, int]=(10, 5)) -> plt.Figure:
        """
        Plot the training and validation accuracy.

        Args:
            figsize (tuple[int, int], optional): The figure size. Defaults to (10, 5).

        Returns:
            matplotlib.figure.Figure: The plotted figure.

        Raises:
            ValueError: If both training and validation results are not accuracy results.
        """
        if not isinstance(self.training_results, AccuracyResults) or not isinstance(self.validation_results, AccuracyResults):
            raise ValueError("Both training and validation results must be accuracy results")
        
        train_accuracies = self.training_results.average(self.num_epochs).as_dict()["accuracy"]
        validation_accuracies = self.validation_results.average(self.num_epochs).as_dict()["accuracy"]
        return self.__plot_train_validation_result(train_accuracies, validation_accuracies, "accuracy", figsize)
    

    def __plot_train_validation_result(self, training_results: list[float], validation_results: list[float], result_name: Literal["loss", "accuracy"], figsize: tuple[int, int]=(10, 5)) -> plt.Figure:
        """
        Plot the training and validation result (loss or accuracy).

        Args:
            training_results (list[float]): The training results.
            validation_results (list[float]): The validation results.
            result_name (Literal["loss", "accuracy"]): The name of the result (loss or accuracy).
            figsize (tuple[int, int], optional): The figure size. Defaults to (10, 5).

        Returns:
            matplotlib.figure.Figure: The plotted figure.
        """
        fig = plt.figure(figsize=figsize)
        epochs = np.arange(1, self.num_epochs + 1)
        
        plt.plot(epochs, training_results, label=f"Training {result_name}", color="blue")
        plt.plot(epochs, validation_results, label=f"Validation {result_name}", color="red")
        plt.title(f"Training and validation {result_name} evolution", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid()
        plt.xlabel("epochs")
        plt.ylabel(result_name)
        return fig
    