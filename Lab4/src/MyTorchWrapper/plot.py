import numpy as np
import scipy
import matplotlib.pyplot as plt
from .evaluation_results import EvaluationResults


class Plotter:
    """
    A class for plotting training and validation results.

    Args:
        training_results (BasicResults): The training results.
        validation_results (BasicResults): The validation results.
        num_epochs (int): The number of epochs used for training.
    """
    def __init__(self, training_results: EvaluationResults, validation_results: EvaluationResults) -> None:
        self.training_results = training_results
        self.validation_results = validation_results
        self.num_epochs = self.training_results.num_epochs
        assert self.num_epochs == self.validation_results.num_epochs, "The number of epochs in the training and validation results must be the same."

    
    def plot_evaluation_per_batch(self, figsize: tuple[int, int]=(10, 5)) -> plt.Figure:
        """
        Plot the evaluation results at each step/batch during training.

        Args:
            figsize (tuple[int, int], optional): The figure size. Defaults to (10, 5).

        Returns:
            matplotlib.figure.Figure: The plotted figure.
        """
        num_subplots = len(self.training_results.metrics)
        fig, axes = plt.subplots(1, num_subplots, figsize=figsize)
        plt.suptitle("Evaluation metrics evolution at training each step / batch", fontsize=14, fontweight="bold")

        for i, metric in enumerate(self.training_results.metrics):
            results = self.training_results[metric]
            window = min(100, len(results))

            # Plot loss evolution with the training dataset per batch
            axes[i].plot(results, label=f"Real {metric}", color="#68CDFF")
            
            # Plot smoothed losses evolution with the training dataset per batch
            smoothed_results = scipy.signal.savgol_filter(results, window, 5)
            axes[i].plot(smoothed_results, label=f"Smoothed {metric}", color="blue")
            
            axes[i].set_title(f"Training {metric} at each step")
            axes[i].set_xlabel("Steps (number of forward passes)")
            axes[i].set_ylabel(metric)
            axes[i].set_ylim((0, None))
            axes[i].set_xlim(None)
            axes[i].legend()
            axes[i].grid()

        return fig
    

    def plot_train_validation_comparison(self, metric: str, figsize: tuple[int, int]=(10, 5)) -> plt.Figure:
        """
        Plot the training and validation result (loss or accuracy).

        Args:
            metric (str): The name of the metric to plot.
            figsize (tuple[int, int], optional): The figure size. Defaults to (10, 5).

        Returns:
            matplotlib.figure.Figure: The plotted figure.
        """
        training_results = self.training_results.average(per_epoch=True)[metric]
        validation_results = self.validation_results.average(per_epoch=True)[metric]

        fig = plt.figure(figsize=figsize)
        epochs = np.arange(1, self.num_epochs + 1)
        
        plt.plot(epochs, training_results, label=f"Training {metric}", color="blue")
        plt.plot(epochs, validation_results, label=f"Validation {metric}", color="red")
        plt.title(f"Training and validation {metric} evolution", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid()
        plt.xlabel("epochs")
        plt.ylabel(metric)
        return fig
    