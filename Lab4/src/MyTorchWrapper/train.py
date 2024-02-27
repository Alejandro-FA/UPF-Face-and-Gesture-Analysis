import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from .evaluation import BasicEvaluation
from .evaluation_results import EvaluationResults, Result
from .test import Tester
from .io import IOManager
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import signal
import types


class Trainer:
    """
    The Trainer class is responsible for training a given model using the provided training dataset.
    It uses a validation dataset to test the performance of the model at each epoch (which can be used to check for overfitting, for example).
    It saves training checkpoints of the model using the io_manager.

    Args:
        evaluation (BasicEvaluation): An instance of the BasicEvaluation class used for evaluating the model.
        epochs (int): The number of training epochs.
        train_data_loader (DataLoader): The data loader for the training dataset.
        validation_data_loader (DataLoader): The data loader for the validation dataset.
        io_manager (IOManager): An instance of the IOManager class used for saving checkpoints of the model.
        device (torch.device): The device (CPU or GPU) on which the model will be trained.
    """

    def __init__(
        self,
        evaluation: BasicEvaluation,
        epochs: int,
        train_data_loader: DataLoader,
        validation_data_loader: DataLoader,
        io_manager: IOManager,
        device: torch.device,
    ) -> None:
        self.evaluation: BasicEvaluation = evaluation
        self.epochs: int = epochs
        self.train_data_loader: DataLoader = train_data_loader
        self.tester = Tester(evaluation, validation_data_loader, device)
        self.device: torch.device = device

        # The IOManager is used to save the model checkpoints and the training results
        self.iomanager: IOManager = io_manager
        self.model_id: int = self.iomanager.next_id_available()

        # Register the signal handler for SIGINT (Ctrl+C)
        self.stop_training: bool = False
        self.previous_signal_handler = None

        # Compute feedback step and total steps for printing progress
        self.total_steps = len(self.train_data_loader)
        self.feedback_step = round(self.total_steps / 3) + 1


    @property
    def model_name(self) -> str:
        return f"model_{self.model_id}"
    

    def catch_signal(self) -> None:
        """
        This method catches the SIGINT signal (Ctrl+C) and calls the signal_handler method.
        """
        self.stop_training = False
        self.previous_signal_handler = signal.signal(signal.SIGINT, self.signal_handler)


    def restore_signal_handler(self) -> None:
        """
        This method restores the previous signal handler for the SIGINT signal.
        """
        signal.signal(signal.SIGINT, self.previous_signal_handler)
    

    def signal_handler(self, signal_code: int, frame: types.FrameType) -> None:
        """
        This function handles the SIGINT signal (Ctrl+C). When the signal is
        received, it asks the user if they want to stop training. If the user
        answers 'yes', it sets the stop_training attribute to True, causing the
        train method to exit at the next iteration.
        
        Note: This function assumes that the train method is running in the
        main thread. If the train method is running in a different thread, the
        signal will be delivered to the main thread and won't be caught by this
        function. If you plan to use this class in a multithreaded environment,
        you'll need to modify this function to handle signals in a thread-safe
        way.
        """
        should_stop = input("\nTraining in progress. Are you sure you want to stop it? (y/n): ")
        if should_stop.lower() == 'yes' or should_stop.lower() == 'y':
            self.stop_training = True
            self.restore_signal_handler()
            print("Training will stop at the end of the current minibatch...")
        else:
            print("Resuming training...")
   

    def train(
            self,
            model: nn.Module,
            optimizer: Optimizer,
            lr_scheduler_epoch: Optional[LRScheduler] = None,
            lr_scheduler_minibatch: Optional[LRScheduler] = None,
            seed_value: Optional[int] = None,
            verbose: bool = True
        ) -> tuple[EvaluationResults, EvaluationResults]:
        """
        Trains the given model using the provided optimizer.

        Args:
            model (nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer used for training the model.
            seed_value (Optional[int], optional): The seed value for random number generation. Defaults to 10.
            verbose (bool, optional): Whether to print training progress. Defaults to True.

        Returns:
            tuple[BasicResults, BasicResults]: A tuple containing the training results and validation results.
        """
        self.catch_signal() # Catch the SIGINT signal (Ctrl+C)

        # Take the next available model id to save checkpoints
        self.model_id = self.iomanager.next_id_available()
        print(f"Training {self.model_name} with {len(self.train_data_loader)} minibatches per epoch...")

        if seed_value is not None: torch.manual_seed(seed_value) # Ensure repeatable results
        model.train() # Set the model in training mode
        model.to(self.device)

        training_results = EvaluationResults()
        validation_results = EvaluationResults()

        for epoch in range(1, self.epochs + 1):
            # Check if the user wants to stop the training
            if self.stop_training: return training_results, validation_results
            
            # Train the model for one epoch
            train_epoch_results = self.__train_epoch(model, optimizer, lr_scheduler_minibatch, epoch, verbose)
            if self.stop_training: return training_results, validation_results

            # Evaluate the performance at the current epoch
            validation_epoch_results = self.tester.test(model)

            # Save the results of the current epoch
            training_results.add_epoch(train_epoch_results)
            validation_results.append(validation_epoch_results)

            # Update the learning rate
            if lr_scheduler_epoch is not None:
                if isinstance(lr_scheduler_epoch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler_epoch.step(metrics=validation_results['loss'][-1])
                else:
                    lr_scheduler_epoch.step()

            # Save the model checkpoint and the results up to the current epoch
            self.iomanager.save_model(model, self.model_id, epoch)
            self.iomanager.save_results(training_results, validation_results, self.model_id)

        self.restore_signal_handler()
        return training_results, validation_results
    

    def __train_epoch(self, model: nn.Module, optimizer: Optimizer, lr_scheduler: Optional[LRScheduler], epoch: int, verbose: bool) -> list[Result]:
        train_epoch_results: list[Result] = []

        # Train model over all batches of the dataset
        for i, (features, labels) in enumerate(self.train_data_loader):
            if self.stop_training: return [] # We don't want to return partial results

            # Move the data to the torch device
            features = features.to(self.device)
            labels = labels.to(self.device) # FIXME: Perhaps we need to use .to(self.device, dtype=torch.long)
            
            outputs = model(features)  # Forward pass
            loss, batch_results = self.evaluation(outputs, labels)  # Evaluation
            train_epoch_results.append(batch_results)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the learning rate
            if lr_scheduler is not None:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    raise ValueError("ReduceLROnPlateau scheduler cannot be used at minibatch level")
                else:
                    lr_scheduler.step()

            # Print the progress
            if verbose and ((i + 1) % self.feedback_step == 0 or i + 1 == self.total_steps):
                current_lr = optimizer.param_groups[0]['lr']
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Current lr: {:.2e}".format(
                        epoch, self.epochs, i + 1, self.total_steps, loss.item(), current_lr
                    )
                )

        return train_epoch_results
