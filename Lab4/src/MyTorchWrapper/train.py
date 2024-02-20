import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from .evaluation import BasicEvaluation
from .evaluation_results import BasicResults
from .test import Tester
from .io import IOManager
from typing import Optional


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
        self.evaluation = evaluation
        self.epochs = epochs
        self.train_data_loader = train_data_loader
        self.tester = Tester(evaluation, validation_data_loader, device)
        self.device = device
        self.iomanager = io_manager
        self.model_id = self.iomanager.next_id_available()


    @property
    def model_name(self) -> str:
        return f"model_{self.model_id}"

   
    def train(self, model: nn.Module, optimizer: torch.optim.Optimizer, seed_value: Optional[int] = 10, verbose: bool = True) -> tuple[BasicResults, BasicResults]:
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
        # Take the next available model id to save checkpoints
        self.model_id = self.iomanager.next_id_available()
        print(f"Training {self.model_name} with {len(self.train_data_loader)} batches per epoch...")

        if seed_value is not None: torch.manual_seed(seed_value) # Ensure repeatable results
        model.train() # Set the model in training mode
        model.to(self.device)

        total_steps = len(self.train_data_loader)
        feedback_step = round(total_steps / 3) + 1
        results = self.evaluation.create_results()
        validation_results = self.evaluation.create_results()

        for epoch in range(self.epochs):
            # Train model over all batches of the dataset
            for i, (features, labels) in enumerate(self.train_data_loader):
                # Move the data to the torch device
                features = features.to(self.device)
                labels = labels.to(self.device) # FIXME: Perhaps we need to use .to(self.device, dtype=torch.long)
                
                outputs = model(features)  # Forward pass
                loss = self.evaluation(outputs, labels, results)  # Evaluation

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose and ((i + 1) % feedback_step == 0 or i + 1 == total_steps):
                    print(
                        "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                            epoch + 1, self.epochs, i + 1, total_steps, loss.item()
                        )
                    )

            # Save the model checkpoint
            self.iomanager.save(model, self.model_id, epoch + 1)

            # Test the model with the validation dataset
            epoch_validation_results = self.tester.test(model)
            validation_results.append(epoch_validation_results)

        return results, validation_results
