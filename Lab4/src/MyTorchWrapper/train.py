import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from .evaluation import BasicEvaluation
from .evaluation_results import BasicResults
from .test import Tester
from .io import IOManager
from typing import Optional


class Trainer:
    """Class to train a Neural Network model."""

    def __init__(
        self,
        evaluation: BasicEvaluation,
        epochs: int,
        train_data_loader: DataLoader,
        validation_data_loader: DataLoader,
        io_manager: IOManager,
        device: torch.device,
    ) -> None:
        """
        Args:
            evaluation (BasicEvaluation): evaluation instance with the desired
            methods of evaluation, including the loss. See the BasicEvaluation
            class for more details.
            epochs (int): number of training epochs
            data_loader (DataLoader): Data with which to train the torch model
            device (torch.device): device in which to perform the computations
        """
        self.evaluation = evaluation
        self.epochs = epochs
        self.train_data_loader = train_data_loader
        self.tester = Tester(evaluation, validation_data_loader, device)
        self.device = device
        self.iomanager = io_manager
        self.model_id = self.iomanager.next_id_available()


    def train(self, model: nn.Module, optimizer: torch.optim.Optimizer, seed_value: Optional[int] = 10, verbose: bool = True) -> tuple[BasicResults, BasicResults]:
        """Train the torch model with the training data provided.

        Args:
            model (nn.Module): the model to train
            optimizer (torch.optim.Optimizer): optimization algorithm to use
            seed_value (int | None, optional): Set a manual random seed to get consistent results.
            If it is None, then no manual seed is set. Defaults to 10.
            verbose (bool, optional): Whether to print training progress or not. Defaults to True.

        Returns:
            dict[str, list[float]]: Performance evaluation of the training
            process at each step.
        """
        if seed_value is not None: torch.manual_seed(seed_value) # Ensure repeatable results
        model.train() # Set the model in training mode
        model.to(self.device)

        total_steps = len(self.train_data_loader)
        feedback_step = round(total_steps / 3) + 1
        results = self.evaluation.create_results()
        validation_results = self.evaluation.create_results()

        # Take the next available model id to save checkpoints
        self.model_id = self.iomanager.next_id_available()
        print(f"Training model {self.model_id} with {len(self.train_data_loader)} batches per epoch...")

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
