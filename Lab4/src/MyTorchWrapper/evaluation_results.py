import numpy as np


def _average_results(results: list[float], batch_sizes: list[int], num_epochs: int = 1) -> list[float]:
    results_per_epoch = len(results) // num_epochs
    if len(results) % num_epochs != 0:
        raise ValueError(f"Length of results ({results_per_epoch}) is not a multiple of num_epochs ({num_epochs}).")
    
    averages = []
    for i in range(num_epochs):
        partial_results = results[i * results_per_epoch:(i + 1) * results_per_epoch]
        partial_batch_sizes = batch_sizes[i * results_per_epoch:(i + 1) * results_per_epoch]
        a = np.sum(np.multiply(partial_results, partial_batch_sizes))
        b = np.sum(partial_batch_sizes)
        averages.append(a / b)

    return averages



class BasicResults:
    """Class used to store and retrieve the results of a training or testing process.
    """    

    def __init__(self) -> None:
        self.loss = []
        self.batch_sizes = []


    def _log_loss(self, loss: float) -> None:
        """Adds a loss result to the results history.

        Args:
            loss (float): The loss value to store.
            batch_size (int): Size of the batch from which the result has been
            obtained. Used to accurately average the results.
        """        
        self.loss.append(loss)


    def _log_batch_size(self, batch_size: int) -> None:
        """Adds a batch size to the history.

        Args:
            batch_size (int): Size of the batch to log.
            Used to accurately average the results.
        """        
        self.batch_sizes.append(batch_size)
    

    def as_dict(self) -> dict[str, float | list[float]]:
        """Create a dictionary representation of all the results.

        Returns:
            Dict[str, Union[float, List[float]]]: Dictionary representation of the results.
        """
        return {'loss': self.loss if len(self.loss) > 1 else self.loss[0]}

    
    def average(self, num_epochs: int = 1) -> 'BasicResults':
        """Averages the results over a given number of epochs.

        Args:
            num_epochs (int): Number of epochs to average over.

        Returns:
            BasicResults: A new BasicResults instance with the averaged results.
        """        
        results = self.__class__()
        results.loss = _average_results(self.loss, self.batch_sizes, num_epochs)
        samples_per_epoch = len(self.loss) // num_epochs
        results.batch_sizes = [np.sum(self.batch_sizes[i * samples_per_epoch:(i + 1) * samples_per_epoch]) for i in range(num_epochs)]
        return results
    

    def append(self, results: 'BasicResults') -> None:
        """Appends the results of another BasicResults instance to this one.

        Args:
            results (BasicResults): Results to append.
        """
        self.loss += results.loss
        self.batch_sizes += results.batch_sizes



class AccuracyResults(BasicResults):
    def __init__(self) -> None:
        super().__init__()
        self.accuracy = []


    def _log_accuracy(self, accuracy: float) -> None:
        """Adds an accuracy result to the results history.

        Args:
            accuracy (float): The accuracy value to store.
        """        
        self.accuracy.append(accuracy)


    def average(self, num_epochs: int = 1) -> BasicResults:
        results = super().average(num_epochs)
        results.accuracy = _average_results(self.accuracy, self.batch_sizes, num_epochs)
        return results


    def as_dict(self) -> dict[str, float | list[float]]:
        dict = super().as_dict()
        dict['accuracy'] = self.accuracy if len(self.accuracy) > 1 else self.accuracy[0]
        return dict


    def append(self, results: 'AccuracyResults') -> None:
        super().append(results)
        self.accuracy += results.accuracy

