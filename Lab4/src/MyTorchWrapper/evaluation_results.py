import numpy as np


class Result:
    """
    Represents the evaluation result for a single batch.

    Attributes:
        __result (dict[str, float]): Dictionary to store the metric name-value pairs.
        batch_size (int): The size of the batch.
    """

    def __init__(self, batch_size: int) -> None:
        self.__result: dict[str, float] = {}
        self.batch_size = batch_size
    
    def __getitem__(self, metric: str) -> float:
        """
        Get the value of a metric.

        Args:
            metric (str): The name of the metric.

        Returns:
            float: The value of the metric.
        """
        return self.__result[metric]
    
    def __setitem__(self, metric: str, value: float) -> None:
        """
        Set the value of a metric.

        Args:
            metric (str): The name of the metric.
            value (float): The value of the metric.
        """
        self.__result[metric] = value

    @property
    def metrics(self) -> list[str]:
        """
        Get the list of metrics.

        Returns:
            list[str]: The list of metrics.
        """
        return list(self.__result.keys())
    


class EvaluationResults:
    """
    Represents the evaluation results for multiple epochs.

    Attributes:
        __results (list[list[Result]]): List of lists of Result objects representing the results for each epoch and batch.
    """

    def __init__(self) -> None:
        self.__results: list[list[Result]] = []
    

    @property
    def num_epochs(self) -> int:
        """
        Get the number of epochs.

        Returns:
            int: The number of epochs.
        """
        return len(self.__results)


    @property
    def metrics(self) -> list[str]:
        """
        Get the list of metrics.

        Returns:
            list[str]: The list of metrics.
        """
        return [] if self.num_epochs == 0 else self.__results[0][0].metrics
    

    def add_batch(self, batch_results: Result) -> None:
        """
        Add the results for a single batch.

        Args:
            batch_results (Result): The results for a single batch.
        """
        if self.num_epochs == 0:
            self.__results.append([])
        else:
            assert self.__results[0][0].metrics == batch_results.metrics, "Metrics do not match."
        self.__results[-1].append(batch_results)


    def add_epoch(self, epoch_results: list[Result]) -> None:
        """
        Add the results for a single epoch.

        Args:
            epoch_results (list[Result]): The results for a single epoch.
        """
        if self.num_epochs > 0:
            assert self.__results[0][0].metrics == epoch_results[0].metrics, "Metrics do not match."
        self.__results.append(epoch_results)


    def append(self, results: 'EvaluationResults') -> None:
        """
        Append the results of another EvaluationResults object.

        Args:
            results (EvaluationResults): The EvaluationResults object to append.
        """
        if not self.num_epochs == 0:
            assert self.metrics == results.metrics, "Metrics do not match."
        self.__results.extend(results.__results)


    def __getitem__(self, metric: str) -> list[float]:
        """
        Get the values of a metric for all epochs and batches.

        Args:
            metric (str): The name of the metric.

        Returns:
            list[float]: The values of the metric for all epochs and batches.
        """
        return [res[metric] for epoch_results in self.__results for res in epoch_results]
    

    def __len__(self) -> int:
        """
        Get the number of batches for which results are available.

        Returns:
            int: The number of batches.
        """
        return sum([len(epoch_results) for epoch_results in self.__results])


    def average(self, per_epoch: bool = True) -> 'EvaluationResults':
        """
        Calculate the average results.

        Args:
            per_epoch (bool, optional): Whether to calculate the average per epoch. Defaults to True.

        Returns:
            EvaluationResults: The averaged results.
        """
        if per_epoch:
            return self.__average_per_epoch(self)
        else:
            raise NotImplementedError("Implement what happens when per_epoch is false")
        

    def as_dict(self) -> dict[str, list[list[float]]]:
        """
        Convert the results to a dictionary.

        Returns:
            dict[str, list[list[float]]]: The results as a dictionary.
        """
        dict = {}
        for key in self.metrics:
            dict[key] = self[key]
        return dict
    

    @staticmethod
    def __average_per_epoch(results: 'EvaluationResults') -> 'EvaluationResults':
        """
        Calculate the average results per epoch.

        Args:
            results (EvaluationResults): The EvaluationResults object.

        Returns:
            EvaluationResults: The averaged results per epoch.
        """
        averaged_results = EvaluationResults()

        for epoch_results in results.__results:
            batch_sizes = [res.batch_size for res in epoch_results]
            avg = Result(batch_size=np.sum(batch_sizes))

            for metric in results.metrics:
                values = [res[metric] for res in epoch_results]
                avg[metric] = np.average(values, weights=batch_sizes)

            averaged_results.add_epoch([avg])

        return averaged_results
