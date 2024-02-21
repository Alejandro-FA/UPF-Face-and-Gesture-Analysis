import re
import os
import torch
from torch import nn
from .evaluation_results import BasicResults
import pickle


class _PathManager:
    """Auxiliary class in charge of resolving the path of input and output files.
    """    
    model_ext = ".ckpt" # Extension of model files
    summary_ext = ".txt" # Extension of summary files
    results_ext = ".pkl" # Extension of results files
    model_folder_pattern = re.compile(r"model_(\d+)")

    def __init__(self, models_dir: str) -> None:
        """
        Args:
            models_dir (str): Folder path where the models are stored.
        """                
        self.models_dir = models_dir


    def get_model_folder(self, model_id: int) -> str:
        """Given a model id, it returns the model folder path.

        Args:
            model_id (int): Identification number of the model.

        Returns:
            str: The model folder path.
        """    
        return os.path.join(self.models_dir, f"model_{model_id}")
    

    def get_model_path(self, model_id: int, epoch: int = 1) -> str:
        """Given a model id, it returns the path of its corresponding model file (.ckpt)

        Args:
            model_id (int): Identification number of the model.
            epoch (Optional[int]): Optional epoch number.

        Returns:
            str: The path of the model file.
        """        
        return os.path.join(self.get_model_folder(model_id), f"epoch-{epoch}{self.model_ext}")
    

    def get_results_path(self, model_id: int, epoch: int = 1) -> str:
        """Given a model id, it returns the path of its corresponding results file.

        Args:
            model_id (int): Identification number of the model.
            epoch (Optional[int]): Optional epoch number.

        Returns:
            str: The path of the results file.
        """
        # NOTE: The epoch is not currently used. It might be used in the future        
        return os.path.join(self.get_model_folder(model_id), f"results{self.results_ext}")
    

    def get_summary_path(self, model_id: int) -> str:
        """Given a model id, it returns its corresponding training summary file path.

        Args:
            model_id (int): Identification number of the model.

        Returns:
            str: The path of the summary file.
        """        
        return os.path.join(self.get_model_folder(model_id), f"summary{self.summary_ext}")



class IOManager:
    """Saves and loads PyTorch models for future reference and use.
    """

    def __init__(self, storage_dir: str) -> None:
        """
        Args:
            storage_dir (str): Folder path where the models are stored.
        """
        if not os.path.isdir(storage_dir):
            os.makedirs(storage_dir)
        self.storage_dir = storage_dir       
        self._path_manager = _PathManager(storage_dir)


    def next_id_available(self) -> int:
        """Returns the next identification number available for a model.

        Returns:
            int: The next available identification number.
        """
        model_dirs = [m for m in os.listdir(self.storage_dir) if self._path_manager.model_folder_pattern.match(m)]
        indices = [int(self._path_manager.model_folder_pattern.search(m).group(1)) for m in model_dirs]
        return 1 if not indices else max(indices) + 1        


    def save_model(self, model: nn.Module, model_id: int, epoch: int = 1) -> None:
        """Given a torch model, it saves it in the storage_dir with the
        provided model_id.

        Args:
            model (nn.Module): Neural Network model to store.
            model_id (int): Identification number with which to store the model.
            epoch (Optional[int]): Optional epoch number.
        """
        os.makedirs(self._path_manager.get_model_folder(model_id), exist_ok=True)       
        file_path = self._path_manager.get_model_path(model_id, epoch)
        torch.save(model.state_dict(), file_path)
    

    def save_results(self, training_results: BasicResults, validation_results: BasicResults, model_id: int, epoch: int = 1):
        """
        Saves the training and validation results for a specific model and epoch.

        Args:
            training_results (BasicResults): Training results to be saved.
            validation_results (BasicResults): Validation results to be saved.
            model_id (int): Identification number of the model.
            epoch (int, optional): Epoch number. Defaults to 1.
        """
        os.makedirs(self._path_manager.get_model_folder(model_id), exist_ok=True) 
        results_path = self._path_manager.get_results_path(model_id, epoch)
        with open(results_path, "wb") as results_file:
            pickle.dump((training_results, validation_results), results_file)


    def load_model(self, model: nn.Module, model_id: int, epoch: int = 1) -> None:
        """Given a torch model and a model_id, it loads all the parameters stored
        in the model file (identified with model_id) inside the model.

        Args:
            model (nn.Module): Neural Network model in which to store the parameters. It must have the appropriate architecture.
            model_id (int): Identification number of the model to load.
            epoch (Optional[int]): Optional epoch number.
        """        
        file_path = self._path_manager.get_model_path(model_id, epoch)
        model.load_state_dict(torch.load(file_path))
        
        
    def load_results(self, model_id: int, epoch: int = 1) -> tuple[BasicResults, BasicResults]:   
        """
        Load the results of a trained model for a specific epoch.

        Args:
            model_id (int): The ID of the model.
            epoch (int, optional): The epoch number. Defaults to 1.

        Returns:
            tuple[BasicResults, BasicResults]: A tuple containing the basic results for training and validation.
        """
        results_path = self._path_manager.get_results_path(model_id, epoch)
        with open(results_path, "rb") as results_file:
            res = pickle.load(results_file)
        return res


    def save_summary(self, summary_content: str, model_id: int) -> None:
        """Given a training summary, it stores its results in a file.

        Args:
            summary_content (str): The content of the summary.
            model_id (int): Identification number of the model from which the the summary has been obtained.
        """        
        file_path = self._path_manager.get_summary_path(model_id)
        with open(file_path, "w") as results_txt:
            results_txt.write(summary_content)
    