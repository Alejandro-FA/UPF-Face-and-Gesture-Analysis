import re
import os
import torch
from torch import nn
from .evaluation_results import EvaluationResults
import pickle
import tempfile
import shutil
from typing import Optional


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


    def get_model_folder(self, model_id: int | str) -> str:
        """Given a model id, it returns the model folder path.

        Args:
            model_id (int | str): Identification number of the model | identification name of the model.

        Returns:
            str: The model folder path.
        """
        if type(model_id) == int:
            return os.path.join(self.models_dir, f"model_{model_id}")
        else:
            return os.path.join(self.models_dir, f"{model_id}")
    

    def get_model_path(self, model_id: int, epoch: int) -> str:
        """Given a model id, it returns the path of its corresponding model file (.ckpt)

        Args:
            model_id (int): Identification number of the model.
            epoch (Optional[int]): Optional epoch number.

        Returns:
            str: The path of the model file.
        """        
        return os.path.join(self.get_model_folder(model_id), f"epoch-{epoch}{self.model_ext}")
    

    def get_results_path(self, model_id: int | str) -> str:
        """Given a model id, it returns the path of its corresponding results file.

        Args:
            model_id (int | str): Identification number of the model. | Identification name of the model.

        Returns:
            str: The path of the results file.
        """    
        return os.path.join(self.get_model_folder(model_id), f"results{self.results_ext}")
    

    def get_summary_path(self, model_id: int) -> str:
        """Given a model id, it returns its corresponding training summary file path.

        Args:
            model_id (int): Identification number of the model.

        Returns:
            str: The path of the summary file.
        """        
        return os.path.join(self.get_model_folder(model_id), f"summary{self.summary_ext}")
    

    def get_last_epoch(self, model_id: int) -> int:
        """Given a model id, it returns the last epoch number available for that model.

        Args:
            model_id (int): Identification number of the model.

        Returns:
            int: The last epoch number available for that model.
        """
        model_folder = self.get_model_folder(model_id)
        if not os.path.isdir(model_folder):
            return 0
        model_files = [f for f in os.listdir(model_folder) if f.endswith(self.model_ext)]
        return len(model_files)



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
    
    
    def model_exists(self, model_name: str) -> bool:
        
        model_dirs = [m for m in os.listdir(self.storage_dir)]
        return model_name in model_dirs


    def next_id_available(self) -> int:
        """Returns the next identification number available for a model.

        Returns:
            int: The next available identification number.
        """
        model_dirs = [m for m in os.listdir(self.storage_dir) if self._path_manager.model_folder_pattern.match(m)]
        indices = [int(self._path_manager.model_folder_pattern.search(m).group(1)) for m in model_dirs]
        return 1 if not indices else max(indices) + 1        


    def save_model(self, model: nn.Module, model_id: int | str, epoch: Optional[int] = None) -> None:
        """Given a torch model, it saves it in the storage_dir with the
        provided model_id.

        Args:
            model (nn.Module): Neural Network model to store.
            model_id (int | str): Identification number with which to store the model. | Identification name with which to store the model.
            epoch (Optional[int]): Optional epoch number.
        """
        epoch = epoch if epoch is not None else self._path_manager.get_last_epoch(model_id) + 1
        os.makedirs(self._path_manager.get_model_folder(model_id), exist_ok=True)       
        file_path = self._path_manager.get_model_path(model_id, epoch)
        torch.save(model.state_dict(), file_path)
    

    def save_results(self, training_results: EvaluationResults, validation_results: EvaluationResults, model_id: int | str):
        """
        Saves the training and validation results for a specific model and epoch.

        Args:
            training_results (BasicResults): Training results to be saved.
            validation_results (BasicResults): Validation results to be saved.
            model_id (int | str): Identification number of the model. | Identification name of the model
            epoch (int, optional): Epoch number. Defaults to 1.
        """
        os.makedirs(self._path_manager.get_model_folder(model_id), exist_ok=True) 
        results_path = self._path_manager.get_results_path(model_id)

        # Create a temporary file in the same directory as the results_path
        dir_name = os.path.dirname(results_path)
        with tempfile.NamedTemporaryFile('wb', dir=dir_name, delete=False) as tmp_file:
            pickle.dump((training_results, validation_results), tmp_file)

        # Rename (move) the temporary file to the final location
        shutil.move(tmp_file.name, results_path)


    def load_model(self, model: nn.Module, model_id: int, epoch: Optional[int] = None) -> None:
        """Given a torch model and a model_id, it loads all the parameters stored
        in the model file (identified with model_id) inside the model.

        Args:
            model (nn.Module): Neural Network model in which to store the parameters. It must have the appropriate architecture.
            model_id (int): Identification number of the model to load.
            epoch (Optional[int]): Optional epoch number.
        """      
        epoch = epoch if epoch is not None else self._path_manager.get_last_epoch(model_id)
        file_path = self._path_manager.get_model_path(model_id, epoch)
        model.load_state_dict(torch.load(file_path))
        print("Model loaded from:", file_path)
        
        
    def load_results(self, model_id: int | str) -> tuple[EvaluationResults, EvaluationResults]:   
        """
        Load the results of a trained model for a specific epoch.

        Args:
            model_id (int | str): The ID of the model | The name of the model.
            epoch (int, optional): The epoch number. Defaults to 1.

        Returns:
            tuple[BasicResults, BasicResults]: A tuple containing the basic results for training and validation.
        """
        results_path = self._path_manager.get_results_path(model_id)
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
    