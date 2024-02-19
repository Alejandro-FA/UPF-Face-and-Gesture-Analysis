import re
import os
import torch
from torch import nn
from typing import Optional


class _PathManager:
    """Auxiliary class in charge of resolving the path of input and output files.
    """    

    model_ext = ".ckpt" # Extension of model files
    summary_ext = ".txt" # Extension of summary files
    filename_pattern = re.compile(r"model_(\d+)(?:-(\d+))?.ckpt")

    def __init__(self, models_dir: str) -> None:
        """
        Args:
            models_dir (str): Folder path where the models are stored.
        """                
        self.models_dir = models_dir


    def get_model_name(self, model_id: int, epoch: Optional[int] = None) -> str:
        """Given a model id, it returns the model name.

        Args:
            model_id (int): Identification number of the model.
            epoch (Optional[int]): Optional epoch number.

        Returns:
            str: The model name.
        """
        suffix = f"-{epoch}" if epoch is not None else ""        
        return "model_" + str(model_id) + suffix
    

    def get_model_path(self, model_id: int, epoch: Optional[int] = None) -> str:
        """Given a model id, it returns the path of its corresponding model file (.ckpt)

        Args:
            model_id (int): Identification number of the model.
            epoch (Optional[int]): Optional epoch number.

        Returns:
            str: The path of the model file.
        """        
        return os.path.join(self.models_dir, self.get_model_name(model_id, epoch) + self.model_ext)
    

    def get_summary_path(self, model_id: int, epoch: Optional[int] = None) -> str:
        """Given a model id, it returns its corresponding training summary file path.

        Args:
            model_id (int): Identification number of the model.
            epoch (Optional[int]): Optional epoch number.

        Returns:
            str: The path of the summary file.
        """        
        return self.models_dir + self.get_model_name(model_id, epoch) + self.summary_ext



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
        files = os.listdir(self.storage_dir)
        models = list(filter(lambda name: self._path_manager.model_ext in name, files))
        indices = [int(self._path_manager.filename_pattern.search(model).group(1)) for model in models]
        return 1 if not indices else max(indices) + 1        
    

    def exists(self, model_id: int, epoch: Optional[int] = None) -> bool:
        """Checks if a given model already exists.

        Args:
            model_id (int): Identification number of the model to search.
            epoch (Optional[int]): Optional epoch number.

        Returns:
            bool: Whether a model has already been saved with the specified id and epoch or not.
        """
        model_path = self._path_manager.get_model_path(model_id, epoch)        
        return os.path.exists(model_path)


    def save(self, model: nn.Module, model_id: int, epoch: Optional[int] = None) -> None:
        """Given a torch model, it saves it in the storage_dir with the
        provided model_id.

        Args:
            model (nn.Module): Neural Network model to store.
            model_id (int): Identification number with which to store the model.
            epoch (Optional[int]): Optional epoch number.
        """        
        file_path = self._path_manager.get_model_path(model_id, epoch)
        torch.save(model.state_dict(), file_path)


    def load(self, model: nn.Module, model_id: int, epoch: Optional[int] = None) -> None:
        """Given a torch model and a model_id, it loads all the parameters stored
        in the model file (identified with model_id) inside the model.

        Args:
            model (nn.Module): Neural Network model in which to store the parameters. It must have the appropriate architecture.
            model_id (int): Identification number of the model to load.
            epoch (Optional[int]): Optional epoch number.
        """        
        file_path = self._path_manager.get_model_path(model_id, epoch)
        model.load_state_dict(torch.load(file_path))


    def save_summary(self, summary_content: str, model_id: int, epoch: Optional[int] = None) -> None:
        """Given a training summary, it stores its results in a file.

        Args:
            summary_content (str): The content of the summary.
            model_id (int): Identification number of the model from which the the summary has been obtained.
            epoch (Optional[int]): Optional epoch number.
        """        
        file_path = self._path_manager.get_summary_path(model_id, epoch)
        with open(file_path, "w") as results_txt:
            results_txt.write(summary_content)
    