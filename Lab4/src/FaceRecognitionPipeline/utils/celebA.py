from torch.utils.data import Dataset, DataLoader
import torchvision.transforms
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as transforms


class CelebA(Dataset):
    """
    Required class to load the CelebA dataset
    """
    DEFAULT_PATH = "data/datasets/CelebA/"
    def __init__(self, path: str = None, transform: torchvision.transforms = None) -> None:
        if path is not None:
            self.path = path
            if not self.path.endswith("/"): self.path += "/"
        else:
            self.path = CelebA.DEFAULT_PATH
        
        self.transform = transform
        self.labels: dict[str, int] = {}
        self.__load_ids()
    
    
    def __load_ids(self):
        """
        Loads the information from the ids file.
        A line has the following format:
            XXXXXX.jpg <id>
        where XXXXXX represents the image number, and <id> represents the id of the person in that image.
        """
        
        print("Loading ids...")
        labels_path = self.path + "ids.txt"
        try:
            file = open(labels_path, "r").read().strip()
        except:
            raise FileNotFoundError(f"Filt {labels_path} could not be found")

        for line in file.splitlines():
            splited_line = line.split(" ")
            img_num = splited_line[0].split(".")[0]
            id = int(splited_line[1])
            self.labels[img_num] = id
        
        print("Ids loaded!")

        
    
    
    ################################ Necerssary functions for pytorch data loader ################################
    def __getitem__(self, index):
        key = list(self.labels.keys())[index]
        image = read_image(f"{self.path}/Img/img_celeba/{key}.jpg", ImageReadMode.RGB)
        
        if self.transform is not None:
            image = self.transform(image)
        
        
        return image, self.labels[key]
    
    def __len__(self):
        return len(self.labels)


