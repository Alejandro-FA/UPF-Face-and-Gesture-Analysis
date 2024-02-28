import os, shutil
import numpy as np
import sys


class OriginalDatasetSplitter:
    
    celebrity_to_id = {
        "Channing Tatum" : 1,
        "Christina Applegate" : 2,
        "Richard E. Grant" : 3,
        "S. Epatha Merkerson" : 4,
        "Farah Fath" : 5,
        "Jim Beaver" : 6,
        "Cheryl Hines" : 7,
        "Michael Vartan" : 8,
        "Hayden Christensen" : 9,
        "Laurence Fishburne" : 10,
        "Kathryn Joosten" : 11,
        "Patrick Warburton" : 12,
        "Jamie Lee Curtis" : 13,
        "Jason Sudeikis" : 14,
        "Billy Burke" : 15,
        "Robert Pattinson" : 16,
        "Melissa Claire Egan" : 17,
        "Morena Baccarin" : 18,
        "Jolene Blalock" : 19,
        "Matthew Lillard" : 20,
        "Alicia Goranson" : 21,
        "Jennie Garth" : 22,
        "Wanda De Jesus" : 23,
        "Tracey E. Bregman" : 24,
        "Tracey Gold" : 25,
        "Brendan Fraser" : 26,
        "Kellan Lutz" : 27,
        "John Travolta" : 28,
        "Pierce Brosnan" : 29,
        "Jasmine Guy" : 30,
        "Swoosie Kurtz" : 31,
        "Diego Luna" : 32,
        "Danny Glover" : 33,
        "David Cross" : 34,
        "Farrah Fawcett" : 35,
        "Paul Walker" : 36,
        "Matt Long" : 37,
        "Andy García" : 38,
        "Casey Affleck" : 39,
        "Carla Gallo" : 40,
        "James Brolin" : 41,
        "Christian Bale" : 42,
        "Nadia Bjorlin" : 43,
        "Valerie Bertinelli" : 44,
        "Alec Baldwin" : 45,
        "Tamara Braun" : 46,
        "Andy Serkis" : 47,
        "Jackson Rathbone" : 48,
        "Robert Redford" : 49,
        "Julie Marie Berman" : 50,
        "Chris Kattan" : 51,
        "Benicio del Toro" : 52,
        "Anthony Hopkins" : 53,
        "Lea Michele" : 54,
        "Jean-Claude Van Damme" : 55,
        "Adrienne Frantz" : 56,
        "Kim Fields" : 57,
        "Wendie Malick" : 58,
        "Lacey Chabert" : 59,
        "Harry Connick Jr." : 60,
        "Cam Gigandet" : 61,
        "Andrea Anders" : 62,
        "Chris Noth" : 63,
        "Cary Elwes" : 64,
        "Aisha Hinds" : 65,
        "Chris Rock" : 66,
        "Neve Campbell" : 67,
        "Susan Dey" : 68,
        "Robert Duvall" : 69,
        "Caroline Dhavernas" : 70,
        "Marilu Henner" : 71,
        "Christian Slater" : 72,
        "Kris Kristofferson" : 73,
        "Shelley Long" : 74,
        "Alan Arkin" : 75,
        "Faith Ford" : 76,
        "Jason Bateman" : 77,
        "Edi Gathegi" : 78,
        "Emile Hirsch" : 79,
        "Joaquin Phoenix" : 80
    }
    
    def __init__(self, cropped_imgs_path: str, target_dataset_path: str, annotations_path: str) -> None:
        self.cropped_images_path = cropped_imgs_path
        self.target_dataset_path = target_dataset_path
        
        if self.__valid_dir_path(self.cropped_images_path) == False:
            raise ValueError(f"Path {self.cropped_images_path} does not exist.")
        else:
            self.cropped_images_path = self.__remove_backslash(cropped_imgs_path)
        
        if self.__valid_dir_path(self.target_dataset_path) == False:
            print(f"Target directory {self.target_dataset_path} does not exist. Creating it...")
            os.makedirs(self.target_dataset_path)
        else:
            self.target_dataset_path = self.__remove_backslash(target_dataset_path)
            print(f"[WARNING] Directory {self.target_dataset_path} already exists. All of its contents will be modified.")
            shutil.rmtree(self.target_dataset_path)
            os.makedirs(self.target_dataset_path)
        
        self.annotations_path = annotations_path
        
        self.ids_count = self.__get_ids_count(cropped_imgs_path)
        self.total_images = np.sum(list(self.ids_count.values()))
        print(f"Total images: {self.total_images}")
    
    def from_cropped_to_dataset(self) -> dict[int, int]:
        """
        This method merges all the images present in the cropped_images_path into the folder specified by target_dataset_path.
        While doing so, it changes the names of the images and generates the labels path.
        
        Returns:
            A dictionary containing the number of images for each id
        """
        ids_count = {}
        total_images = 0
        annotations_file = open(f"{self.annotations_path}", "w")
        
        for directory in os.listdir(self.cropped_images_path):
            if directory not in OriginalDatasetSplitter.celebrity_to_id:
                print(f"Unknown celebrity name. Skipping directory: {directory}")
            else:
                curr_id = OriginalDatasetSplitter.celebrity_to_id[directory]
                complete_path = self.cropped_images_path + f"/{directory}"
                ids_count[curr_id] = 0
                
                for file in os.listdir(complete_path):
                    total_images += 1
                    image_name = self.__generate_image_name(total_images)
                    
                    # Copy the image to the destination path with its new name
                    shutil.copy(f"{complete_path}/{file}", f"{self.target_dataset_path}/{image_name}")
                    
                    # Write the label of the new image to the labels path
                    annotations_file.write(f"{image_name} {curr_id}\n")
                    
                    ids_count[curr_id] += 1
        
        annotations_file.close()
        
        return ids_count
    
    def __get_ids_count(self, path: str):
        ids_count = {}
        for directory in sorted(os.listdir(path)):
            if directory in OriginalDatasetSplitter.celebrity_to_id:
                curr_id = OriginalDatasetSplitter.celebrity_to_id[directory]
                ids_count[curr_id] = 0
                for file in os.listdir(f"{path}/{directory}"):
                    ids_count[curr_id] += 1
                    
        return ids_count
    
    def __generate_image_name(self, image_num):
        image_num = str(image_num)
        remaining_length = len(str(self.total_images)) - len(image_num)
        
        return "0" * remaining_length + image_num + ".jpg"
    
    def __valid_dir_path(self, path: str) -> bool:
        return os.path.isdir(path)
    
    def __remove_backslash(self, path: str):
        if path.endswith("/"): return path[0:-1]
        else: return path