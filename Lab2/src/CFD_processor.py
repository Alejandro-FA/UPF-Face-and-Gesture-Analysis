"""
CFD-BF-999-999-(9)-N.jpg
    ||  |   |   |   |
    ||  |   |   |   |
    ||  |   |   |   ------ Expression:
    ||  |   |   |               N - neutral
    ||  |   |   |               A - angry
    ||  |   |   |               F - fear
    ||  |   |   |               HC - happy, closed mouth
    ||  |   |   |               HO - happy, open mouth
    ||  |   |   |
    ||  |   |   |
    ||  |   |   |
    ||  |   |   |
    ||  |   |   ---------- Ethnic Signifiers (e.g., bindi, sindoor):
    ||  |   |                    1 - removed
    ||  |   |                    2 - intact Image ID
    ||  |   |
    ||  |   |
    ||  |   |
    ||  |   |
    ||  |   -------------- Image ID
    ||  |
    ||  ------------------ Model ID
    ||
    |--------------------- Gender:
    |                           F - female
    |                           M - male
    |
    |
    |
    ---------------------- Ethnicity:
                                A - Asian American
                                B - Black
                                I - Indian Asian
                                L - Latino/a
                                M - Multiracial American
                                W - White
"""

import os
import re


VALID_EXPRESSIONS = ["N", "A", "F", "HC", "HO"]
VALID_GENDER = ["F", "M"]
VALID_ETHNICITY = ["A", "B", "I", "L", "M", "W"]

class CFD_Processor:
    def __init__(self, base_path: str, expressions: list[str], genders: list[str], ethnicities: list[str]) -> None:
        self.base_path = base_path

        if self.__validate_field(expressions, VALID_EXPRESSIONS):
            self.expressions = expressions
        else:
            exit(-1)
            
        if self.__validate_field(genders, VALID_GENDER):
            self.genders = genders
        else:
            exit(-1)
        
        if self.__validate_field(ethnicities, VALID_ETHNICITY):
            self.ethicities = ethnicities
        else:
            exit(-1)
        
        self.images = []
        self.images_processed = 0
        self.images_stored = 0
    
    
    def __validate_field(self, candidates: list[str], valid: list[str]) -> bool:
        for expr in candidates:
            if expr not in valid:
                print(f"Expression {expr} is not valid. Valid expressions: {valid}")
                return False
        
        return True
            
    def process(self):
        self.__rec_process(self.base_path)
        print(f"{self.images_processed} images processed. {self.images_stored} images stored.")
    
    
    def __rec_process(self, file_name: str):
        """
        Processes the directory indicated by self.base_path and stores the paths of the candidate images to self.images
        """
        # print(f"Processing: {file_name}")
        if os.path.isdir(file_name):
            # print("\tDIRECTORY")
            for file in os.listdir(file_name):
                self.__rec_process(os.path.join(file_name, file))
        else:
            # We have found a file. Check if it meets the required characteristics and store it
            # print("\tIMAGE")
            if self.__process_image_name(file_name.split("/")[-1]):
                self.images_stored += 1
                self.images.append(file_name)
            else:
                print(file_name)
                
            self.images_processed += 1

    
    def __process_image_name(self, image_name: str) -> bool:
        reg = re.compile(r"CFD-([A-Z])([A-Z])-([0-9]{0,3})-([0-9]{0,3})(?:-([0-9]))?-([A-Z]{0,2}).jpg")
        matches = re.match(reg, image_name)
        if matches:
            ethnicity = matches.group(1)
            if ethnicity not in VALID_ETHNICITY: return False
            
            gender = matches.group(2)
            if gender not in VALID_GENDER: return False

            expression = matches.group(6)
            if expression not in VALID_EXPRESSIONS: return False
        
            return True
        else:
            return False
                    



processor = CFD_Processor(base_path="data/CFD_Version_3.0/Images/", expressions=["N"], genders=["M", "F"], ethnicities=["A", "B", "I", "L", "M", "W"])
    
processor.process()
print(processor.images)