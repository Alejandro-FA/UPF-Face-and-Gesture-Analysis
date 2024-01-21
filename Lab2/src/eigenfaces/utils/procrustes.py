from utils.landmarks import Landmarks
import numpy as np
from scipy.spatial import procrustes

class Procrustes:
    THRESHOLD = 0.006
    def __init__(self, landmarks_list: list[Landmarks]) -> None:
        self.landmarks = np.dstack([landmark.as_matrix() for landmark in landmarks_list])
        
    
    def gpa(self):
        current_distance = 0
        
        mean_shape = self.landmarks[:, :, 0]
        num_shapes = self.landmarks.shape[2]
        new_shapes = np.zeros(self.landmarks.shape)
        
        while True:
            new_shapes[:, :, 0] = mean_shape
            for i in range(1, num_shapes):
                _, new_shape, _ = procrustes(mean_shape, self.landmarks[:, :, i])
                new_shapes[:, :, i] = new_shape
            
            new_mean = np.mean(new_shapes, axis=2)
            new_distance = np.sqrt(np.sum(np.square(mean_shape - new_mean)))
            print(f"new distance: {new_distance}")
            if new_distance < self.THRESHOLD:
                break
            
            _, new_mean, _ = procrustes(mean_shape, new_mean)
            
            mean_shape = new_mean
            current_distance = new_distance
            
        new_shapes[:, 0, :] *= 2444
        new_shapes[:, 1, :] *= 1718
        
        
        return new_shapes
            