from abc import ABC, abstractmethod
import imageio.v2




class BoundingBox:
    def __init__(self, origin_x: int, origin_y: int, width: int, height: int) -> None:
        self.x0 = origin_x
        self.y0 = origin_y
        self.width = width
        self.height = height


    @property
    def x1(self) -> int:
        return self.x0 + self.width - 1
    

    @property
    def y1(self) -> int:
        return self.y0 + self.height - 1


    @property
    def center(self) -> tuple[int, int]:
        return (self.x0 + self.width // 2, self.y0 + self.height // 2)
    

    def fit_to_image(self, image: imageio.v2.Array) -> 'BoundingBox':
        image_width = image.shape[1]
        image_height = image.shape[0]

        return BoundingBox.from_coords(
            max(0, self.x0),
            max(0, self.y0),
            min(image_width - 1, self.x1),
            min(image_height - 1, self.y1)
        )

        
    def get_coords(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


    def get_area(self) -> int:
        return self.width * self.height


    def overlap(self, bbox2: 'BoundingBox') -> float:
        # Intersection box
        x0 = max(self.x0, bbox2.x0)
        y0 = max(self.y0, bbox2.y0)
        x1 = min(self.x1, bbox2.x1)
        y1 = min(self.y1, bbox2.y1)

        # Areas
        int_Area = max(0, (x1 - x0)) * max(0, (y1 - y0))
        smallest_area = min(self.get_area(), bbox2.get_area())
        return int_Area / smallest_area # NOTE: We only consider smallest bounding box to compute the percentage of overlap
    

    def copy(self) -> 'BoundingBox':
        return BoundingBox(self.x0, self.y0, self.width, self.height)
    

    @staticmethod
    def from_coords(x0, y0, x1, y1) -> 'BoundingBox':
        return BoundingBox(x0, y0, x1 - x0 + 1, y1 - y0 + 1)




class DetectionResult:
    def __init__(self, probability: float, bounding_box: BoundingBox) -> None:
        self.probability: float = probability
        self.bounding_box: BoundingBox = bounding_box
    
    def __str__(self) -> str:
        box = self.bounding_box
        str = f"""
        Probability: {self.probability}
        Bounding box: ({box.x0}, {box.y0}), ({box.width}, {box.height})
        """
        return str




class FaceDetector(ABC):
    def __call__(self, image: imageio.v2.Array | list[imageio.v2.Array]) -> list[DetectionResult] | list[list[DetectionResult]]:
        """
        Detects faces in a single image or a batch of images.

        Args:
            image (imageio.v2.Array | list[imageio.v2.Array]): A single image or a batch of images.

        Returns:
            list[DetectionResult] | list[list[DetectionResult]]: A list of DetectionResult objects if a single image is provided,
            or a list of lists of DetectionResult objects if a batch of images is provided.
        """

        if not isinstance(image, list):
            image = [image]

        det_results = self.detect_faces(image)
        self.__fit_to_image(det_results, image)
        largest = self.__get_largest_images(det_results, 2)

        if len(largest) == 1:
            return largest[0]
        return largest
    

    @abstractmethod
    def save(file_path: str) -> None:
        raise NotImplementedError("Implement in the subclass.")


    @abstractmethod
    def detect_faces(self, images: list[imageio.v2.Array]) -> list[list[DetectionResult]]:
        raise NotImplementedError("Implement in the subclass.")
    

    def __get_largest_images(self, det_results: list[list[DetectionResult]], n: int) -> list[list[DetectionResult]]:
        largest_results = []
        for results in det_results:
            largest = sorted(results, key=lambda res: res.bounding_box.get_area(), reverse=True)[0:n]
            largest_results.append(largest)
        return largest_results
    
    
    def __fit_to_image(self, det_results: list[list[DetectionResult]], images: list[imageio.v2.Array]) -> None:
        for results, img in zip(det_results, images):
            for res in results:
                res.bounding_box = res.bounding_box.fit_to_image(img)
