class BoundingBox:
    def __init__(self, x, y, w, h):
        self.x1 = x
        self.y1 = y
        self.width = w
        self.height = h
    
    def get_resized(self, max_x2, max_y2, margin):
        x1 = max(self.x1 - int(margin * self.width), 0)
        y1 = max(self.y1 - int(margin * self.height), 0)
        x2 = min(self.x1 + int((1 + margin) * self.width), max_x2)
        y2 = min(self.y1 + int((1 + margin) * self.height), max_y2)

        width = x2 - x1
        height = y2 - y1
        return BoundingBox(x1, y1, width, height)
    
    def get_coords(self) -> list[int, int, int, int]:
        return [self.x1, self.y1, self.x1 + self.width, self.y1 + self.height]

    def get_area(self):
        return self.width * self.height

    def overlap(self, bbox2: 'BoundingBox') -> bool:
        # Intersection box
        x1 = max(self.x1, bbox2.x1)
        y1 = max(self.y1, bbox2.y1)
        x2 = min(self.x1 + self.width, bbox2.x1 + bbox2.width)
        y2 = min(self.y1 + self.height, bbox2.y1 + bbox2.height)

        # Areas
        int_Area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        # total_Area = self.get_area() + bbox2.get_area() - int_Area
        # return int_Area / total_Area > threshold
        smallest_area = min(self.get_area(), bbox2.get_area())
        return int_Area / smallest_area # NOTE: We only consider smallest bounding box



class ROI:
    """
    Class used to represent Regions Of Interest of an image
    """
    def __init__(self, base_image, bounding_box: BoundingBox=None, margin:float=0) -> None:
        self.base_image = base_image
        base_image_w = base_image.shape[1]
        base_image_h = base_image.shape[0]

        if bounding_box is None:
            self.bounding_box = BoundingBox(0, 0, base_image_w, base_image_h)
        else:
            self.bounding_box = bounding_box
        
        if margin != 0:
            self.bounding_box = bounding_box.get_resized(base_image_w, base_image_h, margin)

        self.__roi = self.base_image[
            self.bounding_box.y1 : self.bounding_box.y1 + self.bounding_box.height,
            self.bounding_box.x1 : self.bounding_box.x1 + self.bounding_box.width
        ]

    def get_frame(self):
        return self.__roi
    
    def width(self):
        return self.bounding_box.width
    
    def height(self):
        return self.bounding_box.height
    


class OverlapFilter:
    def __init__(self, threshold=0.5) -> None:
        """
        Args:
            threshold (float, optional): Determines the ratio of intersaction
            necessary to consider an overlap. Defaults to 0.5 (the intersaction
            area must be 50% of the smallest BoundingBox or higher).
        """        
        self.threshold = threshold


    def filter(self, bounding_boxes: list[BoundingBox]) -> list[BoundingBox]:
        non_overlapped = []
        
        for i in range(len(bounding_boxes)):
            has_overlap = False
            box_i = bounding_boxes[i]
            for j in range(i, len(bounding_boxes)):
                box_j = bounding_boxes[j]
                if (box_i.overlap(box_j) > self.threshold) and (box_i.get_area() < box_j.get_area()):
                    has_overlap = True
                    break

            if not has_overlap:
                non_overlapped.append(box_i)
                    
        return non_overlapped
    

    def filter_pair(self, boxes1: list[BoundingBox], boxes2: list[BoundingBox]) -> list[BoundingBox]:
        """The same as filter, but prioritizes elements from boxes1. Elements
        from boxes2 are only added if they do not overlap with any element of
        boxes1.

        Returns:
            list[BoundingBox]: A combined list with non-overlapped bounding
            boxes from the two input lists.
        """
        results = boxes1
        for b2 in boxes2:
            has_overlap = False
            for b1 in boxes1:
                if b1.overlap(b2) > self.threshold:
                    has_overlap = True
                    break
            
            if not has_overlap:
                results.append(b2)

        return results