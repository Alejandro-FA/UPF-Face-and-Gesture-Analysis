class BoundingBox:
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x1 = x
        self.y1 = y
        self.width = w
        self.height = h

    @property
    def x2(self):
        return self.x1 + self.width
    
    @property
    def y2(self):
        return self.y1 + self.height
    
    def get_resized(self, max_x2, max_y2, margin) -> 'BoundingBox':
        x1 = max(self.x1 - int(margin * self.width), 0)
        y1 = max(self.y1 - int(margin * self.height), 0)
        x2 = min(self.x1 + int((1 + margin) * self.width), max_x2)
        y2 = min(self.y1 + int((1 + margin) * self.height), max_y2)

        width = x2 - x1
        height = y2 - y1
        return BoundingBox(x1, y1, width, height)
    
    def get_coords(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    def get_area(self) -> int:
        return self.width * self.height

    def overlap(self, bbox2: 'BoundingBox') -> float:
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
        return int_Area / smallest_area # NOTE: We only consider smallest bounding box