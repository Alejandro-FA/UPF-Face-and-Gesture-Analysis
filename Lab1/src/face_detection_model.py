import cv2 as cv
import math


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

    def has_overlap(self, bbox2: 'BoundingBox', threshold) -> bool:
        # Intersection box
        x1 = max(self.x1, bbox2.x1)
        y1 = max(self.y1, bbox2.y1)
        x2 = min(self.x1 + self.width, bbox2.x1 + bbox2.width)
        y2 = min(self.y1 + self.height, bbox2.y1 + bbox2.height)
        # Areas
        int_Area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        total_Area = self.get_area() + bbox2.get_area() - int_Area
        
        return int_Area / total_Area > threshold

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



class FaceDetectionModel:
    def __init__(
        self,
        haar_face_model,
        lbp_face_model,
        eyes_model,
        smile_model,
        upper_body_model,
        profile_model,
        nose_model,
        mouth_model,
        eye_pair_model,
    ) -> None:
        self.face_haarcascade = haar_face_model
        self.face_lbpcascade = lbp_face_model
        self.eyes_cascade = eyes_model
        self.smile_cascade = smile_model
        self.upper_body = upper_body_model
        self.profile_lbpcascade = profile_model
        self.nose_cascade = nose_model
        self.mouth_cacade = mouth_model
        self.eye_pair = eye_pair_model

    def detect_faces(self, image) -> list[tuple[int, int, int, int]]:
        frame_gray = self.preprocess(image)
        base_image = ROI(frame_gray)
        OVERLAP_THRESHOLD = 0.1

        # -- Detect faces
        detected_faces = []
        detected_profiles = []

        
        for box in self.detect_elements(self.profile_lbpcascade, base_image):
            eyesROI = ROI(frame_gray, box)
            eyes = self.detect_elements(self.eyes_cascade, eyesROI)
            if len(eyes) > 0:
                detected_profiles.append(box.get_coords())
        

        faces = self.detect_elements(self.face_lbpcascade, base_image)
        for box in faces:
            faceROI = ROI(frame_gray, box, margin=0.75)
            haar_faces = self.detect_elements(self.face_haarcascade, faceROI, scaleFactor=1.05, minNeighbors=4)

            if len(haar_faces) > 0:
                largest_face = self.__get_largest_faces(haar_faces, 1)[0]
                detected_faces.append(largest_face.get_coords())
            else:
                eyesROI = ROI(frame_gray, box)
                smileROI = ROI(frame_gray, box)
                bodyROI = ROI(frame_gray, box, margin=2)

                if len(self.detect_elements(self.eyes_cascade, eyesROI, scaleFactor=1.05)) > 0 or \
                len(self.detect_elements(self.smile_cascade, smileROI, scaleFactor=1.05)) > 0 or \
                len(self.detect_elements(self.upper_body, bodyROI, scaleFactor=1.05)) > 0  or \
                len(self.detect_elements(self.nose_cascade, smileROI, scaleFactor=1.05)) > 0:
                    adjusted_box = box.get_resized(base_image.width(), base_image.height(), margin=0.25)
                    detected_faces.append(adjusted_box.get_coords())

        
        # FIXME: move apart
        results = detected_faces
        for profile in detected_profiles:
            has_overlap = False
            for face in detected_faces:
                if self.__overlap(profile, face, OVERLAP_THRESHOLD):
                    has_overlap = True
                    break
            
            if not has_overlap:
                results.append(profile)

            # return self.__get_largest_faces(detected_faces, 2)
        results = self.__remove_overlaps2(results, OVERLAP_THRESHOLD)
        
        #TODO: keep working on rotations
        if len(faces) == 0:
            rotation_angles = [10, 15, 20, 25, 30, -10, -15, -20, -25, -30]
            rotated_images = [(self.preprocess(self.__rotate_image(image, angle)), angle) for angle in rotation_angles]
            for frame, angle in rotated_images:
                faces = self.face_haarcascade.detectMultiScale(frame_gray)
                added = False
                rotated_face = None
                for (x, y, w, h) in faces:
                    rotated_face = self.__get_box(x, y, w, h, frame, 0)
                    added = True
                if added:
                    rotated_face_ROI = frame[
                        rotated_face[1]:rotated_face[1] + rotated_face[3],
                        rotated_face[0]:rotated_face[0] + rotated_face[2]
                    ]
                    lbp_faces = self.face_lbpcascade.detectMultiScale(rotated_face_ROI, scaleFactor=1.05)
                    profile_rotated = self.profile_lbpcascade.detectMultiScale(rotated_face_ROI, scaleFactor=1.05)
                    if len(lbp_faces) > 0 or \
                    len(profile_rotated) > 0:
                        results.append(rotated_face)
                    break
                    
        return results


    def preprocess(self, image):
        try:
            frame_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        except:
            frame_gray = image
        frame_gray = cv.equalizeHist(frame_gray)
        return frame_gray
    

    def detect_elements(self, model, roi: ROI, overlap_threshold=1.0, scaleFactor=1.1, minNeighbors=3) -> list[BoundingBox]:
        elements = model.detectMultiScale(roi.get_frame(), scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        bounding_boxes = []
        for x, y, w, h in elements:
            bbox = BoundingBox(roi.bounding_box.x1 + x, roi.bounding_box.y1 + y, w, h)
            bounding_boxes.append(bbox)

        bounding_boxes = self.__remove_overlaps(bounding_boxes, overlap_threshold)
        return bounding_boxes


    def __get_largest_faces(self, faces: list[BoundingBox], n: int) -> list[BoundingBox]:
        largest_faces = sorted(
            faces,
            key= lambda x: (x.width) * (x.height),
            reverse=True
        )[0:n]
        
        return largest_faces
    

    def __get_box(self, x, y, w, h, image, margin: float) -> tuple[int, int, int, int]:
        img_h = image.shape[0]
        img_w = image.shape[1]
        return (
            self.__clamp(x - (0 + margin) * w, 0, img_w),
            self.__clamp(y - (0 + margin) * h, 0, img_h),
            self.__clamp(x + (1 + margin) * w, 0, img_w),
            self.__clamp(y + (1 + margin) * h, 0, img_h)
        )

    def __clamp(self, x: float, min_threshold, max_threshold) -> int:
        x = int(x)
        x = min(x, max_threshold)
        x = max(x, min_threshold)
        return x
    
    def __area_box(self, box: tuple[int, int, int, int]) -> float:
        return (box[2] - box[0]) * (box[3] - box[1])


    def __overlap(self, face1, face2, threshold) -> bool:
        f = face1
        g = face2
        # Intersection box
        x1 = max(f[0], g[0])
        y1 = max(f[1], g[1])
        x2 = min(f[2], g[2])
        y2 = min(f[3], g[3])
        # Areas
        int_Area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        total_Area = (f[2] - f[0]) * (f[3] - f[1]) + (g[2] - g[0]) * (g[3] - g[1]) - int_Area
        
        return int_Area / total_Area > threshold
    
    def __remove_overlaps(self, faces: list[BoundingBox], threshold: float) -> list[BoundingBox]:
        non_overlapped = []
        
        for i in range(len(faces)):
            has_overlap = False
            for j in range(i, len(faces)):
                if faces[i].has_overlap(faces[j], threshold) and (faces[i].get_area() < faces[j].get_area()):
                    has_overlap = True
                    break

            # No overlap
            if not has_overlap:
                non_overlapped.append(faces[i])
                    
        return non_overlapped

    def __remove_overlaps2(self, faces: list[tuple[int, int, int, int]], threshold: float) -> list[tuple[int, int, int, int]]:
        non_overlapped = []
        
        for i in range(len(faces)):
            has_overlap = False
            for j in range(i, len(faces)):
                if self.__overlap(faces[i], faces[j], threshold) and (self.__area_box(faces[i]) < self.__area_box(faces[j])):
                    has_overlap = True
                    break

            # No overlap
            if not has_overlap:
                non_overlapped.append(faces[i])
                    
        return non_overlapped
    
    def __rotate_image(self, image, angle):    
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)

        rotated_image = cv.warpAffine(image, rotation_matrix, (width, height))
        
        return rotated_image