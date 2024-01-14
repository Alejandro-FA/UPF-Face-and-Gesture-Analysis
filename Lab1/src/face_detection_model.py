import cv2 as cv
from utils import BoundingBox, ROI, OverlapFilter


OVERLAP_THRESHOLD = 0.3


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

        self.overlap_filter = OverlapFilter(threshold=OVERLAP_THRESHOLD)

    def detect_faces(self, image) -> list[tuple[int, int, int, int]]:
        frame_gray = self.preprocess(image)
        base_image = ROI(frame_gray)

        detected_faces = []
        detected_profiles = []

        # ---------------------------------------------------------------------
        # -- Detect profile faces
        for box in self.detect_elements(self.profile_lbpcascade, base_image):
            eyesROI = ROI(frame_gray, box)
            eyes = self.detect_elements(self.eyes_cascade, eyesROI)
            # Only add them if it is also possible to find at least 1 eye
            if len(eyes) > 0:
                detected_profiles.append(box)
        
        # ---------------------------------------------------------------------
        # -- Detect frontal faces
        faces = self.detect_elements(self.face_lbpcascade, base_image)
        for box in faces:
            faceROI = ROI(frame_gray, box, margin=0.75)
            haar_faces = self.detect_elements(self.face_haarcascade, faceROI, scaleFactor=1.05, minNeighbors=4)

            # Add face if it is found both with a lbpcascade and a haarcascade
            if len(haar_faces) > 0:
                largest_face = self.__get_largest_faces(haar_faces, 1)[0]
                detected_faces.append(largest_face)

            # If not, try to find other human elements
            else:
                eyesROI = ROI(frame_gray, box)
                smileROI = ROI(frame_gray, box)
                bodyROI = ROI(frame_gray, box, margin=2)

                if len(self.detect_elements(self.eyes_cascade, eyesROI, scaleFactor=1.05)) > 0 or \
                len(self.detect_elements(self.smile_cascade, smileROI, scaleFactor=1.05)) > 0 or \
                len(self.detect_elements(self.upper_body, bodyROI, scaleFactor=1.05)) > 0  or \
                len(self.detect_elements(self.nose_cascade, smileROI, scaleFactor=1.05)) > 0:
                    adjusted_box = box.get_resized(base_image.width(), base_image.height(), margin=0.25)
                    detected_faces.append(adjusted_box)

        
        # ---------------------------------------------------------------------
        # -- Remove repeated faces FIXME: Move to the end to also apply to rotated faces
        results = self.overlap_filter.filter_pair(detected_faces, detected_profiles)
        results = [box.get_coords() for box in results]
        
        # ---------------------------------------------------------------------
        # -- Detect rotated faces
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
    

    def detect_elements(self, model, roi: ROI, overlap_threshold=0.5, scaleFactor=1.1, minNeighbors=3) -> list[BoundingBox]:
        elements = model.detectMultiScale(roi.get_frame(), scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        bounding_boxes = []
        for x, y, w, h in elements:
            bbox = BoundingBox(roi.bounding_box.x1 + x, roi.bounding_box.y1 + y, w, h)
            bounding_boxes.append(bbox)

        bounding_boxes = self.overlap_filter.filter(bounding_boxes)
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
    
    
    def __rotate_image(self, image, angle):    
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)

        rotated_image = cv.warpAffine(image, rotation_matrix, (width, height))
        
        return rotated_image