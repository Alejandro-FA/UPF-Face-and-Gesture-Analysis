import cv2 as cv
from utils import BoundingBox, ROI, OverlapFilter


OVERLAP_THRESHOLD = 0.3


class FaceDetectionModel:
    def __init__(
        self,
        haar_face_model,
        haar_face_model2,
        haar_face_model3,
        haar_face_model4,
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
        self.face_haarcascade2 = haar_face_model2
        self.face_haarcascade3 = haar_face_model3
        self.face_haarcascade4 = haar_face_model4
        self.face_lbpcascade = lbp_face_model
        self.eyes_cascade = eyes_model
        self.smile_cascade = smile_model
        self.upper_body = upper_body_model
        self.profile_lbpcascade = profile_model
        self.nose_cascade = nose_model
        self.mouth_cacade = mouth_model
        self.eye_pair = eye_pair_model

        self.overlap_filter = OverlapFilter(threshold=OVERLAP_THRESHOLD)
        self.log_path = "log7.txt"
        with open(self.log_path, 'w') as file:
            pass # Delete current contents of file for clarity

    def detect_faces(self, image, image_path: str) -> list[tuple[int, int, int, int]]:
        frame_gray = self.preprocess(image)
        base_image = ROI(frame_gray)

        # ---------------------------------------------------------------------
        # -- Detect profile faces
        detected_profiles = []

        for box in self.detect_elements(self.profile_lbpcascade, base_image):
            eyesROI = ROI(frame_gray, box)
            eyes = self.detect_elements(self.eyes_cascade, eyesROI)

            # Only add them if it is also possible to find at least 1 eye
            # NOTE: Does not produce false positives in training
            # Produces 3 false negatives (images 164, 198, 515) that are not
            # re-detected with frontal faces nor rotated faces
            if len(eyes) > 0: 
                detected_profiles.append(box)
                self.__log(image_path, method_id=1, description='profile and eyes detected')

        # ---------------------------------------------------------------------
        # -- Detect frontal faces
        detected_faces = []

        faces = self.detect_elements(self.face_lbpcascade, base_image)
        for box in faces:
            faceROI = ROI(frame_gray, box, margin=0.75)
            haar_faces = self.detect_elements(self.face_haarcascade, faceROI, scaleFactor=1.05, minNeighbors=4)

            # Add face if it is found both with a lbpcascade and a haarcascade
            if len(haar_faces) > 0:
                largest_face = self.__get_largest_faces(haar_faces, 1)[0]
                detected_faces.append(largest_face)
                self.__log(image_path, method_id=2, description='lbp cascade and haar face detected')

            # If not, try to find other human elements
            # NOTE: Does not produce false positives nor false negatives in training
            else:
                eyesROI = ROI(frame_gray, box)
                smileROI = ROI(frame_gray, box)

                if len(self.detect_elements(self.eyes_cascade, eyesROI, scaleFactor=1.05)) > 0 and \
                len(self.detect_elements(self.smile_cascade, smileROI, scaleFactor=1.05)) > 0:
                    adjusted_box = box.get_resized(base_image.width(), base_image.height(), margin=0.25)
                    detected_faces.append(adjusted_box)
                    self.__log(image_path, method_id=3, description='lbp cascade and some other elements')

        # ---------------------------------------------------------------------
        # -- Detect rotated faces
        detected_rotated = []

        if len(faces) == 0:
            rotation_angles = [10, 15, 20, 25, 30, -10, -15, -20, -25, -30]
            rotated_images = [(self.preprocess(self.__rotate_image(image, angle)), angle) for angle in rotation_angles]
            for frame, angle in rotated_images:
                frame_gray = self.preprocess(frame)
                base_image = ROI(frame_gray)
                rotated_faces = self.detect_elements(self.face_haarcascade, base_image)
                face_added = False
                for curr_face in rotated_faces:
                    rotated_face_ROI = ROI(frame_gray, curr_face)
                    if len(self.detect_elements(self.face_lbpcascade, rotated_face_ROI, scaleFactor=1.05)) > 0 or \
                    len(self.detect_elements(self.profile_lbpcascade, rotated_face_ROI, scaleFactor=1.05)) > 0:
                        detected_rotated.append(curr_face)
                        face_added = True
                        self.__log(image_path, method_id=4, description='rotation added')
                if face_added:
                    break

        # ---------------------------------------------------------------------
        # -- Remove repeated faces
        results = self.overlap_filter.filter_pair(detected_rotated, detected_profiles)
        results = self.overlap_filter.filter_pair(detected_faces, results)
         
        return [box.get_coords() for box in results]  


    def preprocess(self, image):
        try:
            frame_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        except:
            frame_gray = image
        frame_gray = cv.equalizeHist(frame_gray)
        return frame_gray
    

    def detect_elements(self, model, roi: ROI, scaleFactor=1.1, minNeighbors=3) -> list[BoundingBox]:
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
    

    def __rotate_image(self, image, angle):
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)

        rotated_image = cv.warpAffine(image, rotation_matrix, (width, height))
        
        return rotated_image
    

    def __log(self, image_path: str, method_id: int, description: str):
        with open(self.log_path, 'a') as file:
            file.write(f"{image_path}: Checkpoint {method_id}. {description}\n")