import cv2 as cv
import math


class ROI:
    """
    Class used to represent Regions Of Interest of an image
    """
    def __init__(self, base_image, bounding_box: tuple[int, int, int, int]=None, margin=0) -> None:
        self.base_image = base_image
        self.base_image_w = base_image.shape[1]
        self.base_image_h = base_image.shape[0]

        if bounding_box is None:
            self.x0_roi = 0
            self.y0_roi = 0
            self.w_roi =  self.base_image_w
            self.h_roi =  self.base_image_h
        else:
            self.x0_roi = bounding_box[0]
            self.y0_roi = bounding_box[1]
            self.w_roi =  bounding_box[2] - bounding_box[0]
            self.h_roi =  bounding_box[3] - bounding_box[1]

        self.margin = margin
        self.__roi = None

    def get_frame(self):
        if not self.__roi:
            y_min = max(self.y0_roi - int(self.margin * self.h_roi), 0)
            y_max = min(self.y0_roi + int((1 + self.margin) * self.h_roi), self.base_image_h)
            x_min = max(self.x0_roi - int(self.margin * self.w_roi), 0)
            x_max = min(self.x0_roi + int((1 + self.margin) * self.w_roi), self.base_image_w)
            self.__roi = self.base_image[y_min : y_max, x_min : x_max]

        return self.__roi



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
        OVERLAP_THRESHOLD = 0.1

        # -- Detect faces
        faces = self.face_lbpcascade.detectMultiScale(frame_gray)
        detected_faces = []
        detected_profiles = []

        for box in self.detect_elements(self.profile_lbpcascade, ROI(frame_gray)):
            eyesROI = ROI(frame_gray, box)
            eyes = self.detect_elements(self.eyes_cascade, eyesROI)
            if len(eyes) > 0:
                detected_profiles.append(box)
        
        for (x,y,w,h) in faces:
            margin = 0.75
            y2_roi = int(y - margin * h)
            x2_roi = int(x - margin * w)
            faceROI = frame_gray[
                y2_roi : y + int((1 + margin)*h),
                x2_roi : x + int((1 + margin)*w)
            ]
            
            #-- In each face, detect eyes
            # eyes = self.eyes_cascade.detectMultiScale(faceROI)
            large_faces = self.face_haarcascade.detectMultiScale(faceROI, scaleFactor=1.05, minNeighbors=4)
            # If a face is detected with the accurate model, find a more suitable bounding box
            # print(len(large_faces))
            if len(large_faces) > 0:
                # detected_faces.append(self.__get_largest_faces(large_faces, 10))
                detected_large_faces = []
                for (x2,y2,w2,h2) in large_faces:
                    detected_large_faces.append(self.__get_box(x2 + x2_roi, y2 + y2_roi , w2, h2, image, 0))

                largest_face = self.__get_largest_faces(detected_large_faces, 1)[0]
                detected_faces.append(largest_face)
            else:
                margin = 0
                eyesROI = frame_gray[
                    y : y + h,
                    x : x + w
                ]
                smileROI = eyesROI
                bodyROI = frame_gray[
                    y - h : y + 3 * h,
                    x - 2 * w : x + 3 * w
                ]
                if len(self.eyes_cascade.detectMultiScale(eyesROI, scaleFactor=1.05)) > 0 or \
                    len(self.smile_cascade.detectMultiScale(smileROI, scaleFactor=1.05)) > 0 or \
                    len(self.upper_body.detectMultiScale(bodyROI, scaleFactor=1.05)) > 0  or \
                    len(self.nose_cascade.detectMultiScale(smileROI, scaleFactor=1.05)) > 0:
                    detected_faces.append(self.__get_box(x,y,w,h,image, 0.25))
            

            detected_faces = self.__remove_overlaps(detected_faces, OVERLAP_THRESHOLD)
        
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
        results = self.__remove_overlaps(results, OVERLAP_THRESHOLD)
        
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
    

    def detect_elements(self, model, roi: ROI, overlap_threshold=1.0, scale_factor=1.1) -> list[tuple[int, int, int, int]]:
        elements = model.detectMultiScale(roi.get_frame(), scaleFactor=scale_factor)
        bounding_boxes = []
        for x, y, w, h in elements:
            # box = FaceDetectionModel.__get_box(x, y, w, h, image, 0)
            bounding_boxes.append((x, y, x + w, y + h))

        bounding_boxes = self.__remove_overlaps(bounding_boxes, overlap_threshold)
        return bounding_boxes


    def __get_largest_faces(self, faces: list[tuple[int, int, int, int]], n: int) -> list[tuple[int, int, int, int]]:
        largest_faces = sorted(
            faces,
            key= lambda x: (x[2] - x[0]) * (x[3] - x[1]),
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
    
    def __remove_overlaps(self, faces: list[tuple[int, int, int, int]], threshold: float) -> list[tuple[int, int, int, int]]:
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