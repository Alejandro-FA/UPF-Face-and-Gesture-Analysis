from .face_detector import FaceDetector, DetectionResult, BoundingBox
import imageio.v2
import cv2


class YuNetDetector(FaceDetector):
    def __init__(self, model_onnx_path: str, threshold: float=0.9) -> None:
        super().__init__()
        self.detector = cv2.FaceDetectorYN.create(
            model=model_onnx_path,
            config="",
            input_size=(320, 320),
            score_threshold=threshold,
            nms_threshold=0.3,
            top_k=100,
        )


    def detect_faces(self, images: list[imageio.v2.Array]) -> list[list[DetectionResult]]:
        global_results = []

        for im in images:
            im_width = im.shape[1]
            im_height = im.shape[0]
            self.detector.setInputSize((im_width, im_height))
            faces = self.detector.detect(im)
            results = self.__get_detection_results(faces)
            global_results.append(results)

        return global_results
    
    
    def __get_detection_results(self, faces: tuple[int, cv2.typing.MatLike]) -> list[DetectionResult]:
        if faces[1] is None:
            return []
        
        results = []
        for face in faces[1]:
            bbox = BoundingBox(origin_x=face[0], origin_y=face[1], width=face[2], height=face[3])
            prob = face[-1]
            results.append(DetectionResult(prob, bbox))
        
        return results
