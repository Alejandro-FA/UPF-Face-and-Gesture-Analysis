from .face_detector import FaceDetector, DetectionResult, BoundingBox
import imageio.v2
from facenet_pytorch import MTCNN
import MyTorchWrapper as mtw


class MTCNNDetector(FaceDetector):
    def __init__(self, use_gpu: bool=False, output_size: int=128, thresholds: tuple[float, float, float]=[0.6, 0.7, 0.7]) -> None:
        super().__init__()
        device = mtw.get_torch_device(use_gpu=use_gpu, debug=False)
        self.mtcnn = MTCNN(image_size=output_size, post_process=False, keep_all=True, device=device, thresholds=thresholds)

    def detect_faces(self, image: imageio.v2.Array) -> list[DetectionResult]:
        boxes, probs = self.mtcnn.detect(image, landmarks=False)
        if boxes is None or probs is None:
            return []

        results = []
        for bbox, prob in zip(boxes, probs):
            bbox = BoundingBox.from_coords(*bbox.astype(int))
            results.append(DetectionResult(prob, bbox))

        return results


    def save(file_path: str) -> None:
        return NotImplementedError()