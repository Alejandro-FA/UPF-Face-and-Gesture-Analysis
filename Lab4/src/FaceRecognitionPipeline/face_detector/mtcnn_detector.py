from .face_detector import FaceDetector
import imageio.v2
from facenet_pytorch import MTCNN
import MyTorchWrapper as mtw
import torch


class MTCNNDetector(FaceDetector):
    def __init__(self, use_gpu=False, output_size=128, thresholds=[0.6, 0.7, 0.7]) -> None:
        super().__init__()
        device = mtw.get_torch_device(use_gpu=use_gpu, debug=False)
        self.detector = MTCNN(image_size=output_size, post_process=True, keep_all=True, device=device, thresholds=thresholds)

    def detect_faces(self, image: imageio.v2.Array | list[imageio.v2.Array]) -> list[torch.Tensor]:
        
        boxes, probs = self.detector.detect(image)
        if boxes is None or probs is None:
            return []

        results = []
        for bbox, prob in zip(boxes, probs):
            # print(bbox)
            results.append(DetectionResult(prob, *bbox))

        return results


    def save(file_path: str) -> None:
        return NotImplementedError()