from face_detection_model import FaceDetectionModel

class ModelFactory:
    def get_model(self) -> FaceDetectionModel:
        haar_face = "opencv/data/haarcascades/haarcascade_frontalface_alt.xml"
        lbp_face = "opencv/data/lbpcascades/lbpcascade_frontalface_improved.xml"
        eyes_path = "opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
        smile_path = "opencv/data/haarcascades_cuda/haarcascade_smile.xml"
        upper_body_path = "opencv/data/haarcascades_cuda/haarcascade_upperbody.xml"
        profile_path = "opencv/data/haarcascades/haarcascade_profileface.xml"
        nose_path = "opencv/data/haarcascades/haarcascade_mcs_nose.xml"
        mouth_path = "opencv/data/haarcascades/haarcascade_mcs_mouth.xml"
        eyes_pair = "opencv/data/haarcascades/haarcascade_mcs_eyepair_big.xml"

        return FaceDetectionModel(
            haar_face_model=haar_face,
            lbp_face_model=lbp_face,
            eyes_model=eyes_path,
            smile_model=smile_path,
            upper_body_model=upper_body_path,
            profile_model=profile_path,
            nose_model=nose_path,
            mouth_model=mouth_path,
            eye_pair_model=eyes_pair
        )