from face_detection_model import FaceDetectionModel
import cv2 as cv

class ModelFactory:
    def get_model(self, save_logs=False) -> FaceDetectionModel:
        haar_face = "opencv/data/haarcascades/haarcascade_frontalface_alt.xml"
        haar_face2 = "opencv/data/haarcascades/haarcascade_frontalface_alt2.xml"
        haar_face3 = "opencv/data/haarcascades/haarcascade_frontalface_alt_tree.xml"
        haar_face4 = "opencv/data/haarcascades/haarcascade_frontalface_default.xml"
        lbp_face = "opencv/data/lbpcascades/lbpcascade_frontalface_improved.xml"
        eyes_path = "opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml"
        smile_path = "opencv/data/haarcascades_cuda/haarcascade_smile.xml"
        upper_body_path = "opencv/data/haarcascades_cuda/haarcascade_upperbody.xml"
        profile_path = "opencv/data/haarcascades/haarcascade_profileface.xml"
        nose_path = "opencv/data/haarcascades/haarcascade_mcs_nose.xml"
        mouth_path = "opencv/data/haarcascades/haarcascade_mcs_mouth.xml"
        eyes_pair_path = "opencv/data/haarcascades/haarcascade_mcs_eyepair_big.xml"

        face_haarcascade = cv.CascadeClassifier()
        face_haarcascade2 = cv.CascadeClassifier()
        face_haarcascade3 = cv.CascadeClassifier()
        face_haarcascade4 = cv.CascadeClassifier()
        face_lbpcascade = cv.CascadeClassifier()
        eyes_cascade = cv.CascadeClassifier()
        smile_cascade = cv.CascadeClassifier()
        upper_body = cv.CascadeClassifier()
        profile_lbpcascade = cv.CascadeClassifier()
        nose_cascade = cv.CascadeClassifier()
        mouth_cacade = cv.CascadeClassifier()
        eyes_pair_cascade = cv.CascadeClassifier()
        
        try:
            face_haarcascade.load(cv.samples.findFile(haar_face))
            face_haarcascade2.load(cv.samples.findFile(haar_face2))
            face_haarcascade3.load(cv.samples.findFile(haar_face3))
            face_haarcascade4.load(cv.samples.findFile(haar_face4))
            face_lbpcascade.load(cv.samples.findFile(lbp_face))
            eyes_cascade.load(cv.samples.findFile(eyes_path))
            smile_cascade.load(cv.samples.findFile(smile_path))
            upper_body.load(cv.samples.findFile(upper_body_path))
            profile_lbpcascade.load(cv.samples.findFile(profile_path))
            nose_cascade.load(cv.samples.findFile(nose_path))
            mouth_cacade.load(cv.samples.findFile(mouth_path))
            eyes_pair_cascade.load(cv.samples.findFile(eyes_pair_path))
        except Exception as e:
            print('--(!)Error loading opencv model file')
            exit(0)

        return FaceDetectionModel(
            haar_face_model=face_haarcascade,
            haar_face_model2=face_haarcascade2,
            haar_face_model3=face_haarcascade3,
            haar_face_model4=face_haarcascade4,
            lbp_face_model=face_lbpcascade,
            eyes_model=eyes_cascade,
            smile_model=smile_cascade,
            upper_body_model=upper_body,
            profile_model=profile_lbpcascade,
            nose_model=nose_cascade,
            mouth_model=mouth_cacade,
            eye_pair_model=eyes_pair_cascade,
            save_logs=save_logs,
        )