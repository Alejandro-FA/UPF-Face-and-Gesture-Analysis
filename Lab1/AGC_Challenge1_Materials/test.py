from __future__ import print_function
import cv2 as cv
import argparse
import os

def detectAndDisplay(frame, face_cascade, eyes_cascade):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 4)

        faceROI = frame_gray[y:y+h,x:x+w]
        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)

    cv.imshow('Capture - Face detection', frame)


if __name__ == '__main__':
    BASE_PATH = 'Lab1/AGC_Challenge1_Materials'
    
    face_cascade_name = 'opencv/data/haarcascades/haarcascade_frontalface_alt.xml'
    eyes_cascade_name = 'opencv/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'

    face_cascade = cv.CascadeClassifier()
    eyes_cascade = cv.CascadeClassifier()

    #-- 1. Load the cascades
    if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
        print('--(!)Error loading face cascade')
        exit(0)
    if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
        print('--(!)Error loading eyes cascade')
        exit(0)


    for file in os.listdir(f'{BASE_PATH}/TRAINING/'):
        file_path = f"{BASE_PATH}/TRAINING/{file}"
        img = cv.imread(file_path)
        
        detectAndDisplay(img, face_cascade, eyes_cascade)
       
        key_pressed = cv.waitKey(0)
        
        if key_pressed == 27:
            break
        elif key_pressed != -1:
            cv.destroyAllWindows()
        
    cv.destroyAllWindows()