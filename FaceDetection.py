import cv2 as cv
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv.VideoCapture(0)
detector = FaceDetector()
meshdetector = FaceMeshDetector(maxFaces=1)

while(True):
    rec, frame = cap.read()
    frame, bbox = detector.findFaces(frame)
    frame, faces = meshdetector.findFaceMesh(frame)
    cv.imshow('frame', frame)
    keyexit = cv.waitKey(5) & 0xFF
    if keyexit == 27:
        break

cv.destroyAllWindows()
cap.release()
