import dlib
import cv2
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
model_path ="mmod_human_face_detector.dat"
cnn_face_detector = dlib.cnn_face_detection_model_v1(model_path)
print("[INFO] camera sensor warming up...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
while(True):
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = cnn_face_detector(gray,1)

    for i,d in enumerate(dets):
        face = d.rect
        left = face.left()
        top = face.top()
        right = face.right()
        bottom = face.bottom()
        cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
    cv2.imshow("capture", frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
            break
vs.release()
cv2.destroyAllWindows()
