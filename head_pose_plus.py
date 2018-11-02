#!/usr/bin/env python
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import cv2
import numpy as np
import dlib

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

print("[INFO] camera sensor warming up...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
time.sleep(2.0)

# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.


while (1):
    frame = vs.read()
    # show a frame
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dets = detector(gray, 0)
    # if len(dets) == 0:
    #     print("no people")
    for i in range(len(dets)):
        # compute the bounding box of the face and draw it on the
        # frame
        (bX, bY, bW, bH) = face_utils.rect_to_bb(dets[i])
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),
                      (0, 255, 0), 1)

        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, dets[i])
        shape = face_utils.shape_to_np(shape)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw each of them
       # for (i, (x, y)) in enumerate(shape):
        # cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        # cv2.putText(frame, str(i + 1), (x - 10, y - 10),
        # q           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dets[i]).parts()])
        # for idx, point in enumerate(landmarks):
        #     # 68点的坐标
        #     pos = (point[0, 0], point[0, 1])
        #     # 利用cv2.circle给每个特征点画一个圈，共68个
        #     cv2.circle(frame, pos, 1, color=(0, 255, 0))
        #
        #     # 利用cv2.putText输出1-68
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     cv2.putText(frame, str(idx + 1), pos, font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        #
        # left_eye = (landmarks[36, 0], landmarks[36, 1])
        # right_eye = (landmarks[45, 0], landmarks[45, 1])
        # nose_tip = (landmarks[30, 0], landmarks[30, 1])
        # left_mouth = (landmarks[48, 0], landmarks[48, 1])
        # right_mouth = (landmarks[54, 0], landmarks[54, 1])
        # chin = (landmarks[8, 0], landmarks[8, 1])
        left_eye = shape[36]
        right_eye = shape[45]
        nose_tip = shape[30]
        left_mouth = shape[48]
        right_mouth = shape[54]
        chin = shape[8]
        size = frame.shape

        # 2D image points. If you change the image, you need to change vector
        image_points = np.array([
            nose_tip,  # Nose tip
            chin,  # Chin
            left_eye,  # Left eye left corner
            right_eye,  # Right eye right corne
            left_mouth,  # Left Mouth corner
            right_mouth  # Right mouth corner
        ], dtype="double")

        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner

        ])

        # Camera internals

        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)

        print("Rotation Vector:\n {0}".format(rotation_vector))
        print("Translation Vector:\n {0}".format(translation_vector))

        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                         translation_vector,
                                                         camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(frame, p1, p2, (255, 0, 0), 2)

    cv2.imshow("capture", frame)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
vs.stop()
