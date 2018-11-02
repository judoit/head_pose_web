#!/usr/bin/env python

import cv2
import numpy as np
import sys
import os
import dlib
import glob

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Ask the detector to find the bounding boxes of each face. The 1 in the
# second argument indicates that we should upsample the image 1 time. This
# will make everything bigger and allow us to detect more faces.

cap = cv2.VideoCapture(0)
while (1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    dets = detector(frame, 1)
    for i in range(len(dets)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, dets[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            pos = (point[0, 0], point[0, 1])
            # 利用cv2.circle给每个特征点画一个圈，共68个
            cv2.circle(frame, pos, 1, color=(0, 255, 0))

            # 利用cv2.putText输出1-68
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(idx + 1), pos, font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        left_eye = (landmarks[36, 0], landmarks[36, 1])
        right_eye = (landmarks[45, 0], landmarks[45, 1])
        nose_tip = (landmarks[30, 0], landmarks[30, 1])
        left_mouth = (landmarks[48, 0], landmarks[48, 1])
        right_mouth = (landmarks[54, 0], landmarks[54, 1])
        chin = (landmarks[8, 0], landmarks[8, 1])
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
cap.release()
cv2.destroyAllWindows()
