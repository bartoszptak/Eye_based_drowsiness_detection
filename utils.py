from os import path

import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

from Events import Events

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    path.join('data', 'shape_predictor_68_face_landmarks.dat'))

EYE_AR_THRESH = 0.15

def cut_img_to_square(img):
    shape = img.shape
    center = shape[1]//2
    size = img.shape[0]//2

    return img[:, center-size:center+size, :]

def get_face_points(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    shape = None
    if len(rects) == 0:
        return Events.NO_FACE

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

    return shape

def get_eyes_points(shape):
    return shape[36:42], shape[42:48] # left, right

def get_eye_aspect_ratio(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])

    return (A + B) / (2.0 * C)

def check_eyes(left, right):
    L = get_eye_aspect_ratio(left)
    R = get_eye_aspect_ratio(right)

    if L < EYE_AR_THRESH and R < EYE_AR_THRESH:
        return Events.EYE_CLOSE

def crop_eyes(image, left, right):
    left_img = image[min(left[:, 1]) - 5:max(left[:, 1]) + 5,
                    min(left[:, 0]) - 5:max(left[:, 0]) + 5]

    right_img = image[min(right[:, 1]) - 5:max(right[:, 1]) + 5,
                    min(right[:, 0]) - 5:max(right[:, 0]) + 5]

    return left_img, right_img

def get_mouth_points(shape):
    return shape[49:60]

def test_draw_face_points(image, shape):
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
