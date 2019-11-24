from os import path
import numpy as np

import cv2
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

from Events import Events

FACE_THRESH = 25
EYE_AR_THRESH = 0.20
FOCUS_THRESH = [0.15, 0.15]

# region general


def get_eye_aspect_ratio(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])

    return (A + B) / (2.0 * C)


def get_coords(pkts):
    def line_intersection(line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return [x, y]

    l = np.array(pkts[0], dtype=np.int32)
    r = np.array(pkts[1], dtype=np.int32)
    c_0 = line_intersection(pkts[3:5], pkts[5:7])
    c_1 = pkts[2]
    c = np.mean((c_0, c_1), axis=0, dtype=np.int32)

    cor = np.stack([l, c, r])-l

    def get_angle(point):
        a = np.array((0, 0))
        b = np.array((point[0], 0))
        c = np.array((0, point[1]))

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / \
            (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.arccos(cosine_angle)*np.sign(-point[1])

    rad = get_angle(cor[2])
    if np.isnan(rad):
        rad = 0

    def rotate(x, y, theta):
        xr = np.cos(theta)*x-np.sin(theta)*y
        yr = np.sin(theta)*x+np.cos(theta)*y
        return [xr, np.abs(yr)]

    cor[1] = rotate(*cor[1], rad)
    cor[2] = rotate(*cor[2], rad)

    cor = cor.astype(np.float32)
    cor[1][0] /= cor[2][0]
    cor[1][1] /= cor[2][0]

    return cor[1]


def check_focus(predicted, img, debug):
    left, right = predicted

    left_pkts = get_coords(left)
    right_pkts = get_coords(right)

    if debug:
        cv2.putText(img, f'GAZE L: [{left_pkts[0]:.2f},{left_pkts[1]:.2f}]', (20, 70),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f'GAZE R: [{right_pkts[0]:.2f},{right_pkts[1]:.2f}]', (20, 85),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f'THRESH: [{(0.5 -FOCUS_THRESH[0]):.2f}<x<{ (0.5+FOCUS_THRESH[0]):.2f}, y<{FOCUS_THRESH[1]:.2f}]', (20, 100),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)

    if not (0.5 - FOCUS_THRESH[0] < left_pkts[0] < 0.5 + FOCUS_THRESH[0]) or not (0.5 - FOCUS_THRESH[0] < right_pkts[0] < 0.5 + FOCUS_THRESH[0]):
        return Events.BAD_FOCUS
    elif (left_pkts[1] + right_pkts[1])/2 > FOCUS_THRESH[1]:
        return Events.BAD_FOCUS


def check_eyes(left, right, img, debug):
    L = get_eye_aspect_ratio(left)
    R = get_eye_aspect_ratio(right)

    if debug:
        cv2.putText(img, f'EYE: L: {L:.2f} R: {R:.2f} THRESH: >{EYE_AR_THRESH:.2f}', (20, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)

    if (L+R)/2 < EYE_AR_THRESH:
        return Events.EYE_CLOSE


def test_draw_face_points(image, shape):
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)


def make_buffer(cor, prev, buff_size=3):
    if prev is None:
        prev = np.stack([cor for _ in range(buff_size)])

    buffer = np.empty((buff_size, *cor.shape))
    buffer[:buff_size-1] = prev[1:]
    buffer[buff_size-1] = cor
    prev = buffer

    sw = sum(buffer[i]*(i+1)
             for i in range(buff_size))/sum(range(buff_size+1))

    return np.array(sw, dtype=np.uint8), buffer
# endregion

# region dlib


def get_detector_and_predictor(predictor_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    return detector, predictor


def get_face_points(image, debug, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (108,108))
    rects = detector(gray, 1)

    shape = None
    if len(rects) == 0:
        return Events.NO_FACE

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

    shape*=4
    if debug:
        test_draw_face_points(image, shape)

    if abs(shape[32][0]-shape[3][0]) < FACE_THRESH or abs(shape[15][0]-shape[36][0])-100 < FACE_THRESH:
        return Events.NO_FACE

    return shape


def get_eyes_points(shape):
    return shape[36:42], shape[42:48]  # left, right
# endregion

# region net


def create_net(graph_path):
    net = cv2.dnn.readNetFromTensorflow(graph_path)

    return net


def predict(image, net):
    blob = cv2.dnn.blobFromImages(image, scalefactor=(
        1/255), size=(128, 128), swapRB=False)
    net.setInput(blob)
    return net.forward()


def predict_eye(images, net):
    input_data = preprocessing(images)
    predicted = predict(input_data, net)
    return postprocessing(predicted)


def preprocessing(images):
    left, right = images

    def normalize(imgdata):
        imgdata = cv2.equalizeHist(imgdata)

        return imgdata

    left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    left = cv2.resize(left, (128, 90))
    left = normalize(left)

    right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    right = cv2.resize(right, (128, 90))
    right = normalize(right)

    _image = np.zeros(shape=(2, 128, 128, 1), dtype=np.uint8)

    _image[0, :90, :128, 0] = left
    _image[1, :90, :128, 0] = right

    return _image


def postprocessing(heatmaps):
    results = np.zeros((2, 7, 2))
    for i, heatmap in enumerate(heatmaps):

        masks = np.zeros((7, 2))
        for j, hm in enumerate(heatmap):
            (_, _, _, maxLoc) = cv2.minMaxLoc(hm)
            masks[j] = maxLoc

        results[i, :, :] = np.multiply(masks, 4)

    return results

# endregion

# region image


def cut_img_to_square(img):
    shape = img.shape
    center = shape[1]//2
    size = img.shape[0]//2

    return img[:, center-size:center+size, :]


def crop_eyes(image, left, right):
    left_img = image[min(left[:, 1]) - 5:max(left[:, 1]) + 5,
                     min(left[:, 0]) - 5:max(left[:, 0]) + 5]

    right_img = image[min(right[:, 1]) - 5:max(right[:, 1]) + 5,
                      min(right[:, 0]) - 5:max(right[:, 0]) + 5]

    return left_img, cv2.flip(right_img, +1)
# endregion
