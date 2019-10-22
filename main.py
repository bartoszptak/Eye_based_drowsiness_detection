import numpy as np

import cv2

from Events import Events
from utils import *


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        ret, img = cap.read()
        assert ret, 'Video capture no work propertly'

        img = cv2.resize(img, (768,432))
        img = cut_img_to_square(img)
        img = cv2.flip(img, +1)

        shape = get_face_points(img)
        if Events.is_event(shape):
            Events.handle(shape)
            continue

        left, right = get_eyes_points(shape)
        eye_close = check_eyes(left, right)
        if Events.is_event(eye_close):
            Events.handle(eye_close)
            continue

        print(get_eye_aspect_ratio(left), get_eye_aspect_ratio(right))


        test_draw_face_points(img, shape)
        cv2.imshow("Inzynierka", img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
