import numpy as np

import cv2

from Events import *
from utils import *


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    sess, inp, outp = create_session_get_in_out(
        path.join('data', 'inference.pb'))

    while True:
        ret, raw = cap.read()
        if not ret:
            print('Video capture no work propertly')
            break

        flipped = cv2.flip(raw, +1)
        squared = cut_img_to_square(flipped)
        img = cv2.resize(squared, (432, 432))

        shape = get_face_points(img)
        if is_event(shape):
            handle(shape, squared, counter_dict)
        else:
            reset_event(Events.NO_FACE, counter_dict)
            left, right = get_eyes_points(shape)
            eye_close = check_eyes(left, right)
            if is_event(eye_close):
                handle(eye_close, squared, counter_dict)
            else:
                reset_event(Events.EYE_CLOSE, counter_dict)
                left_img, right_img = crop_eyes(img, left, right)
                predicted = predict_eye((left_img, right_img), sess, inp, outp)
                focus = check_focus(predicted)
                if is_event(focus):
                    handle(focus, squared, counter_dict)
                else:
                    reset_event(Events.BAD_FOCUS, counter_dict)

            #test_draw_face_points(img, shape)
        cv2.imshow("Eye based drowsiness detection", squared)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
