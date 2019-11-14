import numpy as np

import cv2
import click

from Events import *
from utils import *

net_path = path.join('data', 'frozen.pb')
predictor_path = path.join('data', 'shape_predictor_68_face_landmarks.dat')


@click.command()
@click.option('--debug', is_flag=True, help='#TODO')
def main(debug):
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    net = create_net(net_path)
    detector, predictor = get_detector_and_predictor(predictor_path)

    prev = None
    while True:
        ret, raw = cap.read()
        if not ret:
            print('Video capture no work propertly')
            break

        flipped = cv2.flip(raw, +1)
        squared = cut_img_to_square(flipped)
        img = cv2.resize(squared, (432, 432))

        img, prev = make_buffer(img, prev)

        shape = get_face_points(img, debug, detector, predictor)
        if is_event(shape):
            handle(shape, img, counter_dict)
        else:
            reset_event(Events.NO_FACE, counter_dict)
            left, right = get_eyes_points(shape)
            eye_close = check_eyes(left, right, img, debug)
            if is_event(eye_close):
                handle(eye_close, img, counter_dict)
            else:
                reset_event(Events.NO_FACE, counter_dict)
                left_img, right_img = crop_eyes(img, left, right)
                predicted = predict_eye((left_img, right_img), net)
                focus = check_focus(predicted, img, debug)
                if is_event(focus):
                    handle(focus, img, counter_dict)
                else:
                    reset_event(Events.NO_FACE, counter_dict)

        cv2.imshow("Eye based drowsiness detection", img)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        if k == ord('p') and debug:
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
