import numpy as np
import time

import cv2
import click
import glob
from imutils.video import FPS

from Events import *
from utils import *

net_path = path.join('data', 'frozen.pb')
predictor_path = path.join('data', 'shape_predictor_68_face_landmarks.dat')


@click.command()
@click.option('--debug', is_flag=True, help='#TODO')
def main(debug):
    imgs = glob.glob('im/*.png')

    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    net = create_net(net_path)
    #net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    detector, predictor = get_detector_and_predictor(predictor_path)

    fps = FPS().start()
    
    #_ = vs.read()
    _ = predict_eye(np.zeros((2,128,128,3), dtype=np.uint8), net) 
    init_pin()

    print('[INFO] Started')
    inf_time = 0
    inf_count = 0
    try:
        for im in imgs:
            raw = cv2.imread(im) 
            flipped = cv2.flip(raw, +1)
            squared = cut_img_to_square(flipped)
            img = cv2.resize(squared, (432, 432))

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
                    reset_event(Events.EYE_CLOSE, counter_dict)
                    left_img, right_img = crop_eyes(img, left, right)
                    
                    
                    predicted, tims = predict_eye((left_img, right_img), net)
                    inf_time += tims
                    inf_count += 1

                    focus = check_focus(predicted, img, debug)

                    if is_event(focus):
                        handle(focus, img, counter_dict)
                    else:
                        reset_event(Events.BAD_FOCUS, counter_dict)

            #cv2.imshow("Eye based drowsiness detection", img)
            fps.update()

    except KeyboardInterrupt:
        print("[INFO] STOP")
    finally:
        fps.stop()
        clear_pins()
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        print("[INFO] len of test: {}".format(len(imgs)))
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] inference time: {:2f}".format(inf_time))
        print("[INFO] inference count {}".format(inf_count))
        #cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
