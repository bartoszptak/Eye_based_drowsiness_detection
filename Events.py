from enum import Enum
import os
import time

import cv2


class Events(Enum):
    NO_FACE = ['Look at the road!', 1]
    EYE_CLOSE = ['Open your eyes!', 1]
    BAD_FOCUS = ['Focus on the road!', 2]


counter_dict = {
    Events.NO_FACE: None,
    Events.EYE_CLOSE: None,
    Events.BAD_FOCUS: None,
    'play_sound': None
}


def is_event(Event):
    return isinstance(Event, Events)


def reset_event(Event, counter_dict):
    counter_dict[Event] = None


def handle(Event, img, counter_dict):
    if counter_dict[Event] is None:
        counter_dict[Event] = time.time()

    local_time = time.time()
    if local_time-counter_dict[Event] > Event.value[1]:
        print(f"[ALARM] {Event.value[0]}")

        if counter_dict['play_sound'] is None or (local_time-counter_dict['play_sound']) > 2:
            #os.system(f'spd-say "{Event.value[0]}"')
            #print("[INFO] Alarm")
            counter_dict['play_sound'] = local_time
