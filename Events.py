from enum import Enum

import time
import cv2

class Events(Enum):
    NO_FACE = ['Patrz na droge!', 2]
    EYE_CLOSE = ['Otworz oczy!', 1]
    BAD_FOCUS = ['Skup się na drodze!', 5]


counter_dict = {
    Events.NO_FACE: None,
    Events.EYE_CLOSE: None,
    Events.BAD_FOCUS: None
}

def is_event(Event):
    return isinstance(Event, Events)


def handle(Event, img, counter_dict):
    if counter_dict[Event] is None:
        counter_dict[Event] = time.time()
        
    print(Event.value[1]-(time.time()-counter_dict[Event]))
    if time.time()-counter_dict[Event] > Event.value[1]:
        print('\007')
        img = cv2.putText(img, Event.value[0], (50,50), cv2.FONT_HERSHEY_SIMPLEX,  
                    1, (0,0,255), 1, cv2.LINE_AA)


def reset_event(Event, counter_dict):
    counter_dict[Event] = None

