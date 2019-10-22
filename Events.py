from enum import Enum

class Events(Enum):
    NO_FACE = 'No face detected'
    EYE_CLOSE = 'Eye are close'
    MOUTH_OPEN = 'Mouth are open'

    @staticmethod
    def is_event(Event):
        return isinstance(Event, Events)

    @staticmethod
    def handle(Event):
        print(Event)
