from PyQt6.QtCore import QObject, pyqtSignal
from collections import defaultdict


class EventBus(QObject):
    eventSignal = pyqtSignal(str, object)

    def __init__(self):
        super().__init__()
        self._subscribers = defaultdict(list)
        self.eventSignal.connect(self._handle_event)

    def subscribe(self, event_name, callback):
        self._subscribers[event_name].append(callback)

    def unsubscribe(self, event_name, callback):
        self._subscribers[event_name].remove(callback)

    def publish(self, event_name, data=None):
        self.eventSignal.emit(event_name, data)

    def _handle_event(self, event_name, data):
        for cb in self._subscribers[event_name]:
            cb(data)
