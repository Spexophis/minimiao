import asyncio
import heapq
import inspect
import threading

from PyQt6.QtCore import QObject, QMetaObject, Qt, pyqtSlot


class EventCallback:
    def __init__(self, callback, priority=0):
        self.callback = callback
        self.priority = priority

    def __lt__(self, other):
        # Higher priority runs first
        return self.priority > other.priority


class EventBus(QObject):
    """
    A thread-safe event bus with priority, async support, and main-thread Qt emission.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = threading.RLock()
        self._subscribers = {}  # event_name: [EventCallback, ...]
        self._main_thread_id = threading.get_ident()

    def subscribe(self, event_name, callback, priority=0):
        with self._lock:
            if event_name not in self._subscribers:
                self._subscribers[event_name] = []
            heapq.heappush(self._subscribers[event_name], EventCallback(callback, priority))

    def unsubscribe(self, event_name, callback):
        with self._lock:
            if event_name in self._subscribers:
                self._subscribers[event_name] = [
                    ec for ec in self._subscribers[event_name] if ec.callback != callback
                ]

    def unsubscribe_all(self, event_name=None):
        with self._lock:
            if event_name:
                self._subscribers.pop(event_name, None)
            else:
                self._subscribers.clear()

    def get_subscribers(self, event_name=None):
        with self._lock:
            if event_name:
                return [ec.callback for ec in self._subscribers.get(event_name, [])]
            return {e: [ec.callback for ec in cb] for e, cb in self._subscribers.items()}

    def publish(self, event_name, data=None, on_main_thread=True):
        # Gather callbacks in priority order (high first)
        with self._lock:
            callbacks = list(self._subscribers.get(event_name, []))
        callbacks = sorted(callbacks)  # __lt__ is reversed for priority

        # Emit to each callback
        for event_cb in callbacks:
            cb = event_cb.callback
            # Marshal to main thread if requested and not already there
            if on_main_thread and threading.get_ident() != self._main_thread_id:
                # Qt: Use invokeMethod to run in main (GUI) thread
                QMetaObject.invokeMethod(
                    self,
                    "_invoke_callback",
                    Qt.ConnectionType.QueuedConnection,
                    cb, data
                )
            else:
                self._call_callback(cb, data)

    @pyqtSlot(object, object)
    def _invoke_callback(self, cb, data):
        self._call_callback(cb, data)

    @staticmethod
    def _call_callback(cb, data):
        if inspect.iscoroutinefunction(cb):
            # If async, schedule on event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.create_task(cb(data))
        else:
            try:
                cb(data)
            except Exception as e:
                print(f"[EventBus] Exception in subscriber: {e}")

    def publish_async(self, event_name, data=None, on_main_thread=True):
        # For async publisher (does not block)
        threading.Thread(target=self.publish, args=(event_name, data, on_main_thread), daemon=True).start()
