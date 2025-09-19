# -*- coding: utf-8 -*-
import cv2, threading, time

class LatestFrameGrabber:
    """
    내부 버퍼에 쌓이는 프레임 대신 항상 '가장 최근 프레임'만 읽도록 하는 래퍼.
    .read() -> (ok, frame)
    """
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.latest = None
        self.alive = True
        try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while self.alive:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.002)
                continue
            with self.lock:
                self.latest = frame

    def read(self):
        with self.lock:
            if self.latest is None:
                return False, None
            return True, self.latest.copy()

    def release(self):
        self.alive = False
        try: self.t.join(timeout=0.5)
        except Exception: pass
        try: self.cap.release()
        except Exception: pass
