# -*- coding: utf-8 -*-
"""
PyAV(FFmpeg) 기반 저지연 프레임 그랩버
- dshow(Windows) 카메라: "video=<디바이스 이름>" 형태
- 항상 최신 프레임만 제공 (내부 디코드 스레드가 큐를 비우면서 덮어씀)
- 출력: OpenCV 없이도 동작하지만, downstream에서 OpenCV가 필요하면 frame은 BGR ndarray로 이미 변환됨
"""
import threading, time
from typing import Optional, Tuple
import av
import numpy as np

class PyAvLatestGrabber:
    """
    container.decode()를 별도 스레드에서 계속 돌리면서 최신 프레임 1장만 유지.
    read()는 즉시 최신을 복사해 반환 (버퍼 지연 최소화).
    """
    def __init__(
        self,
        device_name_or_url: str,
        *,
        backend: str = "dshow",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        pixel_format: str = "mjpeg",  # Iriun/USB cam에서 MJPEG로 강제 권장
        timeout_open: float = 5.0
    ):
        """
        device_name_or_url:
          - dshow: "video=Iriun Webcam" 또는 "video=USB2.0 Camera"
          - rtsp/http 등도 그대로 URL로 가능(rtsp://..., http://...)
        """
        self._stopped = False
        self._last_frame = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._exc: Optional[Exception] = None

        # PyAV open
        self._container = av.open(
            device_name_or_url,
            format=backend,  # 예: "dshow", "rtsp"
            timeout=timeout_open,
            options={
                # 공통
                "framerate": str(fps),
                "video_size": f"{width}x{height}",
                # dshow 전용(가능시 적용)
                "pixel_format": pixel_format,  # mjpeg / yuyv422 ...
                # RTSP를 쓸 경우엔 'rtsp_transport':'tcp', 'flags':'low_delay' 등을 줄 수 있음
            }
        )

        # 가장 첫 번째 비디오 스트림
        self._vstream = None
        for s in self._container.streams:
            if s.type == "video":
                self._vstream = s
                break
        if self._vstream is None:
            raise RuntimeError("No video stream found in source")

        # 저지연 디코딩 옵션
        # B-frames 최소화(가능한 경우), 버퍼 줄이기
        self._vstream.thread_type = "AUTO"

        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _loop(self):
        try:
            for frame in self._container.decode(self._vstream):
                if self._stopped:
                    break
                # ndarray (BGR24)로 즉시 변환 → downstream에서 바로 사용
                img = frame.to_ndarray(format="bgr24")

                # 최신 프레임만 유지 (덮어쓰기)
                with self._lock:
                    self._last_frame = img
                    self._cond.notify_all()
        except Exception as e:
            with self._lock:
                self._exc = e
                self._cond.notify_all()

    def read(self, wait_latest: bool = False, wait_ms: int = 0) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Returns: (ok, frame_bgr)
        - wait_latest=True일 때, 새 프레임이 올 때까지 최대 wait_ms 대기
        """
        with self._lock:
            if self._exc is not None:
                raise self._exc
            if self._last_frame is not None:
                return True, self._last_frame.copy()
            if not wait_latest:
                return False, None
            # 최초 프레임 대기
            end = time.time() + (wait_ms / 1000.0)
            while self._last_frame is None and self._exc is None:
                remaining = end - time.time()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=remaining)
            if self._exc is not None:
                raise self._exc
            if self._last_frame is None:
                return False, None
            return True, self._last_frame.copy()

    def release(self):
        self._stopped = True
        try:
            # flush decoder
            self._container.close()
        except Exception:
            pass
