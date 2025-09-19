# -*- coding: utf-8 -*-
import cv2

BACKENDS = {
    "any": 0,
    "dshow": cv2.CAP_DSHOW,          # Windows
    "msmf": cv2.CAP_MSMF,            # Windows
    "v4l2": cv2.CAP_V4L2,            # Linux
    "avfoundation": cv2.CAP_AVFOUNDATION,  # macOS
}

def open_source(src: str, backend: str="any", width: int=None, height: int=None,
                fps: float=None, fourcc: str=None, flip:int=0,
                exposure: float=None, autofocus: int=None):
    """
    src가 'rtsp://'면 FFMPEG, 아니면 웹캠(인덱스/경로).
    반환: (cap, flip)
    """
    if isinstance(src, str) and src.lower().startswith("rtsp://"):
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
    else:
        try:
            idx = int(src)
            cap = cv2.VideoCapture(idx, BACKENDS.get(backend.lower(), 0))
        except (ValueError, TypeError):
            cap = cv2.VideoCapture(src, BACKENDS.get(backend.lower(), 0))

        if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(width))
        if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if fps:    cap.set(cv2.CAP_PROP_FPS,          float(fps))
        if fourcc: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if autofocus is not None:
            try: cap.set(cv2.CAP_PROP_AUTOFOCUS, int(autofocus))
            except Exception: pass
        if exposure is not None:
            try: cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            except Exception: pass
            try: cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
            except Exception: pass

    if not cap.isOpened():
        raise SystemExit(f"[ERR] open failed: {src}")
    return cap, int(flip or 0)
