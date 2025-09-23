#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import json
import requests
from typing import List, Dict, Any, Optional


class LangGraphClient:
    """
    LangGraph FastAPI 클라이언트
      - /image/push           : 이미지 업로드(파일명/경로 반환)
      - /perception/yolo/push : YOLO 결과 업로드 (bbox는 {x1,y1,x2,y2})
      - /perception/gaze/push : 응시 좌표/거리 업로드
    """
    def __init__(self, base: str = "http://127.0.0.1:8010", timeout: float = 10.0, session_id: str = "alpha"):
        self.base = base.rstrip("/")
        self.timeout = float(timeout)
        self.session_id = session_id
        self.sess = requests.Session()

    # ---- 이미지 업로드 ----
    def push_image(self, jpg_bytes: bytes, ext: str = ".jpg") -> Dict[str, Any]:
        """
        파일 필드명: file
        폼 데이터: session_id
        응답 예:
          {
            "ok": true,
            "session_id": "alpha",
            "filename": "1758644451692_frame_abc123.jpg",
            "path": "/images/uploads/alpha/1758...jpg"
          }
        """
        url = f"{self.base}/image/push"
        files = {"file": ("frame"+ext, jpg_bytes, "image/jpeg")}
        data = {"session_id": self.session_id}
        r = self.sess.post(url, files=files, data=data, timeout=self.timeout)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            return {"raw": r.text}

    # ---- YOLO 결과 업로드 ----
    def push_yolo(
        self,
        width: int,
        height: int,
        detections: List[Dict[str, Any]],
        image_filename: Optional[str] = None,
        frame_id: Optional[str] = None,
        ts: Optional[float] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        서버 스키마:
          {
            "session_id": "alpha",
            "ts": 1730000000.123,
            "width": 1280, "height": 720,
            "image_filename": "1758644451692_frame_xxx.jpg",
            "frame_id": null,
            "detections": [
              {
                "label": "person",
                "bbox": {"x1": 10.0, "y1": 20.0, "x2": 110.0, "y2": 140.0},
                "conf": 0.92,
                "track_id": null,
                "extra": {}
              }
            ],
            "meta": { ... }
          }
        ※ 전송은 반드시 json=payload 사용 (Pydantic 422 방지)
        """
        url = f"{self.base}/perception/yolo/push"
        payload = {
            "session_id": self.session_id,
            "ts": float(ts if ts is not None else time.time()),
            "width": int(width) if width is not None else None,
            "height": int(height) if height is not None else None,
            "detections": detections,
            "frame_id": frame_id,
            "image_filename": image_filename,
            "meta": (meta or {}),
        }
        try:
            r = self.sess.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            # FastAPI 422 등일 때 상세 메시지 출력
            try:
                print("[LG] /perception/yolo/push HTTPError:", r.status_code, r.text, flush=True)
            except Exception:
                print("[LG] /perception/yolo/push HTTPError:", e, flush=True)
            return {}
        except Exception as e:
            print("[LG] /perception/yolo/push error:", e, flush=True)
            return {}

    # ---- Gaze 업로드 ----
    def push_gaze(
        self,
        gaze_xy_norm: List[float],
        focus_dist_m: Optional[float],
        hazards: Optional[List[Dict[str, Any]]] = None,
        sudden_entry: bool = False,
        ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        {
          "session_id":"alpha",
          "ts": 1730000000.123,
          "gaze_xy_norm":[0.53, 0.42],
          "focus_dist_m": 1.8,
          "sudden_entry": false,
          "hazards": []
        }
        """
        url = f"{self.base}/perception/gaze/push"
        payload = {
            "session_id": self.session_id,
            "ts": float(ts if ts is not None else time.time()),
            "gaze_xy_norm": [float(gaze_xy_norm[0]), float(gaze_xy_norm[1])],
            "focus_dist_m": (float(focus_dist_m) if (focus_dist_m is not None) else None),
            "sudden_entry": bool(sudden_entry),
            "hazards": (hazards or []),
        }
        try:
            r = self.sess.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            try:
                print("[LG] /perception/gaze/push HTTPError:", r.status_code, r.text, flush=True)
            except Exception:
                print("[LG] /perception/gaze/push HTTPError:", e, flush=True)
            return {}
        except Exception as e:
            print("[LG] /perception/gaze/push error:", e, flush=True)
            return {}
