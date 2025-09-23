#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time
import cv2
import json
import requests
from typing import List, Dict, Any, Tuple, Optional

# perception_client.py 를 같은 폴더에 두거나, 공용 패키지(common/)면 import 경로를 바꿔주세요.
from perception_client import LangGraphClient


def encode_jpeg(frame) -> Tuple[bytes, Tuple[int, int]]:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    h, w = frame.shape[:2]
    return buf.tobytes(), (w, h)


def call_yolo_detect(yolo_url: str, jpg_bytes: bytes, conf: float = 0.25, timeout: float = 5.0) -> Dict[str, Any]:
    """
    YOLO 도커/서버에 POST로 프레임을 보내 추론한다.
    예상 엔드포인트: POST {yolo_url}  files={'file':(...)}  (추가 쿼리: conf 등)
    반환 예(유연 파서가 정규화):
      {
        "detections": [
          {"label":"person","bbox":[x,y,w,h],"conf":0.92, "track_id":123, "extra": {...}},
          ...
        ]
      }
    혹은 단일 best 결과:
      {"bbox": {"x1":..,"y1":..,"x2":..,"y2":..}, "label":"...", "conf":0.87}
    """
    files = {"file": ("frame.jpg", jpg_bytes, "image/jpeg")}
    data = {"conf": str(conf)}
    r = requests.post(yolo_url, files=files, data=data, timeout=timeout)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        # 텍스트로 오면 비상
        return {"raw": r.text}


def to_std_detections(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    dets: List[Dict[str, Any]] = []

    # 케이스 A: 이미 x1,y1,x2,y2
    if isinstance(raw, dict) and isinstance(raw.get("bbox"), dict) and \
       {"x1","y1","x2","y2"} <= set(raw["bbox"].keys()):
        b = raw["bbox"]
        dets.append({
            "label": str(raw.get("label") or "object"),
            "bbox": {"x1": float(b["x1"]), "y1": float(b["y1"]),
                     "x2": float(b["x2"]), "y2": float(b["y2"])},
            "conf": float(raw.get("conf") or 0.0),
            "track_id": None,
            "extra": {}
        })
        return dets

    # 케이스 B: detections 배열 (bbox가 xywh 또는 dict{x,y,w,h})
    if isinstance(raw, dict) and isinstance(raw.get("detections"), list):
        for d in raw["detections"]:
            label = d.get("label") or d.get("class") or d.get("cls") or "object"
            if isinstance(d.get("bbox"), dict) and {"x","y","w","h"} <= set(d["bbox"].keys()):
                x = float(d["bbox"]["x"]); y = float(d["bbox"]["y"])
                w = float(d["bbox"]["w"]); h = float(d["bbox"]["h"])
                x1, y1, x2, y2 = x, y, x + max(0.0, w), y + max(0.0, h)
            else:
                # list/tuple [x,y,w,h] 또는 유사
                bbox = d.get("bbox") or d.get("xywh") or [d.get("x"), d.get("y"), d.get("w"), d.get("h")]
                x = float(bbox[0] or 0.0); y = float(bbox[1] or 0.0)
                w = float(bbox[2] or 0.0); h = float(bbox[3] or 0.0)
                x1, y1, x2, y2 = x, y, x + max(0.0, w), y + max(0.0, h)

            dets.append({
                "label": str(label),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "conf": float(d.get("conf") or d.get("confidence") or d.get("score") or 0.0),
                "track_id": d.get("track_id"),
                "extra": d.get("extra") or {}
            })
        return dets

    # 케이스 C: result 배열 (xywh)
    if isinstance(raw, dict) and isinstance(raw.get("result"), list):
        for d in raw["result"]:
            label = d.get("class_name") or d.get("label") or "object"
            xywh = d.get("xywh") or [d.get("x"), d.get("y"), d.get("w"), d.get("h")]
            x = float(xywh[0] or 0.0); y = float(xywh[1] or 0.0)
            w = float(xywh[2] or 0.0); h = float(xywh[3] or 0.0)
            dets.append({
                "label": str(label),
                "bbox": {"x1": x, "y1": y, "x2": x + max(0.0, w), "y2": y + max(0.0, h)},
                "conf": float(d.get("confidence") or d.get("score") or 0.0),
                "track_id": d.get("track_id"),
                "extra": {}
            })
        return dets

    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="OpenCV 카메라 인덱스")
    ap.add_argument("--session", default="alpha", help="세션 ID")
    ap.add_argument("--lg", default="http://127.0.0.1:8010", help="LangGraph 베이스 URL")
    ap.add_argument("--yolo", default="http://127.0.0.1:8090/detect", help="YOLO 추론 URL(POST /detect)")
    ap.add_argument("--conf", type=float, default=0.25, help="YOLO confidence")
    ap.add_argument("--interval", type=float, default=0.0, help="자동 추론 주기(초). 0이면 키보드로만 촬영")
    ap.add_argument("--show", action="store_true", help="윈도우에 카메라 미리보기 표시")
    args = ap.parse_args()

    lg = LangGraphClient(base=args.lg, session_id=args.session)
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[ERR] cannot open camera:", args.cam)
        return

    last_t = 0.0
    print("[info] Press 'r' to detect & push, 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        now = time.time()
        do_fire = False
        key = -1
        if args.show:
            cv2.imshow("yolo-forwarder", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                do_fire = True

        if args.interval > 0 and (now - last_t) >= args.interval:
            do_fire = True
            last_t = now

        if not do_fire:
            continue

        try:
            jpg, (w, h) = encode_jpeg(frame)

            # 1) YOLO 추론
            raw = call_yolo_detect(args.yolo, jpg, conf=args.conf)
            dets = to_std_detections(raw)
            print(f"[yolo] detections={len(dets)}")

            # 2) LangGraph에 이미지 업로드 → filename
            img_resp = lg.push_image(jpg)
            filename = img_resp.get("filename") or img_resp.get("data", {}).get("filename")

            # 3) YOLO 결과 푸시
            push_resp = lg.push_yolo(
                width=w,
                height=h,
                detections=dets,
                image_filename=filename,
                ts=now,
            )
            print(f"[lg] yolo_push ok kept={push_resp.get('kept')} file={filename}")

        except Exception as e:
            print("[ERR]", e)

    cap.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
