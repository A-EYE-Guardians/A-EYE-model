#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
웹캠/RTSP 체커보드 캡처(grab) + 캘리브레이션(solve).
- PyAV(FFmpeg) 기반 저지연 캡처(권장)
- src가 'video=장치명' 또는 rtsp/http면 PyAV 사용
- src가 숫자(장치 인덱스)면 OpenCV 경로로 폴백 (저지연 세팅 포함)

저장: fx/fy/cx/cy/K/dist, 재투영 통계
미리보기: 코너 검출 상태 + 자동 저장 HUD

실행 예(파워쉘 한 줄):
  python scripts\calib_intrinsics.py grab --src "video=Iriun Webcam" --av_backend dshow --pixel_format mjpeg --width 640 --height 480 --fps 30 --flip 0 --save_dir calib\intrinsics\eye --chess_cols 9 --chess_rows 6
"""

import os, glob, time, json, argparse
import numpy as np
import cv2
import av
av.logging.set_level(av.logging.DEBUG)

# OpenCV 폴백 경로(프로젝트 내 기존 유틸)
from io_open import BACKENDS, open_source
from utils_framegrabber import LatestFrameGrabber

# PyAV 저지연 그랩버
try:
    from pyav_grabber import PyAvLatestGrabber  # scripts/ 에 파일 생성
    _HAS_PYAV = True
except Exception:
    _HAS_PYAV = False


# ------------------------ 공통 유틸 ------------------------
def find_corners(img, cols, rows, use_sb=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    patt = (cols, rows)  # OpenCV는 (cols, rows)
    ok, corners = False, None
    if use_sb and hasattr(cv2, "findChessboardCornersSB"):
        try:
            flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            ok, corners = cv2.findChessboardCornersSB(gray, patt, flags=flags)
        except Exception:
            ok = False
    if not ok:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, patt, flags=flags)
        if ok:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), term)
    return ok, corners


def build_object_points(cols, rows, square_m):
    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_m)
    return objp


# ------------------------ GRAB (PyAV 우선) ------------------------
def _open_pyav_or_fallback(args):
    """
    src가 'video=장치명' 또는 URL이면 PyAV 시도, 아니면 OpenCV 폴백.
    반환: (read_fn, release_fn, flip_runtime)
      - read_fn(): (ok, frame_bgr)
      - release_fn(): None
      - flip_runtime: bool (미러 필요 시)
    """
    # 🔒 안전 정규화: 개행/탭/다중 공백 제거 (PowerShell 줄바꿈 실수 방지)
    raw = str(args.src)
    src = " ".join(raw.replace("\r", " ").replace("\n", " ").split())

    use_pyav = False
    if src.lower().startswith(("video=", "rtsp://", "http://", "https://")):
        use_pyav = True
    if use_pyav and not _HAS_PYAV:
        print("[WARN] PyAV를 import하지 못해 OpenCV 경로로 폴백합니다. (pip install av 필요)")
        use_pyav = False

    if use_pyav:
        # PyAV 경로
        grabber = PyAvLatestGrabber(
            device_name_or_url=src,
            backend=args.av_backend,        # dshow / rtsp / ...
            width=args.width or 640,
            height=args.height or 480,
            fps=int(args.fps or 30),
            pixel_format=args.pixel_format  # 'mjpeg' 권장 (안되면 'yuyv422')
        )
        def _read():
            return grabber.read(wait_latest=True, wait_ms=400)
        def _release():
            try: grabber.release()
            except Exception: pass
        return _read, _release, bool(args.flip)

    # OpenCV 폴백 (장치 인덱스 등)
    cap, flip_cv = open_source(
        src, args.backend, args.width, args.height, args.fps, args.fourcc,
        flip=args.flip, exposure=args.exposure, autofocus=args.autofocus
    )
    grabber = LatestFrameGrabber(cap)
    def _read():
        return grabber.read()
    def _release():
        try: grabber.release()
        except Exception: pass
    return _read, _release, bool(flip_cv)


def cmd_grab(args):
    os.makedirs(args.save_dir, exist_ok=True)
    read_fn, release_fn, flip = _open_pyav_or_fallback(args)

    auto_every = 0.8
    auto_on = False
    last_auto = 0.0
    count = 0
    print("[INFO] SPACE=저장, E=자동 토글, ESC=종료")
    try:
        while True:
            ok, frame = read_fn()
            if not ok:
                cv2.waitKey(1); continue
            if flip: frame = cv2.flip(frame, 1)

            okc, corners = find_corners(frame, args.chess_cols, args.chess_rows)
            vis = frame.copy()
            if okc:
                cv2.drawChessboardCorners(vis, (args.chess_cols, args.chess_rows), corners, okc)
                cv2.putText(vis, "DETECTED", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            else:
                cv2.putText(vis, "NO CORNERS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"auto={'ON' if auto_on else 'OFF'} | saved={count}", (12, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("grab_intrinsics", vis)

            now = time.time()
            if auto_on and okc and now - last_auto > auto_every:
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(args.save_dir, f"cap_{ts}.jpg")
                cv2.imwrite(path, frame)
                print("[AUTO SAVE]", path)
                last_auto = now
                count += 1

            k = cv2.waitKey(1) & 0xFF
            if k == 27: break
            elif k == ord(' '):
                if okc:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(args.save_dir, f"cap_{ts}.jpg")
                    cv2.imwrite(path, frame)
                    print("[SAVE]", path); count += 1
                else:
                    print("[WARN] 코너 검출 실패, 저장 안 함")
            elif k == ord('e'):
                auto_on = not auto_on
    finally:
        release_fn()
        cv2.destroyAllWindows()


# ------------------------ SOLVE (OpenCV 루틴) ------------------------
def cmd_solve(args):
    paths = sorted(glob.glob(args.img_glob))
    if len(paths) < 8:
        raise SystemExit(f"[ERROR] 이미지가 너무 적습니다({len(paths)}장). 20장↑ 권장.")

    objp = build_object_points(args.chess_cols, args.chess_rows, args.square_m)
    all_obj, all_img = [], []
    gray_size = None
    used = 0
    for p in paths:
        img = cv2.imread(p)
        if img is None: continue
        okc, corners = find_corners(img, args.chess_cols, args.chess_rows)
        if not okc: continue
        gray_size = (img.shape[1], img.shape[0])
        all_obj.append(objp.copy())
        all_img.append(corners.astype(np.float32))
        used += 1

    if used < 8:
        raise SystemExit(f"[ERROR] 유효 이미지가 부족({used}). 더 다양한 포즈로 캡처하세요.")

    flags = (cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_K3)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj, all_img, gray_size, None, None, flags=flags, criteria=term
    )

    # 재투영 통계
    errs=[]
    for obj, rv, tv, corners in zip(all_obj, rvecs, tvecs, all_img):
        proj,_ = cv2.projectPoints(obj, rv, tv, K, dist)
        d = np.linalg.norm(proj.reshape(-1,2) - corners.reshape(-1,2), axis=1)
        errs.append(d)
    errs = np.concatenate(errs)
    err_med = float(np.median(errs)); err_p95 = float(np.percentile(errs,95))

    out = {
        "image_size": [int(gray_size[0]), int(gray_size[1])],
        "board_size": [int(args.chess_cols), int(args.chess_rows)],
        "square_size_m": float(args.square_m),
        "K": K.reshape(-1).astype(float).tolist(),
        "fx": float(K[0,0]), "fy": float(K[1,1]),
        "cx": float(K[0,2]), "cy": float(K[1,2]),
        "dist": dist.reshape(-1).astype(float).tolist(),
        "rms": float(rms),
        "reproj_median_px": err_med,
        "reproj_p95_px": err_p95,
        "num_used": int(used),
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("[OK] saved:", args.out_json)
    print(f"RMS={rms:.3f}px  med={err_med:.3f}px  p95={err_p95:.3f}px  used={used}")


# ------------------------ CLI ------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    apg = sub.add_parser("grab", help="PyAV 우선 캡처 + 코너 미리보기/저장")
    apg.add_argument("--src", type=str, required=True,
                     help="PyAV: 'video=Iriun Webcam' / 'video=Iriun Webcam #2' / 'rtsp://...'\n"
                          "숫자(예: 1)를 주면 OpenCV 폴백 사용")

    # PyAV 전용
    apg.add_argument("--av_backend", type=str, default="dshow", help="PyAV 포맷(dshow, rtsp, ...)")
    apg.add_argument("--pixel_format", type=str, default="mjpeg", help="PyAV 캡처 포맷(mjpeg 권장)")

    # 공통
    apg.add_argument("--width", type=int, default=None)
    apg.add_argument("--height", type=int, default=None)
    apg.add_argument("--fps", type=float, default=None)
    apg.add_argument("--flip", type=int, default=0)

    # OpenCV 폴백에서만 적용
    apg.add_argument("--backend", type=str, default="dshow", choices=list(BACKENDS.keys()),
                     help="[OpenCV 폴백용] dshow/msmf 등")
    apg.add_argument("--fourcc", type=str, default=None, help="[OpenCV 폴백용] 예: MJPG")
    apg.add_argument("--exposure", type=float, default=None, help="[OpenCV 폴백용] 수동 노출값")
    apg.add_argument("--autofocus", type=int, default=None, help="[OpenCV 폴백용] 0/1")

    apg.add_argument("--save_dir", type=str, required=True)
    apg.add_argument("--chess_cols", type=int, required=True)
    apg.add_argument("--chess_rows", type=int, required=True)

    aps = sub.add_parser("solve", help="저장된 이미지로 intrinsic 캘리브레이션")
    aps.add_argument("--img_glob", type=str, required=True)
    aps.add_argument("--chess_cols", type=int, required=True)
    aps.add_argument("--chess_rows", type=int, required=True)
    aps.add_argument("--square_m", type=float, required=True)
    aps.add_argument("--out_json", type=str, required=True)
    aps.add_argument("--write_preview", type=str, default=None)

    args = ap.parse_args()
    if args.cmd == "grab":
        cmd_grab(args)
    else:
        cmd_solve(args)


if __name__ == "__main__":
    main()
