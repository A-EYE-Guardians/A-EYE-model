#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
calib_intrinsics.py

RTSP 스트림에서 체스보드 이미지를 캡처(grab)하고,
캡처된 이미지들로 카메라 내부 파라미터(K, dist)를 추정(solve).

사용 예:
  # 캡처
  python calib_intrinsics.py grab --url rtsp://IP:8554/cam --save_dir calib/intrinsics/eye --chess_cols 9 --chess_rows 6

  # 해결
  python calib_intrinsics.py solve --img_glob "calib/intrinsics/eye/*.jpg" --chess_cols 9 --chess_rows 6 --square_m 0.0245 --out_json calib/intrinsics/intrinsic_eye.json
"""

import os, glob, time, json, argparse
import numpy as np
import cv2

def find_corners(img, cols, rows, use_sb=True):
    """
    체스보드 내부 코너 검출.
    - cols, rows: '내부 코너' 수 (예: 9x6)
    - use_sb: OpenCV의 findChessboardCornersSB 사용 여부(가능하면 더 강건)
    반환: (ok, corners) ; corners shape: (N,1,2), float32 (서브픽셀 포함)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pattern_size = (cols, rows)

    corners = None
    ok = False

    if use_sb and hasattr(cv2, "findChessboardCornersSB"):
        # SB 버전 시도
        try:
            flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
            ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
        except Exception:
            ok = False

    if not ok:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
        if ok:
            # 서브픽셀 보정
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), term)

    return ok, corners

def build_object_points(cols, rows, square_m):
    """
    체스보드 평면상의 '3D' 오브젝트 포인트 생성(Z=0).
    (0,0), (1,0), ... (cols-1, rows-1) * square_m
    """
    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_m)
    return objp

def cmd_grab(args):
    os.makedirs(args.save_dir, exist_ok=True)
    cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise SystemExit(f"RTSP 열기 실패: {args.url}")

    auto_every = 0.8   # 자동 저장 주기(초)
    auto_on = False
    last_auto = 0.0
    count = 0

    print("[INFO] SPACE=저장, E=자동캡처 토글, ESC=종료")
    while True:
        ok, frame = cap.read()
        if not ok:
            cv2.waitKey(1); continue

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
        if k == 27:   # ESC
            break
        elif k == ord(' '):  # SPACE
            if okc:
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(args.save_dir, f"cap_{ts}.jpg")
                cv2.imwrite(path, frame)
                print("[SAVE]", path)
                count += 1
            else:
                print("[WARN] 코너 검출 실패, 저장 안 함")
        elif k == ord('e'):
            auto_on = not auto_on

    cap.release()
    cv2.destroyAllWindows()

def cmd_solve(args):
    paths = sorted(glob.glob(args.img_glob))
    if len(paths) < 8:
        raise SystemExit(f"[ERROR] 이미지가 너무 적습니다({len(paths)}장). 최소 10~20장 권장.")

    all_obj = []
    all_img = []
    gray_size = None

    objp = build_object_points(args.chess_cols, args.chess_rows, args.square_m)

    used = 0
    for p in paths:
        img = cv2.imread(p)
        if img is None: 
            continue
        okc, corners = find_corners(img, args.chess_cols, args.chess_rows)
        if not okc: 
            continue
        # 이미지 크기
        gray_size = (img.shape[1], img.shape[0])
        all_obj.append(objp.copy())
        all_img.append(corners)
        used += 1

    if used < 8:
        raise SystemExit(f"[ERROR] 코너 검출 성공 이미지가 너무 적습니다({used}장). 다양한 뷰로 더 캡처하세요.")

    # 캘리브레이션
    flags = cv2.CALIB_RATIONAL_MODEL  # k4~k6 포함(왜곡 많을 때 안정적)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj, all_img, gray_size, None, None, flags=flags, criteria=term
    )

    # 왜곡계수 길이 통일(최대 8개로 저장)
    dist = dist.reshape(-1, 1).flatten()
    if dist.size < 8:
        dist = np.pad(dist, (0, 8 - dist.size), mode='constant')

    out = {
        "image_size": [int(gray_size[0]), int(gray_size[1])],
        "board_size": [int(args.chess_cols), int(args.chess_rows)],
        "square_size_m": float(args.square_m),
        "K": K.reshape(-1).tolist(),          # 9개(row-major)
        "dist": dist.reshape(-1).tolist(),    # 8개로 패딩
        "rms": float(rms),
        "num_used": int(used),
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("[OK] saved:", args.out_json)
    print(f"  RMS reprojection error = {rms:.4f} (작을수록 좋음)")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    apg = sub.add_parser("grab", help="RTSP에서 코너 검출 미리보기 및 이미지 저장")
    apg.add_argument("--url", type=str, required=True)
    apg.add_argument("--save_dir", type=str, required=True)
    apg.add_argument("--chess_cols", type=int, required=True, help="내부 코너 수 (가로)")
    apg.add_argument("--chess_rows", type=int, required=True, help="내부 코너 수 (세로)")

    aps = sub.add_parser("solve", help="저장된 이미지로 intrinsic 추정")
    aps.add_argument("--img_glob", type=str, required=True)
    aps.add_argument("--chess_cols", type=int, required=True)
    aps.add_argument("--chess_rows", type=int, required=True)
    aps.add_argument("--square_m", type=float, required=True)
    aps.add_argument("--out_json", type=str, required=True)

    args = ap.parse_args()
    if args.cmd == "grab":
        cmd_grab(args)
    elif args.cmd == "solve":
        cmd_solve(args)

if __name__ == "__main__":
    main()
