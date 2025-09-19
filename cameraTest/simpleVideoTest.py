import cv2, time

cam1_url = "rtsp://192.168.162.44:8554/cam"
cam2_url = "rtsp://192.168.162.68:8554/cam"

cap1 = cv2.VideoCapture(cam1_url, cv2.CAP_FFMPEG)
cap2 = cv2.VideoCapture(cam2_url, cv2.CAP_FFMPEG)
cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not (cap1.isOpened() or cap2.isOpened()):
    raise SystemExit("둘 다 안 열림")

cv2.namedWindow("cams", cv2.WINDOW_NORMAL)
last, cnt = time.time(), 0

while True:
    ok1,f1 = cap1.read() if cap1.isOpened() else (False, None)
    ok2,f2 = cap2.read() if cap2.isOpened() else (False, None)

    tiles = []
    if ok1: tiles.append(cv2.resize(f1, (640,480)))
    if ok2: tiles.append(cv2.resize(f2, (640,480)))

    if tiles:
        out = tiles[0] if len(tiles)==1 else cv2.hconcat(tiles)
        cv2.imshow("cams", out)

    cnt += int(ok1) + int(ok2)
    now = time.time()
    if now - last >= 2:
        print(f"~{cnt/(now-last):.1f} FPS (성공 프레임 합)")
        cnt = 0; last = now

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('s') and ok1:
        cv2.imwrite("snap_cam1.jpg", f1)
        if ok2: cv2.imwrite("snap_cam2.jpg", f2)

cap1.release(); cap2.release(); cv2.destroyAllWindows()
