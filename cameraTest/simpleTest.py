import cv2

cam1 = cv2.VideoCapture("rtsp://192.168.162.44:8554/cam", cv2.CAP_FFMPEG)
cam2 = cv2.VideoCapture("rtsp://192.168.162.68:8554/cam", cv2.CAP_FFMPEG)

print("opened:", cam1.isOpened(), cam2.isOpened())

ok1,f1 = cam1.read()
ok2,f2 = cam2.read()
print("read:", ok1, ok2)

if ok1: cv2.imwrite("cam1.jpg", f1)
if ok2: cv2.imwrite("cam2.jpg", f2)

cam1.release(); cam2.release()
