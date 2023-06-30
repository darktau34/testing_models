import cv2 as cv

rtsp_login = "admin"
rtsp_password = "123$$sin"
ip_address = "192.168.0.87"
rtsp_port = "554"
channel = "0"

connection_address = "rtsp://" + rtsp_login + ":" + rtsp_password + "@" + ip_address + ":" + rtsp_port + "/" + channel

def resize_frame(frame):
    scale_percent = 60
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    frame = cv.resize(frame, dim)
    return frame

cap = cv.VideoCapture()
cap.open(connection_address)
if not cap.isOpened():
    print("Не открылось")
    exit(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = resize_frame(frame)
    cv.imshow('Video', frame)
    key = cv.waitKey(2)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
