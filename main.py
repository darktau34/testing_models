import cv2 as cv
from ultralytics import YOLO
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.draw.color import ColorPalette
import time


model = YOLO('yolov8m.pt')
CLASS_NAMES = model.model.names
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.4, text_padding=5)


def frames_handler(frame):
    results = model.track(source=frame, tracker='bytetrack.yaml', persist=True)[0]
    detections = Detections(
        xyxy = results.boxes.xyxy.cpu().numpy(),
        confidence= results.boxes.conf.cpu().numpy(),
        class_id = results.boxes.cls.cpu().numpy().astype(int),
        tracker_id = results.boxes.id.cpu().numpy().astype(int)
    )

    labels = [
        f"#{tracker_id} {CLASS_NAMES[class_id]} {confidience:0.2f}"
        for _, confidience, class_id, tracker_id
        in detections
    ]
    
    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    return frame
    
def resize_frame(frame):
    scale_percent = 60
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    frame = cv.resize(frame, dim)
    return frame


def video_stream(cap):
    prev_frame_time = 0
    new_frame_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = resize_frame(frame)
        frame = frames_handler(frame)
        
        new_frame_time = time.time()
        fps = int(1/(new_frame_time-prev_frame_time))
        prev_frame_time = new_frame_time
        
        cv.putText(frame, str(fps), (10, 30), cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0, 255), 2)
        
        cv.imshow('Video', frame)
        key = cv.waitKey(2)
        if key == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()
    return False
        
def get_camera():
    rtsp_login = "admin"
    rtsp_password = "123$$sin"
    ip_address = "192.168.0.87"
    rtsp_port = "554"
    channel = "0"

    connection_address = "rtsp://" + rtsp_login + ":" + rtsp_password + "@" + ip_address + ":" + rtsp_port + "/" + channel
    
    cap = cv.VideoCapture()
    cap.open(connection_address)
    if not cap.isOpened():
        print("Видеопоток не открылся!")
        
    return cap


def main():
    cap = get_camera()
    video_stream(cap)

if __name__ == '__main__':
    main()
