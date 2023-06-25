import cv2 as cv
from ultralytics import YOLO
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.draw.color import ColorPalette
import time


model = YOLO('yolov8s.pt')
CLASS_NAMES = model.model.names
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.4, text_padding=5)


def frames_handler(frame):
    # tracker='bytetrack.yaml'
    # tracker='botsort.yaml'
    results = model.track(source=frame, tracker='botsort.yaml', persist=True)[0]
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

def video_detection(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Видеофайл не открылся!")

    fps = cap.get(cv.CAP_PROP_FPS)
    frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    print("Кадров в секунду: {} \nКоличество кадров: {}".format(fps, frame_count))

    prev_frame_time = 0
    new_frame_time = 0

    while cap.isOpened():
        # Processing time for this frame = Current time – time when previous frame processed
        #  FPS = 1/ (Processing time for this frame)
        ret, frame = cap.read()
        frame = resize_frame(frame)
        frame = frames_handler(frame)
        new_frame_time = time.time()
        fps = int(1/(new_frame_time-prev_frame_time))
        prev_frame_time = new_frame_time
        cv.putText(frame, str(fps), (10, 30), cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0, 255), 2)
        if not ret:
            break
        cv.imshow('Video', frame)
        key = cv.waitKey(2)
        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    return False


def main():
    video_path = r'video/street.mp4'
    video_detection(video_path)
    


if __name__ == "__main__":
    main()
