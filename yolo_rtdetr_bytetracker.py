import cv2 as cv
from ultralytics import YOLO
from ultralytics import RTDETR
import yolox
from yolox.tracker.byte_tracker import BYTETracker, STrack
print("yolox.__version__:", yolox.__version__)
from supervision.tools.detections import Detections, BoxAnnotator
from supervision.draw.color import ColorPalette
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
import time
from typing import List
import numpy as np


# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy) # совмещает track boxes и detection boxes, у каждого track box'a 
                                                       # есть значение вероятности? принадлежности к тому или иному detection box'y
    track2detection = np.argmax(iou, axis=1)    # определяем к какому detection box'y относится track box

    tracker_ids = [None] * len(detections)  # создаем кол-во полей с id равное кол-ву detectin box'ов

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id # назначем id's

    return tracker_ids



@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


def resize_frame(frame):
    scale_percent = 60
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    frame = cv.resize(frame, dim)
    return frame


def frames_handler(frame):
    results = model(frame)[0]

    detections = Detections(
        xyxy = results.boxes.xyxy.cpu().numpy(),
        confidence= results.boxes.conf.cpu().numpy(),
        class_id = results.boxes.cls.cpu().numpy().astype(int)
    )
    tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
    tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
    detections.tracker_id = np.array(tracker_id)
    labels = [
        f"#{tracker_id} {CLASS_NAMES[class_id]} {confidience:0.2f}"
        for _, confidience, class_id, tracker_id
        in detections
    ]
    frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
    return frame


def video_processing(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Видеофайл не открылся!")

    fps = cap.get(cv.CAP_PROP_FPS)
    frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    print("Кадров в секунду: {} \nКоличество кадров: {}".format(fps, frame_count))

    prev_frame_time = 0
    new_frame_time = 0

    while cap.isOpened():
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
    video_path = 'video/street.mp4'
    video_processing(video_path)


byte_tracker = BYTETracker(BYTETrackerArgs())
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.4, text_padding=5)


model = YOLO('yolov8s.pt')
# model = RTDETR("rtdetr-l.pt")


CLASS_NAMES = model.model.names
main()
