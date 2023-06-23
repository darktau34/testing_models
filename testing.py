import torch
import cv2 as cv
import pandas as pd
from ultralytics import RTDETR
from ultralytics import YOLO



# https://www.freecodecamp.org/news/how-to-detect-objects-in-images-using-yolov8/#get_started

# model = RTDETR("rtdetr-l.pt")
# model = YOLO('yolov8n.pt')
objects_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                     7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                     12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
                     19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
                     26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis'}
def frames_handler(frame):
    predict = model.predict(frame, save=False)  # predict
    cords = predict[0].boxes.xyxy.tolist()
    class_id = predict[0].boxes.cls.tolist()
    conf = predict[0].boxes.conf.tolist()

    for i in range(len(cords)):
        vertex1 = (int(cords[i][0]), int(cords[i][1]))
        vertex2 = (int(cords[i][2]), int(cords[i][3]))
        cv.rectangle(frame, vertex1, vertex2, (0, 252, 124), 2)

    for i in range(len(cords)):
        pos_text = (int(cords[i][0]), int(cords[i][1]) - 5)
        name = objects_names[class_id[i]]
        conf_percent = str(round(conf[i] * 100, 1))
        text = name + ' ' + conf_percent + '%'
        cv.putText(frame, text, pos_text, cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 170, 66, 255), 1)

    return frame
def yolov8():
    model = YOLO('yolov8m.pt')  # load a pretrained model
    results = model.predict('images/road_cars.jpg', save=True)
    print(results[0].boxes.cls.tolist())
    print(results[0].boxes.xyxy.tolist())
    print(results[0].boxes.conf.tolist())

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

    while cap.isOpened():
        ret, frame = cap.read()
        frame = resize_frame(frame)
        # frame = frames_handler(frame)
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
    # video_detection(video_path)
    


if __name__ == "__main__":
    main()
