import torch
import cv2 as cv
import pandas as pd
from ultralytics import RTDETR
from ultralytics import YOLO


def yolov5():
    model = torch.hub.load("ultralytics/yolov5", "yolov5l", pretrained=True)

    img = cv.imread("images/road_cars.jpg")
    scale_percent = 60
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img = cv.resize(img, dim)

    detected = model(img)

    detected.show()
    # cv.imshow('image', image_resize)
    # cv.waitKey(0)
    cv.destroyAllWindows()


def rtdetr():
    model = RTDETR("rtdetr-l.pt")
    model.info()  # display model information
    model.predict("images/road_cars.jpg")  # predict


def yolov8():
    model = YOLO('yolov8n.pt')  # load a pretrained model
    results = model.predict('images/road_cars.jpg', save=False)
    print(results[0].boxes.cls.tolist())
    print(results[0].boxes.xyxy.tolist())
    print(results[0].boxes.conf.tolist())



def video_detection(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Видеофайл не открылся!")

    fps = cap.get(cv.CAP_PROP_FPS)
    frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    print("Кадров в секунду: {} \nКоличество кадров: {}".format(fps, frame_count))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv.imshow('Video', frame)
        key = cv.waitKey(20)
        if key == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
    return False


def main():
    video_path = r'video/streetball.mp4'
    # video_detection(video_path)
    yolov8()

if __name__ == "__main__":
    main()
