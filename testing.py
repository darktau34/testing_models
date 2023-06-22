import torch
import cv2 as cv
import pandas as pd
from ultralytics import RTDETR
from ultralytics import YOLO
from imageai.Detection import ObjectDetection

def imageai_retina():
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath("imageai_models/retinanet_resnet50_fpn_coco-eeacb38b.pth")
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image="images/road_cars.jpg", output_image_path="runs/imagenew.jpg",
                                                 minimum_percentage_probability=50)
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

# https://www.freecodecamp.org/news/how-to-detect-objects-in-images-using-yolov8/#get_started
def rtdetr():
    img_path = "images/road_cars.jpg"
    model = RTDETR("rtdetr-l.pt")
    model.info()  # display model information
    predict = model.predict(img_path, save=False)  # predict
    cords = predict[0].boxes.xyxy.tolist()
    class_id = predict[0].boxes.cls.tolist()
    conf = predict[0].boxes.conf.tolist()

    objects_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
            12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat'}

    img = cv.imread(img_path)
    print('Кол-во рамок: {} \nКол-во id: {}'.format(len(cords), len(class_id)))
    for i in range(len(cords)):
        vertex1 = (int(cords[i][0]), int(cords[i][1]))
        vertex2 = (int(cords[i][2]), int(cords[i][3]))
        cv.rectangle(img, vertex1, vertex2, (0, 252, 124), 2)

    for i in range(len(cords)):
        pos_text = (int(cords[i][0]), int(cords[i][1]) - 5)
        name = objects_names[class_id[i]]
        conf_percent = str(round(conf[i] * 100, 1))
        text = name + ' ' + conf_percent + '%'
        cv.putText(img, text, pos_text, cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 170, 66, 255), 1)


    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# model = RTDETR("rtdetr-l.pt")
model = YOLO('yolov8n.pt')
objects_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                     7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                     12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
                     19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
                     26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis'}
def rtdetr_frames(frame):
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
        frame = rtdetr_frames(frame)
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
    # yolov8()
    # rtdetr()
    # imageai_retina()


if __name__ == "__main__":
    main()
