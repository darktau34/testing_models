from imageai.Detection import ObjectDetection
import cv2 as cv
import time

def resize_frame(frame):
    scale_percent = 60
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    frame = cv.resize(frame, dim)
    return frame


def frames_handler(frame):
    returned_image, detections = detector.detectObjectsFromImage(input_image=frame, output_type='array', minimum_percentage_probability=30)
    
    frame = returned_image

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


detector = ObjectDetection()


detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath("tiny-yolov3.pt")
# detector.setModelTypeAsRetinaNet()
# detector.setModelPath("retinanet.pth")


detector.loadModel()
video_detection('video/street.mp4')