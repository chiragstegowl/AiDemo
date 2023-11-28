import queue
from ultralytics import YOLO
import cv2
from time import sleep
from threading import Thread

model = YOLO("yolov8m.pt")

process_quene = queue.Queue()


def get_emb_frame(frame):

    data_labels = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    font = cv2.FONT_HERSHEY_SIMPLEX
    results = model.predict(frame)
    result = results[0]

    for box in result.boxes:
        bounding_box = box.xyxy[0].tolist()
        bounding_box = [ int(item) for item in bounding_box]
        probability = box.conf[0].tolist()
        probability = round(probability, 2)
        show_text = data_labels[int(box.cls[0].tolist())]
        show_text = f"{show_text}: {probability}"
        cv2.putText(frame, show_text, (bounding_box[0] - 20, bounding_box[1] - 10), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (255, 255, 255), 3)

    return frame


def get_frame_video():
    count = 0
    cap = cv2.VideoCapture("video_1.mp4")

    while cap.isOpened():
        _, frame = cap.read()
        print(count)
        
        count = count + 1
        frame = get_emb_frame(frame)
        process_quene.put(frame)

        if count == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            print("Reached the last frame.")
            break

    print('out')
    cap.release()
    cv2.destroyAllWindows()



def create_new_video():
    re_try = 0
    count = 0
    video_name = 'output_video.avi'
    sleep(10)

    while True:
        try:
            frame = process_quene.get_nowait()
            count = count + 1
            print(f'Count {count}')
            if count == 1:
                height, width, _ = frame.shape
                output_video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))

            output_video.write(frame)
            re_try = 0

        except queue.Empty:
            print(f"Retry {re_try}")
            sleep(10)
            re_try = re_try + 1

        if re_try >= 3:
            print('Break')
            output_video.release()
            break

def start_process():        
    thread_1 = Thread(target = get_frame_video)
    thread_2 = Thread(target = create_new_video)
    thread_1.start()
    thread_2.start()
    thread_1.join()
    thread_2.join()


start_process()
print("thread finished...exiting")
