import queue
from threading import Thread
from time import sleep
import cv2
import mediapipe as mp
import numpy as np

process_quene = queue.Queue()
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.1)


def get_emb_frame(frame):
    face_detection_results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame_height, frame_width, _ = frame.shape

    for face in face_detection_results.detections:
        bounding_box = np.multiply(
            [
                face.location_data.relative_bounding_box.xmin,
                face.location_data.relative_bounding_box.ymin,
                face.location_data.relative_bounding_box.width,
                face.location_data.relative_bounding_box.height,
            ],
            [frame_width, frame_height, frame_width, frame_height],
        ).astype(int)

    roi = frame[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]
    roi = cv2.GaussianBlur(roi, (171, 171), 300)
    frame[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]] = roi

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
