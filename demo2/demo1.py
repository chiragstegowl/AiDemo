import queue
from threading import Thread
from time import sleep
import cv2
import numpy as np

process_quene = queue.Queue()


def get_emb_frame(frame, count):
    cv2.imwrite(f"ima-{count}.png", frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = [25, 50, 50]
    upper = [27, 255, 255]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(hsv, lower, upper)
    no_red = cv2.countNonZero(mask)

    if int(no_red) > 15000:
        print("IN")
        cv2.putText(frame,'Fire: True',(10, 70), font, 2, (0, 0, 255), 2, cv2.LINE_AA)


    return frame


def get_frame_video():
    count = 0
    cap = cv2.VideoCapture("video_1.mp4")

    while cap.isOpened():
        _, frame = cap.read()
        
        count = count + 1
        frame = get_emb_frame(frame, count)
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
    video_name = 'output-pexels-vyacheslav-prisichev-9780684.avi'
    sleep(5)

    while True:
        try:
            frame = process_quene.get_nowait()
            count = count + 1

            if count == 1:
                height, width, _ = frame.shape
                output_video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 25, (width, height))

            output_video.write(frame)
            re_try = 0

        except queue.Empty:
            print(f"Retry {re_try}")
            sleep(5)
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
