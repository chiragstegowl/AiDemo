import cv2
import mediapipe as mp
import numpy as np

sample_img = cv2.imread("BackgroundImage.png")
print('IN')
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.1) as face_detection:
    face_detection_results = face_detection.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    frame_height, frame_width, c = sample_img.shape

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


        cv2.rectangle(sample_img, bounding_box, color=(255, 255, 255), thickness=2)
        roi = sample_img[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]
        roi = cv2.GaussianBlur(roi, (171, 171), 300)
        sample_img[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]] = roi


cv2.imwrite("img_test_1.png", sample_img) 
