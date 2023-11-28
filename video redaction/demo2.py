from mtcnn import MTCNN
import cv2

detector = MTCNN()
img = cv2.imread("BackgroundImage.png")
detections = detector.detect_faces(img)


for detection in detections:
    print('=================')
    print(detection['confidence']) 
    print(detection['box']) 
    bounding_box = detection['box']

    roi = img[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]]
    roi = cv2.GaussianBlur(roi, (17, 17), 30)
    img[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0]+bounding_box[2]] = roi

    # cv2.rectangle(img, (bounding_box[0], bounding_box[1]),
    #           (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]), (255, 255, 255), 2)

cv2.imwrite("img_test.png", img) 
