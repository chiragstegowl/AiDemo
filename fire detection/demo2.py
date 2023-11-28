import cv2
import numpy as np

img = cv2.imread("ima-1.png")
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.imwrite("img_test.png", img) 

blur = cv2.GaussianBlur(img, (21, 21), 0)
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

lower = [25, 50, 50]
upper = [30, 255, 255]
lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")

mask = cv2.inRange(hsv, lower, upper)
output = cv2.bitwise_and(img, hsv, mask=mask)
no_red = cv2.countNonZero(mask)

number_of_white_pix = np.sum(mask == 255) 
number_of_black_pix = np.sum(mask == 0) 
  
print('Number of white pixels:', number_of_white_pix) 
print('Number of black pixels:', number_of_black_pix)

print(no_red)
if int(no_red) > 15000:
    print("IN")
    cv2.putText(img,'Fire True',(10, 70), font, 2, (0, 0, 255), 2, cv2.LINE_AA)


cv2.imwrite("imga_test.png", img) 
cv2.imwrite("imga_test_4.png", hsv) 
cv2.imwrite("imga_test_1.png", output) 
cv2.imwrite("imga_test_2.png", mask) 
cv2.imwrite("imga_test_3.png", output) 
