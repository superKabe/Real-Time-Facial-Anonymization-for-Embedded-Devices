import warnings
warnings.filterwarnings('ignore', message='GStreamer')

import cv2
import time


print("Please look into the camera")
fullName = input('Enter your first name: ')
print("Please look into the camera and press SPACE to capture")

fullName = str(fullName).replace(" ", "-")

cam = cv2.VideoCapture(0)
cv2.namedWindow("Camera")

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Camera", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "images/{}.png".format(fullName)
        cv2.imwrite(img_name, frame)
        break


time.sleep(3)

cam.release()
cv2.destroyAllWindows()