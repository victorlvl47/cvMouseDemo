import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

widthCam, heightCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, widthCam)
cap.set(4, heightCam)

pTime = 0

detector = htm.handDetector(maxHands=1)

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), 
        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
