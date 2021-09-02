import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy

widthCam, heightCam = 640, 480

# frame reduction
frameR = 100

cap = cv2.VideoCapture(0)
cap.set(3, widthCam)
cap.set(4, heightCam)

pTime = 0

detector = htm.handDetector(maxHands=1)

wScr, hScr = autopy.screen.size()

smoothening = 5

# previous x/y coordinates
prevX, prevY = 0, 0

# current x/y coordinates
currentX, currentY = 0, 0

while True:
    success, img = cap.read()

    # find hand landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # get the tip of the index and middle finger
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # print(x1, y1, x2, y2)

        # check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        cv2.rectangle(img, (frameR, frameR), 
            (widthCam - frameR, heightCam - frameR), (255, 0, 255), 2)

        # index finger, moving mode
        if fingers[1] == 1 and fingers[2] == 0:
            # convert coordinates
            x3 = np.interp(x1, (frameR, widthCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, heightCam - frameR), (0, hScr))

            # smoothening
            currentX = prevX + (x3 - prevX) / smoothening
            currentY = prevY + (y3 - prevY) / smoothening

            # move mouse
            autopy.mouse.move(wScr - currentX, currentY)
            cv2.circle(img, (x1, y1), 15, (119, 170, 193), cv2.FILLED)
            prevX, prevY = currentX, currentY

        # index and middle finger are up, click mode
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, 
                    (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()


    # fps
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), 
        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
