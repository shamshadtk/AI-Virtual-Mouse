import cv2
import numpy as np
import HandTrackingModule as htm
import time
import autopy
import pyautogui as ppy

######################
wCam, hCam = 500, 300
frameR = 100  # Frame Reduction
smoothening = 7  # random value
######################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
# print(wScr, hScr)

while True:
    # Find the landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # Get the tip of the index and middle finger
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Check which fingers are up
        fingers = detector.fingersUp()
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)

        # Only Index Finger: Moving Mode
        if fingers == [0, 1, 0, 0, 0]:

            # Convert the coordinates
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            # Smooth Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Move Mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # Both Index and middle are up: Left Clicking Mode
        if fingers == [0, 1, 1, 0, 0]:

            # Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)

            # Click mouse if distance short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

        # Both Index and thumb are up: Right Clicking Mode
        if fingers == [1, 1, 0, 0, 0]:

            # Find distance between fingers
            length, img, lineInfo = detector.findDistance(4, 8, img)
            # print(length)

            # Click mouse if distance is short
            if length < 30:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           10, (0, 255, 0), cv2.FILLED)

                autopy.mouse.click(autopy.mouse.Button.RIGHT)

        # if three Fingers: scrolling up Mode
        if fingers == [0, 1, 1, 1, 0]:
            ppy.scroll(-30)

        # if four Fingers: scrolling down Mode
        if fingers == [0, 1, 1, 1, 1]:
            ppy.scroll(30)

    # Frame rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 20),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
