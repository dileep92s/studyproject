'''lane detection'''
import time
import cv2
import numpy as np


# cap = cv2.VideoCapture(r'../data/solidYellowLeft.mp4')
cap = cv2.VideoCapture(r'../data/challenge.mp4')
# cap = cv2.VideoCapture(r'../data/solidWhiteRight.mp4')
roi = None
iroi = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_o = frame
    mymask = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    mymask_w = cv2.inRange(mymask, np.array([0, 200, 0], dtype=np.uint8), np.array([255, 255, 255], dtype=np.uint8))
    mymask_y = cv2.inRange(mymask, np.array([10, 0, 100], dtype=np.uint8), np.array([255, 255, 255], dtype=np.uint8))
    mymask = cv2.bitwise_or(mymask_w, mymask_y)
    frame = cv2.bitwise_and(frame, frame, mask=mymask)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if roi is None:
        roi = np.zeros(gray.shape, np.uint8)
        height, width = gray.shape
        pts = np.array([[0.05*width, 0.90*height], [0.45*width, 0.6*height],
                        [0.55*width, 0.6*height],  [0.95*width, 0.90*height]], np.int32)
        cv2.fillPoly(roi, [pts], (255, 255, 255))
        iroi = cv2.bitwise_not(roi)

    gray = cv2.Canny(gray, 100, 200)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = np.bitwise_and(gray, roi)
    frame1 = cv2.bitwise_and(frame_o, frame_o, mask=iroi)
       
    lines = cv2.HoughLines(gray, 1, np.pi/180, 30)
    if lines is not None:
        ldone, rdone = False, False
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]

            x1 = int(rho*np.cos(theta) + 2000*(-np.sin(theta)))
            y1 = int(rho*np.sin(theta) + 2000*(np.cos(theta)))

            x2 = int(rho*np.cos(theta) - 2000*(-np.sin(theta)))
            y2 = int(rho*np.sin(theta) - 2000*(np.cos(theta)))

            if rho > 0 and not ldone:
                cv2.line(frame_o, (x1, y1), (x2, y2), (255, 0, 255), 4)
                ldone = True

            if rho < 0 and not rdone:
                cv2.line(frame_o, (x1, y1), (x2, y2), (255, 0, 255), 4)
                rdone = True

    frame2 = cv2.bitwise_or(frame_o, frame_o, mask=roi)
    res = cv2.bitwise_or(frame1, frame2)
    cv2.imshow("vidg", gray)
    cv2.imshow("vid", res)
    # time.sleep(0.05)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
