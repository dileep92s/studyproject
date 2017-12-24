'''lane detection'''
import time
import cv2
import numpy as np


cap = cv2.VideoCapture(r'../data/challenge.mp4')

ret, frame = cap.read()
roi = None
if ret:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = np.zeros(frame.shape, np.uint8)
    roi.reshape(frame.shape)
    height, width = frame.shape
    pts = np.array([[0.05*width, 0.90*height], [0.45*width, 0.6*height], [0.55*width, 0.6*height], [0.95*width, 0.90*height]], np.int32)
    cv2.fillPoly(roi, [pts], (255, 255, 255))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.Canny(gray, 100, 200)
    gray = np.bitwise_and(gray, roi)

    lines = cv2.HoughLines(gray,1,np.pi/180,200)
    if not lines == None:
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(gray,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imshow('frame', gray)
    
    time.sleep(0.05)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()