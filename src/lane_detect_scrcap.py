'''lane detection'''
import time
import cv2
import numpy as np
from grabscreen import grab_screen


roi = None
iroi = None

while 1:
    frame = grab_screen(region=(40, 80, 640, 480))
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if roi is None:
        roi = np.zeros(gray.shape, np.uint8)
        height, width = gray.shape
        pts = np.array([[0.05*width, 0.90*height], [0.45*width, 0.6*height], [0.55*width, 0.6*height], [0.95*width, 0.90*height]], np.int32)
        cv2.fillPoly(roi, [pts], (255, 255, 255))
        iroi = cv2.bitwise_not(roi)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.Canny(gray, 100, 200)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = np.bitwise_and(gray, roi)

    frame1 = cv2.bitwise_and(frame,frame, mask=iroi)
    lines = cv2.HoughLinesP(gray,1,np.pi/180,100,np.array([]),100,50)
    yy2 = 0
    yy1 = 0
    if lines is not None:
        for line in lines:
            # print line
            for x1,y1,x2,y2 in line:
                if y1 not in range(yy1-200,yy1+200):
                    cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    yy1 = (y1 + yy1)/2;
                if y2 not in range(yy2-200,yy2+200):
                    cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),2)
                    yy2 = (y2 + yy2)/2;

    
    frame2 = cv2.bitwise_or(frame, frame, mask=roi)
    res = cv2.bitwise_or(frame1,frame2)

    cv2.imshow("vidg",gray)
    cv2.imshow("vid",res)
    time.sleep(0.05)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
