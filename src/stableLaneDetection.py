'''lane detection'''
import time
import cv2
import numpy as np


def readVideo(videoPath):
    cap = cv2.VideoCapture(videoPath)
    return cap

def genROI(img, pts):
    roi = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(roi, [pts], (255, 255, 255))
    return roi

def maskWY(frame):
    mymask = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    mymask_w = cv2.inRange(mymask, np.array([0, 200, 0], dtype=np.uint8), np.array([255, 255, 255], dtype=np.uint8))
    mymask_y = cv2.inRange(mymask, np.array([10, 0, 100], dtype=np.uint8), np.array([255, 255, 255], dtype=np.uint8))
    mymask = cv2.bitwise_or(mymask_w, mymask_y)
    mymask = cv2.bitwise_and(frame, frame, mask=mymask)
    return mymask

# cap = readVideo('../data/solidYellowLeft.mp4')
cap = readVideo('../data/challenge.mp4')
# cap = readVideo('../data/solidWhiteRight.mp4')

roi = None
iroi = None
temp =0
while cap.isOpened():
    temp =0
    ret, frame = cap.read()
    if not ret:
        break
    frame_o = frame
    frame = maskWY(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    pts = np.array([[0.05*width, 0.90*height], [0.45*width, 0.6*height],\
                    [0.55*width, 0.6*height],  [0.95*width, 0.90*height]], np.int32)
    roi  = genROI(gray, pts)
    cv2.imshow("roi", roi)
    iroi = cv2.bitwise_not(roi)
    gray = np.bitwise_and(gray, roi)  
     
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    gray = cv2.Canny(gray, 100, 200) 
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow("yw2", gray) 
    lines = cv2.HoughLinesP(gray,1,np.pi/180,100,minLineLength=100,maxLineGap=200)
    xi1 = []
    yi1 = []
    xi2 = []
    yi2 = []

    if lines is not None:
        for i in range(0, len(lines)):
        	x1,y1,x2,y2 = lines[i][0]
        	if((y1-y2)/(x1-x2)>0):
        		xi1.append(x1)
	        	xi1.append(x2)
	        	yi1.append(y1)
	        	yi1.append(y2)
	        if((y1-y2)/(x1-x2)<0):
	        	xi2.append(x1)
	        	xi2.append(x2)
	        	yi2.append(y1)
	        	yi2.append(y2)
        	#cv2.line(frame_o, (x1, y1), (x2, y2), (255, 0, 255), 4)
        print (yi1)
        print ("XXXXXXXXX")
        print (yi2)
        A1 = np.array([ xi1, np.ones(len(xi1))])
        if(len(xi1)>0 and len(yi1)>0):
        	w1 = np.linalg.lstsq(A1.T,yi1)[0]

        A2 = np.array([ xi2, np.ones(len(xi2))])
        if(len(xi2)>0 and len(yi2)>0):
        	w2 = np.linalg.lstsq(A2.T,yi2)[0]
        if(len(xi1)>0 and len(yi1)>0):
        	cv2.line(frame_o, (xi1[0], int(w1[0]*xi1[0]+w1[1])), (xi1[1], int(w1[0]*xi1[1]+w1[1])), (255, 0, 255), 12)
        if(len(xi2)>0 and len(yi2)>0):
        	cv2.line(frame_o, (xi2[0], int(w2[0]*xi2[0]+w2[1])), (xi2[1], int(w2[0]*xi2[1]+w2[1])), (255, 0, 255), 12)



    frame1 = cv2.bitwise_and(frame_o, frame_o, mask=iroi)
    frame2 = cv2.bitwise_or(frame_o, frame_o, mask=roi)
    res = cv2.bitwise_or(frame1, frame2)
    cv2.imshow("vidg", gray)
    cv2.imshow("vid", res)
    # time.sleep(0.05)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print (lines)
cap.release()
cv2.destroyAllWindows()
