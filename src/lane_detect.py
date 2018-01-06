'''lane detection'''
import time
import cv2
import numpy as np

# resize to this resolution to improve performance
width, height = (640, 360)
resize = (width, height)
# region of interest
roi = np.zeros((height, width), np.uint8)
#bottom left, top left, top right, bottom right
roi_pt = np.array([ [0.05*width, 0.9*height], [0.45*width, 0.6*height],
                    [0.55*width, 0.6*height],  [0.95*width, 0.9*height]], np.int32)
# create a roi mask
roi = cv2.fillPoly(roi, [roi_pt], (255, 255, 255))

# capture video from webcam
# cap = cv2.VideoCapture(0)

# capture video from video file
# cap = cv2.VideoCapture(r'../data/solidYellowLeft.mp4')
# cap = cv2.VideoCapture(r'../data/solidWhiteRight.mp4')
cap = cv2.VideoCapture(r'../data/challenge.mp4')

# needed for storing the output
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, resize)


while cap.isOpened():
    # read frames
    ret, frame = cap.read()
    if not ret:
        break
    
    # change resolution to improve performance
    frame = cv2.resize(frame, resize)
    cv2.imshow("orig", frame)

    # step1 convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # step2 edge detection
    gray = cv2.Canny(gray, 50, 200)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    # step3 region of interest
    gray = cv2.bitwise_and(gray, roi)
    cv2.imshow("canny", gray)
    # cv2.waitKey(0)
    
    # step4 hough line transform
    lines = cv2.HoughLinesP(gray, 1, np.pi/180, 50, np.array([]), 10, 200)
    # print(lines)
    if lines is not None:
        # strategy - take average slope for left lanes and right lanes
        calc = {'left':np.array([0.0,0.0,0.0]), 'right':np.array([0.0,0.0,0.0])}

        for line in lines:
            line = line[0]
            # slope
            m = 0
            y = (line[3]-line[1])
            x = (line[2]-line[0])
            if x != 0:
                m = y/x
            
            if m == 0 or m < 0 and m > -0.5:
                continue
            elif m > 0 and m < 0.5:
                continue            
            # intercept
            b = line[1] - (m * line[0])
            # weight / length of line
            w = np.linalg.norm(np.reshape(line, (2,2)))
            
            # filter out horizontal lines
            if m <= 0 and m > -0.5:
                continue
            elif m > 0 and m < 0.5:
                continue

            if m < 0:
                calc['left'] += np.array([m*w, b*w, w])
            else:
                calc['right'] += np.array([m*w, b*w, w])

        for key in calc:
            value = calc[key]
            weight = value[2]
            if weight == 0.0:
                continue            
            slope = value[0]/weight
            intercept = value[1]/weight
            # draw overlay
            x1 = int((height-intercept)/slope)
            x2 = int(((height*0.65)-intercept)/slope)
            frame = cv2.line(frame, (x1, height), (x2, int(0.65*height)), (255,0,0),4)                       

    # display and save the output video
    cv2.imshow("final", frame)
    out.write(frame)
    time.sleep(1/30)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        quit(1)

# close all opened files
cap.release()
out.release()
cv2.destroyAllWindows()
