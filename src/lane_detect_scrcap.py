'''lane detection'''
import time
import cv2
import numpy as np
from grabscreen import grab_screen
from directkeys import PressKey, ReleaseKey, W, A, S, D
from threading import Thread

class KeyPress(Thread):
    ''' press keys'''

    def __init__(self, key, delay):
        self.key = key
        self.delay = delay
        Thread.__init__(self)

    def run(self):
        if self.key == D:
            self.right()
        elif self.key == A:
            self.left()
        elif self.key == W:
            self.straight()

    def straight(self):
        PressKey(W)
        time.sleep(self.delay)
        ReleaseKey(A)
        ReleaseKey(D)

    def left(self):
        PressKey(A)
        time.sleep(self.delay)
        ReleaseKey(W)
        ReleaseKey(D)
        ReleaseKey(A)

    def right(self):
        PressKey(D)
        time.sleep(self.delay)
        ReleaseKey(A)
        ReleaseKey(W)
        ReleaseKey(D)      


# resize to this resolution to improve performance
width, height = (640, 360)
resize = (width, height)
# region of interest
roi = np.zeros((height, width), np.uint8)
#bottom left, top left, top right, bottom right
roi_pt = np.array([ [0.01*width, 0.99*height], [0.01*width, 0.8*height], [0.47*width, 0.55*height],
                    [0.53*width, 0.55*height],  [0.99*width, 0.8*height], [0.99*width, 0.99*height]], np.int32)
# create a roi mask
roi = cv2.fillPoly(roi, [roi_pt], (255, 255, 255))

# needed for storing the output
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, resize)

for i in range(1,3):
    print(i)
    time.sleep(1)
print("ready")

xo = 0
yo = 20
key_press = True
while True:
    # read frames
    frame = grab_screen(region=(xo, yo, 1024-10, 768-10))   
    # change resolution to improve performance
    frame = cv2.resize(frame, resize)
    cv2.imshow("orig", frame)

    # step1 convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # step2 edge detection
    gray = cv2.Canny(gray, 50, 100)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    # step3 region of interest
    gray = cv2.bitwise_and(gray, roi)
    cv2.imshow("canny", gray)
    # cv2.waitKey(0)
    
    # step4 hough line transform
    lines = cv2.HoughLinesP(gray, 1, np.pi/180, 100, np.array([]), 20, 200)
    # print(lines)
    if lines is not None:
        # strategy - take average slope for left lanes and right lanes
        calc = {'left':np.array([0.0,0.0,0.0]), 'right':np.array([0.0,0.0,0.0])}

        for line in lines:
            line = line[0]
            x1, y1, x2, y2 = line
            # slope
            m = 0
            y = float(y2-y1)
            x = float(x2-x1)
            if x != 0:
                m = y/x
            # print(line, m)
            # frame = cv2.line(frame, (x1, y1), (x2, y2), (150,150,150), 1)
            
            # ignore horizontal lines
            if m == 0 or m < 0 and m > -0.03:
                continue
            elif m > 0 and m < 0.03:
                continue            

            # intercept
            b = line[1] - (m * line[0])
            # weight / length of line
            w = np.linalg.norm(np.reshape(line, (2,2)))

            if m < 0:
                calc['left'] += np.array([m*w, b*w, w])
            else:
                calc['right'] += np.array([m*w, b*w, w])

        dots = []
        #d raw overlay

        for key in calc:
            value = calc[key]
            weight = value[2]
            if weight == 0.0:
                continue            
            slope = value[0]/weight
            intercept = value[1]/weight
            # draw overlay
            y1 = int(height)
            x1 = int((y1-intercept)/slope)
            
            y2 = int(height*0.65)
            x2 = int((y2-intercept)/slope)

            frame = cv2.line(frame, (x1, y1), (x2, y2), (255,0,0),2)

            y1 = int(height*0.9)
            x1 = int((y1-intercept)/slope)
            frame = cv2.circle(frame, (x1, y1), 2, (0,255,0), 2)
            dots.append([x1,y1])
        
        # draw deviation from centre 
        if len(dots) == 2:
            x1 = dots[0][0]
            x2 = dots[1][0]
            x1 = int((x1+x2)/2)
            y1 = int(0.9*height)
            frame = cv2.circle(frame, (x1, y1), 2, (0,0,255), 2)

            x2 = int(0.5*width)
            frame = cv2.circle(frame, (x2, y1), 2, (255,0,255), 2)
            deviation = (1-(x1/x2))*100
            t = 0.01*np.abs(deviation)
            if deviation > 3 and  deviation < 60:
                print("left")
                q = KeyPress(A, t)
                q.start()
                # left(t)
            elif deviation < -3 and deviation > -60:
                print("right")
                q = KeyPress(D, t)
                q.start()
                # right(t)
            # else:
            print("straight")
            # straight(0.5)
            q = KeyPress(W, 0.5)
            q.start()

            deviation = "%0.2f" %deviation
            cv2.putText(frame, str(deviation), (x2,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1, cv2.LINE_AA)

    x1 = int(width/2)
    y1 = int(height*0.9)
    frame = cv2.circle(frame, (x1, y1), 2, (255,0,255), 2)
    # display and save the output video
    cv2.imshow("final", frame)
    print(time.time())
    # out.write(frame)
    # time.sleep(1/30)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close all opened files
out.release()
cv2.destroyAllWindows()
