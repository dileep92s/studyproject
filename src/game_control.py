'''lane detection'''
import time
from threading import Thread

import cv2
import numpy as np

from directkeys import A, D, PressKey, ReleaseKey, S, W
from grabscreen import grab_screen


class KeyPress(Thread):
    ''' press keys'''

    def __init__(self, key, delay):
        self.key = key
        self.delay = 0 if delay < 0 else delay
        Thread.__init__(self)

    def run(self):
        if self.key == D:
            PressKey(D)
            time.sleep(self.delay)
            ReleaseKey(D) 
        elif self.key == A:
            PressKey(A)
            time.sleep(self.delay)
            ReleaseKey(A)
        elif self.key == W:
            PressKey(W)
            time.sleep(self.delay)


class Adrive:

    def __init__(self, record=False):
        self.width, self.height = (640, 360)
        self.roi_mask = self.create_roi()
        self.record = record
        self.frame = None
        self.err_p_old = 0
        
        if self.record:
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            self.record = cv2.VideoWriter('output.avi', fourcc, 20.0, (self.width, self.height))

    def create_roi(self):
        mask = np.zeros((self.height, self.width), np.uint8)
        roi_pt = np.array([ [0.01*self.width, 0.99*self.height], [0.01*self.width, 0.8*self.height], 
                            [0.47*self.width, 0.55*self.height], [0.53*self.width, 0.55*self.height],  
                            [0.99*self.width, 0.8*self.height], [0.99*self.width, 0.99*self.height]], np.int32)
        mask = cv2.fillPoly(mask, [roi_pt], (255, 255, 255))
        return mask

    def prepare_frame(self):
        grayf = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        edgef = cv2.Canny(grayf, 50, 100)
        maskf = cv2.bitwise_and(edgef, self.roi_mask)
        blurf = cv2.GaussianBlur(maskf, (5, 5), 0)
        return blurf

    def get_line_length(self, x1, y1, x2, y2):
        xsq = (x1-x2)**2
        ysq = (y1-y2)**2
        return np.sqrt(xsq + ysq)

    def get_slope_intercept(self, lines):

        slope = {'left':{'slope':0, 'intercept':0, 'weight':0}, 'right':{'slope':0, 'intercept':0, 'weight':0}}

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                line_length = self.get_line_length(x1, y1, x2, y2)
                if (x2-x1):
                    line_slope = float(y2-y1)/(x2-x1)
                else:
                    # line_slope = 0.000001
                    continue
                line_intercept = y1 - (line_slope * x1)
                if line_slope < 0:
                    slope['left']['slope'] += line_slope * line_length
                    slope['left']['intercept'] += line_intercept * line_length
                    slope['left']['weight'] += line_length

                elif line_slope > 0:
                    slope['right']['slope'] += line_slope * line_length
                    slope['right']['intercept'] += line_intercept * line_length
                    slope['right']['weight'] += line_length
        return slope

    def draw_overlay(self, slope):

        reference = {'left':0, 'right':0, 'mid':0}

        for key in slope:
            if slope[key]['weight']:
                line_slope = slope[key]['slope'] / slope[key]['weight']
                intercept = slope[key]['intercept'] / slope[key]['weight']

                y1 = int(self.height)
                x1 = int((y1 - intercept)/line_slope)
                
                y2 = int(self.height*0.65)
                x2 = int((y2 - intercept)/line_slope)

                self.frame = cv2.line(self.frame, (x1, y1), (x2, y2), (255,0,0), 2)
                
                y1 = int(self.height*0.8)
                x1 = int((y1-intercept)/line_slope)
                self.frame = cv2.circle(self.frame, (x1, y1), 2, (255,0,255), 2)

                reference[key] = x1

        mid_x = int(self.width/2)
        mid_y = int(self.height*0.8)
        self.frame = cv2.line(self.frame, (mid_x, mid_y+10), (mid_x, mid_y-10), (0,255,0), 2)
        reference['mid'] = mid_x

        return reference


    def lane_keeping(self, reference):
        left_x = reference['left']
        right_x = reference['right']
        mid_x  = reference['mid']
        
        if left_x and right_x:
            my_x = int((left_x + right_x)/2)
            my_y = int(self.height*0.8)
            self.frame = cv2.line(self.frame, (my_x, my_y+10), (my_x, my_y-10), (0,0,255), 2)

            err =  (my_x-mid_x)*100/mid_x
            err_p = abs(err)
            err_d = abs(self.err_p_old - err_p)
            self.err_p_old = err_p

            delay = 0.01 * err_p - 0.005 * err_d
            print(err, err_p, err_d, delay)
            if err < 100 and err > 3:
                print("right") 
                key = KeyPress(D, delay)
                key.start()
            elif err < -3 and err > -100:
                print("left") 
                key = KeyPress(A, delay)
                key.start()
            elif err < 3 and err > -3:
                print("straight") 
            key = KeyPress(W, 0.1)
            key.start()            
        
            err = "%0.2f" %err
            cv2.putText(self.frame, str(err), (mid_x-20, my_y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

    def start(self):
        while True:
            self.frame = grab_screen(region=(0, 0, 1024-10, 768-10))
            self.frame = cv2.resize(self.frame, (self.width, self.height))
            maskf = self.prepare_frame()
            lines = cv2.HoughLinesP(maskf, 1, np.pi/180, 100, np.array([]), 20, 200)
            slope = self.get_slope_intercept(lines)
            refer = self.draw_overlay(slope)
            self.lane_keeping(refer)
            cv2.imshow("result", self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if self.record:
                self.record.write(self.frame)

    def __del__(self):
        if self.record:
            self.record.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    myapp = Adrive(record=True)
    for i in range(3):
        print(i)
        time.sleep(1)
    myapp.start()
