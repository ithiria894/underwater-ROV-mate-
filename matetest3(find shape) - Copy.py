import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
import matplotlib.pyplot as plt
from directkeys import PressKey,ReleaseKey, W, A, S, D

slope = 0
# slopes =[]
directions = ''

def findSlope(lines):
    slopes=[]
    try:
    # if lines is not None:
        for line in lines:
            if line is not None:
                # for x1, y1, x2, y2 in line:

                x1,y1,x2,y2=line.reshape(4)
                # print(line.reshape(4))

                if x1!=x2 and y1!=y2:
                    slope = (y2-y1)/(x2-x1)
                    # print(slope)

                    slopes.append(slope)

    except:
        pass
    return slopes

def roi(img):
    vertices = np.array([[10,500],[10,300], [300,200], [500,200], [800,300], [800,500]], np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def findContours(image):
    orig_image = image.copy()
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image2 = cv2.threshold(image2, 127, 255, 1)
    image2 = cv2.Canny(image2,30,200)
    # Find contours 
    contours, hierarchy = cv2.findContours(image2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = 200000000000000
    min_area = 100
    for c in contours:    
        area = cv2.contourArea(c)
        if min_area < area < max_area:
            
            approx = cv2.approxPolyDP(c, 0.03*cv2.arcLength(c,True),False)

            cv2.drawContours(orig_image, [c], 0,(0,255,0), -1)
            # cv2.drawContours(orig_image, [c], 0,(225,0,0), -1)
            x,y,w,h = cv2.boundingRect(c)
            # cv2.rectangle(orig_image3,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.circle(orig_image, (x+int(w/2),y+int(h/2)), 5, (0,0,255), -1) 
    return orig_image

def main():
    last_time = time.time()
    while True:

        screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        if screen is not None:
            # new_screen = process_img(screen)
            # print('Frame took {} seconds'.format(time.time()-last_time))
            image = screen
            original_image = image
          
            processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # processed_img = cv2.GaussianBlur(processed_img,(5,5),0) #new

            processed_img =  cv2.Canny(processed_img, 100, 170,apertureSize=3) #try
            
            processed_img = roi(processed_img)

            # lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), minLineLength=15,maxLineGap=5)
            lines = cv2.HoughLinesP(processed_img, 1, np.pi / 180, 15,np.array([]), 50,20)
            # print(lines)

            slopes = findSlope(lines)

            image = findContours(image)
            # if lines is not None:
            #     avg_lines = avg_slope(processed_img,lines)
                # draw_lines(process_img,avg_lines)
            if lines is not None:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        length = np.sqrt([(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)])
                        if length>200:    
                            cv2.line(image, (x1, y1), (x2, y2),(0, 255, 0), 3)

                    # new_screen= processed_img
            cv2.imshow('window', image)
        else:
            pass


        last_time = time.time()

        # plt.imshow(new_screen)
        #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        # cv2.waitKey(0)

        try:
            absSlopes = np.abs(slopes)
            maxSlope=max(absSlopes)
            # print(maxSlope)
            index = np.where(absSlopes == maxSlope)
            finalslope=slopes[index[0][0]]


        except:
            pass

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

            break
main()