import numpy as np
from PIL import ImageGrab
import cv2
import time
import pyautogui
import matplotlib.pyplot as plt
from directkeys import PressKey,ReleaseKey, W, A, S, D


# slopes =[]


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
    orig_image = image
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image2 = cv2.threshold(image2, 127, 255, 1)
    image2 = cv2.Canny(image2,30,200)
    # Find contours 
    contours, hierarchy = cv2.findContours(image2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = 200000000000000
    min_area = 100
    areaList=[]
    cxList=[]
    targetCx =None
    if contours is not None:
        for c in contours:    
            
            area = cv2.contourArea(c)
            if min_area < area < max_area:
                areaList.append(area)
                approx = cv2.approxPolyDP(c, 0.03*cv2.arcLength(c,True),False)

                cv2.drawContours(orig_image, [c], 0,(0,255,0), -1)
                # cv2.drawContours(orig_image, [c], 0,(225,0,0), -1)
                x,y,w,h = cv2.boundingRect(c)
                # cv2.rectangle(orig_image3,(x,y),(x+w,y+h),(0,0,255),2)
                cx=x+int(w/2)
                cxList.append(cx)
                cy=y+int(h/2)
                cv2.circle(orig_image, (cx,cy), 5, (0,0,255), -1) 
    
        if areaList:
            maxArea=max(areaList)
            index=np.where(areaList==maxArea)
            targetCx = cxList[index[0][0]]

    return targetCx,orig_image


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(A)
    PressKey(W)
    # ReleaseKey(W)
    ReleaseKey(D)
    # ReleaseKey(A)

def right():
    PressKey(D)
    PressKey(W)
    ReleaseKey(A)
    # ReleaseKey(W)
    # ReleaseKey(D)

def main():
    last_time = time.time()
    while True:

        screen =  np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        if screen is not None:
            # cv2.imshow('window1', screen)

            height=screen.shape[0]
            width=screen.shape[1]
            mp=width/2
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

            original_image,targetCx  = findContours(original_image)
          
            # if lines is not None:
            #     avg_lines = avg_slope(processed_img,lines)
                # draw_lines(process_img,avg_lines)
            if lines is not None:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        length = np.sqrt([(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)])
                        if length>200:    
                            cv2.line(original_image, (x1, y1), (x2, y2),(0, 255, 0), 3)

                    # new_screen= processed_img
            cv2.imshow('window2', original_image)
        else:
            pass


        last_time = time.time()

        # plt.imshow(new_screen)
        #cv2.imshow('window',cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
        # cv2.waitKey(0)

        # try:
        #     absSlopes = np.abs(slopes)
        #     maxSlope=max(absSlopes)
        #     # print(maxSlope)
        #     index = np.where(absSlopes == maxSlope)

        #     finalslope=slopes[index[0][0]]

        #     if finalslope>0:
        #         # print('left')
        #         left()
        #         # PressKey(A)
        #     elif finalslope<0:
        #         right()
        #         # print('right')
        #         # PressKey(D)
        #     else:
        #         straight()
        #         # print('straight')
        # except:
        #     pass

        if targetCx is not None:
            try:
                if targetCx<mp:
                    # right()
                    # print('right')

                    PressKey(D)

                elif targetCx>mp:
                    # print('left')
                    # left()

                    PressKey(A)
                else:
                    # straight()
                    # print('straight')

                    ReleaseKey(A)
                    ReleaseKey(D)
            except:
                pass    

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

            break
main()