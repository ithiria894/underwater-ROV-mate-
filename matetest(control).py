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

# def make_coords(img,line_param):
#     slope,intercept=line_param

#     y1 = img.shape[0]
#     y2 = int((y1*(3/5)))
#     x1 = int((y1-intercept)/slope)
#     x2 = int((y2-intercept)/slope)
#     try:
#         return np.array((x1,y1,x2,y2))    
#     except UnboundLocalError:
#         pass


# def avg_slope(img,lines):
#     left_fit =[]
#     right_fit=[]

#     for line in lines:
#         x1,y1,x2,y2=line.reshape(4)
#         parameters = np.polyfit((x1,x2),(y1,y2),1)
#         try:
#             slope = parameters[0]
#         except TypeError:
#             slope = 0
#         try:
#             intercept = parameters[1]
#         except TypeError:
#             intercept = 0
#         if slope <0:
#             left_fit.append((slope,intercept))
#         else:
#             right_fit.append((slope,intercept))
#     if left_fit:
#         left_fit_avg=np.average(left_fit,axis=0)
#         left_line=make_coords(img,left_fit_avg)
#     if right_fit:
#         right_fit_avg=np.average(right_fit,axis=0)
#         right_line=make_coords(img,right_fit_avg)

#     return np.array((x1,y1,x2,y2))

def draw_lines(img, lines):
    try:
    # if lines is not None:
        for line in lines:
            if line is not None:
                # for x1, y1, x2, y2 in line:

                # print(line)
                x1,y1,x2,y2=line.reshape(4)
                # print(line.reshape(4))
                # if x1!=x2 and y1!=y2:
                #     slope = (y2-y1)/(x2-x1)
                #     # print(slope)
                #     slopes.append(slope)

                cv2.line(img,(x1,y1),(x2,y2),(0, 255, 0),10)
            # coords = line[0]
            # cv2.line(img, (coords[0],coords[1]), (coords[2],coords[3]), [255,0,0], 3)
                # slopes.append(slope)
        # print(slopes)
        # absSlopes = np.abs(slopes)
        # maxSlope=max(absSlopes)
        # print(maxSlope)
    except:
        pass
    return img

def findSlope(lines):
    slopes=[]
    try:
    # if lines is not None:
        for line in lines:
            if line is not None:
                # for x1, y1, x2, y2 in line:

                # print(line)
                x1,y1,x2,y2=line.reshape(4)
                # print(line.reshape(4))
                if x1!=x2 and y1!=y2:
                    slope = (y2-y1)/(x2-x1)
                    # print(slope)
                    slopes.append(slope)
        # print(slopes)
        # absSlopes = np.abs(slopes)
        # maxSlope=max(absSlopes)
        # # print(maxSlope)
    except:
        pass
    return slopes

def roi(img):
    vertices = np.array([[10,500],[10,300], [300,200], [500,200], [800,300], [800,500]], np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, [vertices], 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

# def process_img(image):
#     original_image = image
#     # convert to gray
#     processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # edge detection
#     # processed_img = cv2.GaussianBlur(processed_img,(5,5),0) #new
#     # processed_img =  cv2.Canny(processed_img, threshold1 = 50, threshold2=150) #new
#     processed_img =  cv2.Canny(processed_img, threshold1 = 100, threshold2=170,apertureSize=3) #try
#     processed_img = roi(processed_img)
#     lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), minLineLength=15,maxLineGap=5)
#     # print(lines)
#     slopes = findSlope(lines)
#     # if lines is not None:
#     #     avg_lines = avg_slope(processed_img,lines)

#         # draw_lines(process_img,avg_lines)
#     draw_lines(process_img,lines)
#     # print(slopes)

    
    # return processed_img


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
            # new_screen = process_img(screen)
            # print('Frame took {} seconds'.format(time.time()-last_time))
            image = screen
            original_image = image
            # convert to gray
            processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # edge detection
            # processed_img = cv2.GaussianBlur(processed_img,(5,5),0) #new
            # processed_img =  cv2.Canny(processed_img, threshold1 = 50, threshold2=150) #new
            processed_img =  cv2.Canny(processed_img, threshold1 = 100, threshold2=170,apertureSize=3) #try
            processed_img = roi(processed_img)
            lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), minLineLength=15,maxLineGap=5)
            # print(lines)
            slopes = findSlope(lines)
            # print(slopes)

            # if lines is not None:
            #     avg_lines = avg_slope(processed_img,lines)

                # draw_lines(process_img,avg_lines)
            processed_img = draw_lines(image,lines)

            new_screen= processed_img
            cv2.imshow('window', new_screen)
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
        #         print('left')
        #         left()
        #         # PressKey(A)
        #     elif finalslope<0:
        #         right()
        #         print('right')
        #         # PressKey(D)
        #     # else:
        #     #     straight()
        #     #     print('straight')
        # except:
        #     pass

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

            break
main()