# importing all the required libraries 
import cv2
import numpy as np
import math 
import pyautogui as p
import time as t 

# Reading the camera 
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# # to get franme height 
# print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def nothing(x):
    pass

# creating named window
cv2.namedWindow("TrackYourHand")
cv2.resizeWindow("TrackYourHand",(380,300))

# creating trackbars 
cv2.createTrackbar("Thresh","TrackYourHand",0,255,nothing)
cv2.createTrackbar("Lower_H","TrackYourHand",0,255,nothing)
cv2.createTrackbar("Lower_S","TrackYourHand",0,255,nothing)
cv2.createTrackbar("Lower_V","TrackYourHand",0,255,nothing)
cv2.createTrackbar("Upper_H","TrackYourHand",255,255,nothing)
cv2.createTrackbar("Upper_S","TrackYourHand",255,255,nothing)
cv2.createTrackbar("Upper_V","TrackYourHand",255,255,nothing)


while(True):
    _,frame = cap.read()
    frame = cv2.flip(frame,2)
    frame = cv2.resize(frame,(600,500))

    # creating rectangle window in frame 
    cv2.rectangle(frame,(0,1),(300,500),(255,0,0),0)
    crop_image = frame[1:500,0:300]

    # converting crop image to hsv
    hsv = cv2.cvtColor(crop_image,cv2.COLOR_BGR2HSV)

    # detect your hand using trackbar
    l_h = cv2.getTrackbarPos("Lower_H","TrackYourHand")
    l_S = cv2.getTrackbarPos("Lower_S","TrackYourHand")
    l_V = cv2.getTrackbarPos("Lower_V","TrackYourHand")

    U_h = cv2.getTrackbarPos("Upper_H","TrackYourHand")
    U_S = cv2.getTrackbarPos("Upper_S","TrackYourHand")
    U_V = cv2.getTrackbarPos("Upper_V","TrackYourHand")
    
    # lower bound and upper bound of particular color
    l_b = np.array([l_h,l_S,l_V])
    u_b = np.array([U_h,U_S,U_V])

    # creting mask
    mask = cv2.inRange(hsv,l_b,u_b)

    # filtering mask
    fltr = cv2.bitwise_and(crop_image,crop_image, mask=mask)

    mask1 = cv2.bitwise_not(mask)
    th = cv2.getTrackbarPos("Thresh","TrackYourHand")
    _,thresh = cv2.threshold(mask1,th,255,cv2.THRESH_BINARY)

    dilation = cv2.dilate(thresh,(3,3),iterations=6)
    erosion = cv2.erode(thresh,(3,3),iterations=6)
    # finding contours
    cnts,hier = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # print(cnts)
    try:
        cm = max(cnts,key=lambda x: cv2.contourArea(x))
        # print(cm)
        epsilon = 0.0005*cv2.arcLength(cm,True)
        # data = cv2.approxPolyDP(cm,epsilon,True)
        hull = cv2.convexHull(cm)

        cv2.drawContours(crop_image,[cm],-1,(50,50,150),2)
        cv2.drawContours(crop_image,[hull],-1,(0,255,0),2)

        # convexity defects 
        hull = cv2.convexHull(cm,returnPoints=False)
        defects = cv2.convexityDefects(cm,hull)
        # print(defects.shape[0])
        # print("Area==",cv2.contourArea(hull) - cv2.contourArea(cm))
        count_defects = 0
        for i in range(defects.shape[0]):
            s,e,f,d =defects[i,0]

            start = tuple(cm[s][0])
            end = tuple(cm[e][0])
            far = tuple(cm[f][0])
            # cosine rule for finding the angle between fingers 
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14
            # print(angle)
            if angle<=50:
                count_defects+=1
                cv2.circle(crop_image,far,5,[255,255,255],-1)
        # print(count_defects)

        if count_defects == 0:
            cv2.putText(frame, " ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255),2)

        elif count_defects == 1:
            p.press("space")
            cv2.putText(frame, "Play/Pause", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)

        elif count_defects == 2:
            p.press("up")
            cv2.putText(frame, "Volume UP", (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)

        elif count_defects == 3:
            p.press("down")
            cv2.putText(frame, "Volume Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)

        elif count_defects == 4:
            p.press("right")
            cv2.putText(frame, "Forward", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,(0,0,255), 2)
        else:
            pass
    except:
        pass

    cv2.imshow("Frame",frame)
    # cv2.imshow("mask",mask)
    cv2.imshow("filter",fltr)
    # cv2.imshow("mask1",mask1)
    cv2.imshow("dilation",dilation)
    # cv2.imshow("thresh",thresh)
    cv2.imshow("erosion",erosion)

    if(cv2.waitKey(1) & 0xFF)==27:
        break
cap.release()
cv2.destroyAllWindows()

