"""
基于opencv的海岸线分割：
采用分水岭算法
"""
import cv2
import numpy as np

# 填充函数:用于处理小的轮廓和空洞
def fill_contours(img, contours, h = 50, w = 50):
    c_max = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)

        # 处理掉小的轮廓区域，这个区域的大小自己定义。
        if (area < (h  * w )):
            c_min = []
            c_min.append(cnt)
            # thickness不为-1时，表示画轮廓线，thickness的值表示线的宽度。
            cv2.drawContours(img, c_min, -1, (0, 0, 0), thickness=-1)
            continue
        #
        c_max.append(cnt)

    cv2.drawContours(img, c_max, -1, (255, 255, 255), thickness=-1)
    cv2.imshow("tianchong", img)

img = cv2.imread('image/2.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

print(ret)
cv2.imshow('show',thresh)

#noise removal
#opening operator是先腐蚀后膨胀，可以消除一些细小的边界，消除噪声
kernel=np.ones((3,3),np.uint8)
opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)
cv2.imshow('opening',opening)

sure_bg=cv2.dilate(opening,kernel,iterations=3)
cv2.imshow('bg',sure_bg)

#finding sure foreground area
dist_transfrom=cv2.distanceTransform(opening,cv2.DIST_L2 ,5)
#cv2.imshow('dist_transfrom',dist_transfrom)
ret,sure_fg=cv2.threshold(dist_transfrom,0.7*dist_transfrom.max(),255,0)

cv2.imshow('sure_fg',sure_fg)

#finding unknow region
sure_fg=np.uint8(sure_fg)
unknow=cv2.subtract(sure_bg,sure_fg) #背景-前景
# cv2.imshow('unknow',unknow)

ret,maker=cv2.connectedComponents(sure_fg)
maker=maker+1
maker[unknow==255]=0
maker = cv2.watershed(img,maker)

img[maker == -1] = [0,0,255]
cv2.imshow('result',img)

cv2.waitKey(0)
cv2.destroyAllWindows()