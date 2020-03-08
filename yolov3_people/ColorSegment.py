# coding：utf-8
# 黄色检测
import numpy as np
import cv2

"""
fill_contours 填充函数，将分割线内一些空洞补充。
parameter: img=输入图片，counter=findcounter找到的边界，h，w为高宽，小于hw的会被填充
return：返回填充后的图片
"""
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
    # cv2.imshow("tianchong", img)
    return img

'''
ColorSegement：基于颜色分割算法，分割海水和沙滩
parameter：image=输入源图片，lower=颜色阈值下限，upper=颜色阈值上限
return：返回海水沙滩分割后的图片
'''
def ColorSegement(image, lower, upper):
    # 读取图片文件
    image = cv2.imread(image)
    # cv2.imshow("image",image)
    # 根据阈值找到对应颜色，设置蒙版，与运算截取出沙滩颜色的图片
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    # cv2.imshow("cut", output)

    #去噪
    #先腐蚀后膨胀，可以消除一些细小的边界，消除噪声
    kernel=np.ones((2,2),np.uint8)
    fushi=cv2.morphologyEx(output,cv2.MORPH_OPEN,kernel,iterations=3)
    # cv2.imshow(u'fushi',fushi)
    pengzhang=cv2.dilate(fushi,kernel,iterations=3)
    # cv2.imshow(u"pengzhang",pengzhang)

    # 转成灰度图
    gray = cv2.cvtColor(pengzhang,cv2.COLOR_BGR2GRAY)
    # 二值化，以方便边缘分割
    ret, binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)

    # 设置kernel，用开运算继续去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(10, 10))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("open",opened)
    # 膨胀，使其圆润
    pengzhang2=cv2.dilate(opened,kernel,iterations=3)
    # cv2.imshow("pengzhang2",pengzhang2)

    # findcontour找轮廓
    aa,contours, hierarchy = cv2.findContours(pengzhang2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 填充空洞，再找一次，可以去除空洞，使边界完整
    tianchong = fill_contours(pengzhang2, contours, h = 20, w = 50)
    aa,contours, hierarchy = cv2.findContours(tianchong, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    cv2.imshow("counter",image)
    cv2.waitKey(0)

    return image, contours[0]

if __name__ == "__main__":
    # 输入图片的路径
    image = './image/5.jpg'
    lower = np.array([100, 160, 180])  # 颜色下限
    upper = np.array([180, 220, 255])  # 颜色上限
    img,counter = ColorSegement(image, lower=lower, upper=upper)