from ColorSegment import *
from darknet import *
if __name__ == '__main__':
    # 输入图片的路径
    # image = './image/4.jpg'
    # # 创建NumPy数组，设置沙滩颜色区间
    # lower = np.array([0,70,70])  # 颜色下限
    # upper = np.array([120,255,255])  # 颜色上限
    # 输入图片的路径
    image = './image/5.jpg'
    lower = np.array([100, 160, 180])  # 颜色下限
    upper = np.array([180, 220, 255])  # 颜色上限
    img, contours = ColorSegement(image,lower=lower,upper=upper)
    # cv2.imshow("contour",img)
    cv2.imwrite("contour.jpg",img)
    # 调用performDetect函数，返回人物坐标数组
    a,person = performDetect("contour.jpg")
    # print(person)
    PeopleInSea = 0
    PeopleOnBeach = 0
    for p in person:
        # 人的脚的坐标
        xFootCoord = p[2][0]
        yFootCoord = int((p[2][1] + p[3][1])/2)
        FootCoordPoint = (xFootCoord,yFootCoord)
        # 该函数确定该点是在轮廓内部，外部还是位于边缘上（或与顶点重合）。
        # 当measureDist = false时，返回值分别为+ 1，-1和0。它相应地返回正（内部），负（外部）或零（边缘）值。
        res = cv2.pointPolygonTest(contours, FootCoordPoint, measureDist=False)
        if res == -1:
            PeopleInSea += 1
        else:
            PeopleOnBeach +=1
    print("Num of People In Sea: ",PeopleInSea)
    print("Num of People On Beach: ",PeopleOnBeach)
    cv2.waitKey(0)
