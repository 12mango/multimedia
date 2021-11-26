import base64
import os
import cv2
import requests
from aip import AipOcr  # 百度AI的文字识别库
import matplotlib.pyplot as plt
import time
import queue
import numpy as np
from OCR import *
from extract_frame import *


def view_img(img):
    """
    显示图像
    :param img 待显示图像
    """

    cv2.imshow("img", img)
    cv2.waitKey()


vis = None # 染色访问数组

def BFS(img, x, y, flag):
    """
    广度优先搜索 染色 找图像边缘的外接矩形
    :param x,y 搜索的起始坐标
    :param flag 是否删除搜索到的点 True/False 删除/不删除
    :ret ret 外接矩形是否符合阈值
    """

    class POS:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    Q = queue.Queue()

    # 八方向坐标
    dx = [0, 1, 0, -1, 1, -1, 1, -1]
    dy = [1, 0, -1, 0, -1, 1, 1, -1]

    global vis
    theta_low = 1
    theta_high = 50
    n = img.shape[0]
    m = img.shape[1]
    Q.put(POS(x, y))
    vis[x][y] = 1

    # 外接矩形边界初始化
    min_x = x
    max_x = x
    min_y = y
    max_y = y

    # 广度优先搜索
    while not Q.empty():
        top = Q.get()
        for i in range(8):
            X = dx[i] + top.x
            Y = dy[i] + top.y
            if X < 0 or X >= n or Y < 0 or Y >= m or img[X][Y] == 0:
                continue
            if flag:
                img[X][Y] = 0
            elif vis[X][Y]:
                continue
            min_x = min(min_x, X)
            max_x = max(max_x, X)
            min_y = min(min_y, Y)
            max_y = max(max_y, Y)
            vis[X][Y] = 1
            tmp = POS(X, Y)
            Q.put(tmp)
    len_x = (max_x - min_x + 1)
    len_y = (max_y - min_y + 1)
    """
    if not flag:
        print(len_x, len_y)
    """
    """
    ret = True
    if len_x<=theta_low and len_y<=theta_low:
        ret=False
    if len_x>=theta_high and len_y>=theta_high:
        ret=False
    if len_x<=theta_low and len_y<=theta_high and
    """
    ret = len_x >= theta_low and len_x <= theta_high and len_y >= theta_low and len_y <= theta_high

    return ret


def size_filter(img):
    """
    尺寸限制
    :param img 待处理图像
    """

    # np.set_printoptions(threshold=np.inf)
    # print(img)

    n = img.shape[0]
    m = img.shape[1]
    global vis
    vis = np.zeros((n, m))

    #扫描图像 对每个没有染色的白色像素进行搜索
    for i in range(n):
        for j in range(m):
            if img[i][j] > 0:
                # 染色 找外接矩形
                flag = BFS(img, i, j, False)
                # 外接矩形不符合尺寸
                if not flag:
                    img[i][j] = 0
                    BFS(img, i, j, True)


def find_sucessive_area(arr, theta, n):
    """
    寻找连续行/列
    :param arr 行/列的像素密度列表
    :param theta 阈值
    :param n 列表边界
    :ret ret 包含连续列表信息元组的列表
    """

    ret = []
    i = 0
    while i < n:
        tmp = i
        while tmp < n and arr[tmp]:
            tmp += 1

        if tmp - i > theta:
            ret.append((i, tmp))
        i = tmp
        i += 1
    return ret


def caption_area_location(img):
    """
    字幕区域定位
    :param img 待处理图像
    """

    theta = 45
    n = img.shape[0]
    m = img.shape[1]
    row = np.zeros(n)
    column = np.zeros(m)
    # 统计区域像素
    for i in range(n):
        for j in range(m):
            x = img[i][j]
            if x > theta:
                row[i] += 1
                column[j] += 1

    # 求灰度比例
    row /= m
    column /= n

    # 根据阈值分段
    row = row > 0.08
    column = column > 0.005

    """
    for i in range(n):
        for j in range(m):
            if img[i][j]>theta:
                img[i][j] *= (row[i] and column[j])
            else:
                img[i][j]=0
    """

    # 根据连续区域 找候选字幕行和列
    alter_caption_row = find_sucessive_area(row, 5, n)
    print(alter_caption_row)
    alter_caption_column = find_sucessive_area(column, 10, m)
    print(alter_caption_column)

    # 候选字幕区域
    class AREA:
        def __init__(self, U, D, L, R):
            self.U = U
            self.D = D
            self.L = L
            self.R = R

    alter_caption_rect = []

    # 候选行和列组合成候选区域
    for x in alter_caption_row:
        for y in alter_caption_column:
            alter_caption_rect.append(AREA(x[0], x[1], y[0], y[1]))

    # 区域合并
    while True:
        flag = False
        tmp = []
        vis = [False for _ in alter_caption_rect]
        for i in range(len(alter_caption_rect)):
            for j in range(len(alter_caption_rect)):
                if i == j:
                    continue
                if vis[i] or vis[j]:
                    continue
                x = alter_caption_rect[i]
                y = alter_caption_rect[j]
                # hori = max(abs(x.L-y.L),abs(x.R-y.R))
                # vert = abs(x.D-y.U)
                # 左右区域合并
                hori = abs(x.R - y.L)
                vert = max(abs(x.U - y.U), abs(x.D - y.D))
                theta_2 = 5
                if hori < theta_2 and vert < theta_2:
                    vis[i] = vis[j] = True
                    flag = True
                    new_area = (AREA(min(x.U, y.U), max(x.D, y.D), min(x.L, y.L), max(x.R, y.R)))
                    tmp.append(new_area)
        for i in range(len(alter_caption_rect)):
            if not vis[i]:
                tmp.append(alter_caption_rect[i])
        alter_caption_rect = tmp
        if not flag:
            break

    # 过滤不合法区域
    tmp = []
    theta_3 = 10
    theta_4 = 9
    theta_5 = 100
    for x in alter_caption_rect:
        flag = True
        width = x.R - x.L
        height = x.D - x.U
        if width / height < 2:  # 宽高比不合适
            flag = False
        if x.U < theta_3 or n - x.D < theta_3 or x.L < theta_3 or m - x.R < theta_3:  # 字幕区域离图像边界过近
            flag = False
        if x.U < n / 8:  # 字幕太靠上
            flag = False
        if height < theta_4:  # 字幕太小看不清
            flag = False
        if width * height < theta_5:  # 区域面积过小
            flag = False
        if flag:
            tmp.append(x)

    # 保留最大区域
    id = 0
    max_area = 0
    for i in range(len(tmp)):
        # view_img(img[x.U:x.D, x.L:x.R])
        x = tmp[i]
        width = x.R - x.L
        height = x.D - x.U
        if width * height > max_area:
            id = i
            max_area = width * height

    return (tmp[id].U, tmp[id].D, tmp[id].L, tmp[id].R)

"""
def binarization(img):
    thresh = 1
    _, img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)  # 输入灰度图，输出二值图
"""

"""
def expansion(img):  # 扩大黑色区域
    IMG = np.copy(img)
    n = img.shape[0]
    m = img.shape[1]
    for i in range(n):
        for j in range(m):
            if img[i][j]:
                continue
            for k in range(8):
                X = dx[k] + i
                Y = dy[k] + j
                if X < 0 or X >= n or Y < 0 or Y >= m:
                    continue
                IMG[X][Y] = 0
    return IMG
"""

def relocation(img, caption_area):
    """
    对原图像的字幕区域重新切割
    :param caption_area 切割区域
    """

    # 切割
    img = img[caption_area[0]:caption_area[1], caption_area[2]:caption_area[3]]

    # 灰度化
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img


def abstract(path_material,path_captions,begin, end, step_size):
    """
    截取字幕
    """

    for i in range(begin, end, step_size):

        fname = path_material % str(i)
        print(fname)
        img = cv2.imread(fname)
        origin_img = img
        if img is None:
            break
        #print(img.shape)

        """
        # 灰度化+二值化+手工选取字幕区域
        cropped = img
        imgray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        thresh = 200
        ret, binary = cv2.threshold(imgray, thresh, 255, cv2.THRESH_BINARY)  # 输入灰度图，输出二值图
        binary1 = cv2.bitwise_not(binary)  # 取反
        cv2.imwrite(path_img % str(i), binary1)
        """

        # 灰度化
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        """
        # sobel边缘检测
        img_a = cv2.Sobel(img, cv2.CV_16S, 1, 0,ksize=5)  # 垂直
        #view_img(cv2.convertScaleAbs(img_a))
        img_b = cv2.Sobel(img, cv2.CV_16S, 0, 1,ksize=5)  # 水平
        #view_img(cv2.convertScaleAbs(img_b))
        img = (img_a+img_b)/2
        # 浮点型转成uint8型
        img = cv2.convertScaleAbs(img)
        #view_img(img)
        """

        # Laplace边缘检测
        img = cv2.Laplacian(img, cv2.CV_8U, ksize=5)

        # 尺寸限制
        size_filter(img)

        # 字幕区域定位
        caption_area = caption_area_location(img)

        # 提取原图像字幕区域
        origin_img = relocation(origin_img, caption_area)
        cv2.imwrite(path_captions % str(i), origin_img)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # 视频ID
    ID = "2"

    # 初始化变量
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    path_material = './video/material' + ID + '/%s.jpg'  # 视频转为图片存放的路径（帧）
    path_img = './video/img' + ID + '/%s.jpg'  # 图片经过边缘提取后存放的路径
    path_captions = './video/captions' + ID + '/%s.jpg'  # 图片经过边缘提取后存放的路径
    begin = 10  # 处理图像的起始序号
    end = 20  # 截止序号
    step_size = 10  # 步长
    videoName = "material" + ID  # 视频的文件名

    extract_video_frames(videoName)
    abstract(path_material,path_captions,begin, end, step_size)
    captions_OCR(path_captions, begin, end, step_size)

    # easyOCR 还未解决环境问题
    """
    reader = easyocr.Reader(['ch_sim', 'en'])
    result = reader.readtext('./video/captions2/10.jpg')
    """
