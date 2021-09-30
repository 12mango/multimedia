import base64
import os
import cv2
import requests
from aip import AipOcr  # 百度AI的文字识别库
import matplotlib.pyplot as plt
import time
import queue
import numpy as np
import random

def view_img(img):
    """
    显示图像
    :param img 待显示图像
    """

    cv2.imshow("img", img)
    cv2.waitKey()

def task1(file_name):
    """
    将一个图像灰度化
    """
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def task2(file_name,rate):
    """
    有损压缩
    :param file_name:
    :param rate: 压缩度 jpg 0~100 数值越小 压缩比越高
    :return:
    """
    img = cv2.imread(file_name)
    cv2.imwrite("saveImg.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, rate])

def Embed(file_from,file_to,str):
    """
    将字符串隐藏进图像
    :param file_from: 源文件
    :param file_to: 目标文件
    :param str: 待隐藏的字符串
    :return:
    """
    img = cv2.imread(file_from)
    n = img.shape[0]
    m = img.shape[1]
    for i in range(n):
        for j in range(m):
            img[i][j][2]=ord(str[i*m+j]) # 字符转ascii
    cv2.imwrite(file_to,img)

def Extract(file_to):
    """
    将图像中隐藏的字符串提取出来
    :param file_to: 文件名
    :return: 提取的字符串
    """
    img = cv2.imread(file_to)
    arr = img[:,:,2]
    arr =list(np.concatenate(arr))
    return ''.join([chr(ch) for ch in arr])

def task3(file_from,file_to):
    """
    把文字字符串的每一位，替换掉BMP图片的每一个像素的红色的最后一位
    :param file_from: 源文件
    :param file_to: 目标文件
    """
    img = cv2.imread(file_from) # BGR
    n=img.shape[0]
    m=img.shape[1]
    str = [random.choice('zyxwvutsrqponmlkjihgfedcba') for _ in range(n*m)]
    print(str[:10])
    Embed(file_from,file_to,str)
    print(Extract(file_to)[:10])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_name="lxh.bmp"
    file_to="target.bmp"
    """
    # 不同压缩比计时
    start=time.perf_counter()
    task2(file_name, 100) # 慢
    end=time.perf_counter()
    print(end - start)
    start = time.perf_counter()
    task2(file_name, 0) # 快
    end = time.perf_counter()
    print(end-start)
    """
    task3(file_name,file_to)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
