import base64
import os
import cv2
import requests
from aip import AipOcr  # 百度AI的文字识别库
import matplotlib.pyplot as plt
import time
import queue
import numpy as np

def extract_video_frames(videoName):
    # 在这里把后缀接上
    video_path = os.path.join("video/", videoName + '.mp4')
    times = 0
    frameFrequency = 10  # 提取视频的频率，每10帧提取一个
    outPutDirName = './video/' + videoName + '/'
    if not os.path.exists(outPutDirName):
        # 如果文件目录不存在则创建目录
        os.makedirs(outPutDirName)
    camera = cv2.VideoCapture(video_path)
    while True:
        times += 1
        res, image = camera.read()
        if not res:
            print('not res , not image')
            break
        if times % frameFrequency == 0:
            cv2.imwrite(outPutDirName + str(times) + '.jpg', image)  # 文件目录下将输出的图片名字命名为10.jpg这种形式
            print(outPutDirName + str(times) + '.jpg')
    print('图片提取结束')