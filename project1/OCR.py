"""
    调用百度API的OCR来识别字幕
"""

import base64
import os
import cv2
import requests
from aip import AipOcr  # 百度AI的文字识别库
import matplotlib.pyplot as plt
import time
import queue
import numpy as np


# 定义一个函数，用来访问百度API，
def requestApi(img):
    request_url = "https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic"
    params = {"image": img, 'language_type': 'CHN_ENG'}
    access_token = '24.8b5758ec70247b64ccb608d52e720c1c.2592000.1634969842.282335-24889932'
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    results = response.json()
    return results


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        # 将读取出来的图片转换为b64encode编码格式
        return base64.b64encode(fp.read())


def captions_OCR(path_captions, begin, end, step_size):
    for i in range(begin, end, step_size):
        time.sleep(0.5)  # api-qpr=2
        fname = path_captions % str(i)
        print(fname)
        image = get_file_content(fname)
        try:
            # print(requestApi(image))
            ret = requestApi(image)
            # print(ret)
            results = ret['words_result']
            if ret.get('error_msg'):
                print(ret['error_msg'])
            for item in results:
                print(item['words'])
        except Exception as e:
            print(e)
