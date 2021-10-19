import os
from scipy.spatial import KDTree
import cv2
import numpy as np
from matplotlib import pyplot as plt
#import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.cluster  import KMeans
#from sklearn.preprocessing import Imputer

def display_hist(hsv):
    """
    显示颜色直方图
    :param :hsv 图像的hsv 三维列表
    """

    hist = []
    for row in hsv:
        for pix in row:
            H, S, V = pix[0], pix[1], pix[2]

            # 分段函数
            H_p = (H >= 11) + (H >= 21) + (H >= 38) + (H >= 78) + (H >= 96) + (H >= 136) + (H >= 148)
            if H >= 158:
                H_p = 0
            S_p = (S >= 52) + (S >= 179)
            V_p = (V >= 52) + (V >= 179)

            G = 9 * H_p + 3 * S_p + V_p
            hist.append(G)

    # print(hist)
    plt.hist(hist, np.arange(0, 72), density=True)
    plt.show()


def color_hist(raw_data):
    """
    计算颜色直方图
    :param :raw_data rgb图像
    :return: 颜色直方图
    """

    hsv = [cv2.cvtColor(img, cv2.COLOR_BGR2HSV) for img in raw_data]
    display_hist(hsv[0])
    data = []
    for i in range(len(hsv)):
        hist = [0 for _ in range(72)]
        for row in hsv[i]:
            for pix in row:
                H, S, V = pix[0], pix[1], pix[2]
                H_p = (H >= 11) + (H >= 21) + (H >= 38) + (H >= 78) + (H >= 96) + (H >= 136) + (H >= 148)
                if H >= 158:
                    H_p = 0
                S_p = (S >= 52) + (S >= 179)
                V_p = (V >= 52) + (V >= 179)
                G = 9 * H_p + 3 * S_p + V_p
                hist[G] += 1
        hist_sum = sum(hist)
        hist = [x / hist_sum for x in hist]
        # print(hist)
        data.append(hist)
    return data