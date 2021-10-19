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

# Compute low order moments(1,2,3)
def color_moments(raw_data):
    """
    计算颜色矩
    :param raw_data: 图像列表
    :return: data:颜色矩
    """

    data = []
    for img in raw_data:
        # Convert BGR to HSV colorspace
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Split the channels - h,s,v
        h, s, v = cv2.split(hsv)

        # Initialize the color feature
        color_feature = []
        # N = h.shape[0] * h.shape[1]
        # The first central moment - average
        h_mean = np.mean(h)  # np.sum(h)/float(N)
        s_mean = np.mean(s)  # np.sum(s)/float(N)
        v_mean = np.mean(v)  # np.sum(v)/float(N)
        color_feature.extend([h_mean, s_mean, v_mean])
        # The second central moment - standard deviation
        h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
        s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
        v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
        color_feature.extend([h_std, s_std, v_std])
        # The third central moment - the third root of the skewness
        h_skewness = np.mean(abs(h - h.mean()) ** 3)
        s_skewness = np.mean(abs(s - s.mean()) ** 3)
        v_skewness = np.mean(abs(v - v.mean()) ** 3)
        h_thirdMoment = h_skewness ** (1. / 3)
        s_thirdMoment = s_skewness ** (1. / 3)
        v_thirdMoment = v_skewness ** (1. / 3)
        color_feature.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])

        data.append(color_feature)
    return data
    # return query_KNN(data)

