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

def sys_moments(img):
    """
    opencv_python自带求矩以及不变矩的函数
    :param img: 灰度图像，对于二值图像来说就只有两个灰度0和255
    :return: 返回以10为底对数化后的hu不变矩
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    moments = cv2.moments(gray)  # 返回的是一个字典，三阶及以下的几何矩（mpq）、中心矩(mupq)和归一化的矩(nupq)
    humoments = cv2.HuMoments(moments)  # 根据几何矩（mpq）、中心矩(mupq)和归一化的矩(nupq)计算出hu不变矩
    # 因为直接计算出来的矩可能很小或者很大，因此取对数好比较,这里的对数底数为e,通过对数除法的性质将其转换为以10为底的对数，一般是负值，因此加一个负号将其变为正的
    humoment = -(np.log(np.abs(humoments))) / np.log(10)
    return humoment
