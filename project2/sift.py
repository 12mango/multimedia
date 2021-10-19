import os
from scipy.spatial import KDTree
import cv2
import numpy as np
from matplotlib import pyplot as plt
# import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

def getClusterCentures(raw_data,Clusters):
    sift_det = cv2.SIFT_create()
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in raw_data]
    des_mat = np.zeros((1, 128))  # 特征描述子矩阵
    des_list=[]

    # 对每个图片寻找特征点
    for i, img in enumerate(imgs):
        _, des = sift_det.detectAndCompute(img, None)  # 特征点和特征描述子
        if des != []:
            des_mat = np.row_stack((des_mat, des))
            des_list.append(des)
            # print(des.shape)

    des_mat = des_mat[1:, :]

    kmeans = KMeans(n_clusters=Clusters)
    kmeans.fit(des_mat)
    centers = kmeans.cluster_centers_
    return des_list, centers


def des_to_features(des, centers, Clusters):
    """
    单个特征描述子转换成特征向量
    :param Clusters:
    :return:
    """
    feat_vec = np.zeros((1, Clusters), 'float32')
    for i in range(len(des)): # range(128)
        feat_k_rows = np.ones((Clusters, 128), 'float32')  # 全1数组
        feat_k_rows = feat_k_rows * des[i]  #
        feat_k_rows = np.sum((feat_k_rows - centers) ** 2, axis=1)
        id = np.argmin(feat_k_rows)
        feat_vec[0][id] += 1  # 离聚类中心最近的特征描述子 构造直方图
    return feat_vec


def get_all_features(des_list, centers, Clusters):
    """
    所有特征描述子转换成特征向量
    """
    n = len(des_list)
    features = np.zeros((n, Clusters),'float32')
    for i in range(n):
        if des_list[i] != []:
            features[i] = des_to_features(des_list[i], centers, Clusters)
    return features