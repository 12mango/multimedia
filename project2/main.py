import os
from scipy.spatial import KDTree
import cv2
import numpy as np
from matplotlib import pyplot as plt
# import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
# from sklearn.preprocessing import Imputer
import sift
import hu_moments
import color_moments
import color_hist


def read_file_names_from_file(file):
    """
    :param :file 数据集下面有个txt里面有所有类别的名称
    :return: 类别名列表
    """

    with open(file, 'r') as f:
        files = []
        for line in f:
            files.append(line.split('\n')[0])
    return files


def query_KNN(data, id, n):
    """
    :param :data 数据
    :param :n 第三维
    :return: :kNN的序号 不包含自己
    """

    kNN = 98
    data = np.asarray(data)
    """
    tree = KDTree(data)
    _, nearest = tree.query(x=data[0], k=kNN, p=2)
    print(nearest)
    print(names[nearest])
    print(Class[nearest])
    """
    dis = [sum([np.linalg.norm(data[id][j] - data[i][j]) for j in range(n)]) for i in range(len(data))]
    nearest = [x[0] for x in sorted(enumerate(dis), key=lambda x: x[1])]  # np.argsort()
    nearest = nearest[1:kNN + 1]
    return nearest


def img_splice(img_list):
    """
    把检索的结果拼起来
    :param img_list:检索的结果 5张图片
    """

    ret = np.zeros(shape=(300, 1700, 3), dtype=np.uint8)
    for i, img in enumerate(img_list):
        img = cv2.resize(img, (300, 300))
        for j in range(300):
            for k in range(300):
                ret[j][i * 350 + k] = img[j][k]
    # print(ret)
    cv2.imshow('result', ret)
    cv2.waitKey(0)


def calscore_and_showimg(raw_data, nearest, Class, target_id, retrieve_num):
    """
    计算分数 展示检索结果
    :param raw_data: 图像数组
    :param nearest: knn 序号
    :param Class: 类别数
    :param target_id: 检索目标序号
    :param retrieve_num: 检索图像数
    :return:
    """
    target_Class = Class[target_id]
    score = np.sum(Class[nearest] == target_Class)
    #print(names[nearest])
    print(score)
    raw_data = np.asarray(raw_data)
    img_splice(raw_data[nearest[:retrieve_num]])


def retrieval_imgs(raw_data, img_features,Class):
    """
    检索图像
    :param 图像列表 类别列表
    :return:
    """
    retrieve_num = 5  # 检索数量
    target_id = 0  # 目标图像的id

    nearest = query_KNN(img_features, target_id, 1)
    calscore_and_showimg(raw_data, nearest, Class, target_id, retrieve_num)

    # data=[sys_moments(img) for img in raw_data]  # 计算Hu矩
    # data=color_hist(raw_data)
    # data=color_moments(raw_data)
    # print(data)

    # nearest=query_KNN(data,7)
    # print(nearest)
    # print(names[nearest])


if __name__ == '__main__':

    # 获取类别列表
    data_dir = "./Images"
    things = os.listdir(data_dir)
    kinds = []
    for file in things:
        if file == "name.txt":
            kinds = read_file_names_from_file(data_dir + '/' + file)
    # print(kinds)

    # 读取图像数据存入列表
    raw_data = []
    Class = []
    names = []
    for i, kind in enumerate(kinds):
        kind_dir = data_dir + '/' + kind
        imgs = os.listdir(kind_dir)
        for img in imgs:
            fname = kind_dir + '/' + img
            raw_data.append(cv2.imread(fname))
            Class.append(i)
            names.append(fname)
    names = np.asarray(names)
    Class = np.asarray(Class)

    #img_features = color_hist.color_hist(raw_data)
    #img_features = color_moments.color_moments(raw_data)
    #img_features = hu_moments.sys_moments(raw_data)

    Clusters = 3
    des_list, centers = sift.getClusterCentures(raw_data, Clusters)
    img_features = sift.get_all_features(des_list, centers, Clusters)
    retrieval_imgs(raw_data, img_features, Class)
