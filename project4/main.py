import base64
import os
import cv2
import requests
import matplotlib.pyplot as plt
import time
import queue
import numpy as np
import color_hist
from sklearn.cluster import KMeans


def extract_video_frames(videoName):
    # 在这里把后缀接上
    video_path = os.path.join("video/", videoName + '.flv')
    times = 0
    frameFrequency = 1  # 提取视频的频率，每10帧提取一个
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


def abstract(path_material, begin, end, step_size):
    """
    计算直方图
    """
    imgs = []
    for i in range(begin, end, step_size):
        fname = path_material % str(i)
        print(fname)
        img = cv2.imread(fname)
        if img is None:
            break
        imgs.append(img)
    hist = np.array(color_hist.color_hist_2(imgs))
    np.save('video/hist/hist2.npy', np.array(hist))


def find_abrupt_from_hist():
    """
    颜色直方图法 找突变镜头
    :return:
    """
    hist = np.load('video/hist/hist.npy')
    delta = [np.linalg.norm(hist[i] - hist[i + 1]) for i in range(len(hist) - 1)] # 相邻帧差值

    # 展示折线图
    """
    print(delta)
    plt.plot(range(begin,end-step_size,step_size),delta, 'r-')
    plt.show()
    """

    # 筛选突变镜头
    ans = []
    for i in range(len(delta)):
        if delta[i] > 0.15:
            ans.append(str((i + 1) * 10))
    print(','.join(ans))


def find_abrupt_from_spatio_temporal(begin, end, step_size, path_material):
    """
    时空切片 找突变镜头
    :return:
    """

    imgs = []
    for i in range(begin, end, step_size):
        fname = path_material % str(i)
        # print(fname)
        img = cv2.imread(fname)
        if img is None:
            break
        imgs.append(img)

    row = imgs[0].shape[0]  # 1080
    column = imgs[0].shape[1]  # 1920
    rows = [row // 4, row // 2, row // 4 * 3]
    columns = [column // 4, column // 2, column // 4 * 3]

    # 构造时空切片
    slice = np.zeros(shape=(row, len(imgs), 3), dtype=np.uint8)
    for i, img in enumerate(imgs):
        slice[:, i, :] = img[:, columns[0], :]

    # cv2.imshow('result', slice)
    # cv2.waitKey(0)

    # 求每一帧的突变点个数
    pixels = []
    for j in range(slice.shape[1]):
        if j == 0:
            continue
        delta = 10
        tot = 0
        for i in range(slice.shape[0]):
            up = max(0, i - delta - 1)
            down = min(slice.shape[0], i + delta + 1)
            tot += int(all([any([abs(int(slice[i][j][k]) - int(slice[I][j - 1][k])) > 10 for k in range(3)]) for I in
                            range(up, down)]))  # 在RGB中有一个差值大于10则不匹配 所有都不匹配则是突变点
        pixels.append(tot)

    # 突变点个数大于阈值则是突变镜头
    ans = []
    for i in range(len(pixels)):
        if pixels[i] > 200:
            ans.append(str(i))
    print(','.join(ans))

    # plt.plot(range(begin, end - step_size, step_size), pixels, 'r-')
    # plt.show()


def find_gradudal_from_slidingwindow(begin, end, step_size, path_material):
    """
    滑动窗口 求渐变镜头
    :return:
    """
    alpha = 3
    size = 3

    now = 0
    tot_value = 0
    tot_cnt = 0
    ans = []
    hist = np.load('video/hist/hist.npy')
    delta = [np.linalg.norm(hist[i] - hist[i + 1]) for i in range(len(hist) - 1)]

    plt.plot(range(begin, end - step_size, step_size), delta, 'r-')
    plt.show()

    # 滑动窗口
    while True:
        # 窗口超出范围
        if now + size > len(delta):
            break
        window = [delta[i] for i in range(now, now + size)]
        if tot_cnt == 0:
            tot_value = sum(window)
            tot_cnt = size
        else:
            peek = max(window)
            if peek < alpha * tot_value / tot_cnt:  # 小于alpha*局部平均值
                tot_value += window[size - 1]
                tot_cnt += 1 # 最后一帧加入平均值
            else:  # 大于alpha*局部平均值
                # print(peek*tot_cnt/tot_value)
                # print(str((now + np.argmax(window) + 1) * 10))
                ans.append(str((now + np.argmax(window) + 1) * 10))
                tot_value = 0
                tot_cnt = 0
                now += np.argmax(window) + 1
        now += 1
    print(','.join(ans))


def find_gradudal_from_spatio_temporal(begin, end, step_size, path_material):
    """
    时空切片 找渐变镜头
    :return:
    """

    #print("save pixels")
    #np.save('video/hist/pixels.npy', np.array(pixels))

    pixels=np.load('video/hist/pixels.npy')

    alpha = 10
    beta = 15
    size = 60

    now = 0
    tot_value = 0
    tot_cnt = 0
    ans = []
    delta = pixels.tolist()

    """
    # 对±w的邻域进行平滑
    w = 1
    tmp = [0] * w + delta + [0] * w
    delta = [np.mean([tmp[j] for j in range(i - w, i + w + 1)]) for i in range(w, len(tmp) - w)]
    """

    #plt.plot(range(begin, end - step_size, step_size), delta, 'r-')
    #plt.show()
    while True:
        if now + size > len(delta):
            break
        window = [delta[i] for i in range(now, now + size)]
        if tot_cnt == 0:
            tot_value = sum(window)
            tot_cnt = size
        else:
            peek = max(window)
            if peek <= alpha * tot_value / tot_cnt:  # 小于alpha*局部平均值 注意peek==0
                tot_value += window[size - 1]
                tot_cnt += 1
            elif peek > beta * tot_value / tot_cnt:  # 大于beta*局部平均值
                tot_value = 0
                tot_cnt = 0
                now += np.argmax(window)
            else:  # 在alpha和beta*局部平均值之间
                # print(str(now + np.argmax(window)))
                # print(peek*tot_cnt/tot_value)
                ans.append(str(now + np.argmax(window)))
                tot_value = 0
                tot_cnt = 0
                now += np.argmax(window)
        now += 1
    print(','.join(ans))

if __name__ == '__main__':
    # 视频ID
    ID = ""

    # 初始化变量
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    path_material = './video/material' + ID + '/%s.jpg'  # 视频转为图片存放的路径（帧）
    begin = 10  # 处理图像的起始序号
    end = 2200  # 截止序号
    step_size = 10  # 步长
    videoName = "material" + ID  # 视频的文件名

    # extract_video_frames(videoName)
    # abstract(path_material, begin, end, step_size)

    # hist = np.load('video/hist/hist.npy')
    # delta = [np.linalg.norm(hist[i] - hist[i + 1]) for i in range(len(hist) - 1)]
    # find_abrupt_from_spatio_temporal(begin, end, step_size, path_material)

    hists = np.load('video/hist/hist2.npy')

    mean=[sum([i*hist[i] for i in range(200)]) for hist in hists] # 均值==期望
    var =[sum([(i-mean[i])**2*hist[i] for i in range(200)]) for hist in hists] # 离散型方差
    delta = [var[i]-var[i-1] for i in range(1,len(var))] # 求相邻帧差值

    plt.plot(range(begin, end-step_size, step_size), delta, 'r-')
    plt.show()

    #find_gradudal_from_slidingwindow(begin, end, step_size, path_material)
    #find_gradudal_from_spatio_temporal(begin, end, step_size, path_material)