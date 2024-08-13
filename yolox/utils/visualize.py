#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ["vis"]


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color


def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]
    # 创建一个与原图宽度相同的白色背景，用于顶视图表示
    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    #text_scale = max(1, image.shape[1] / 1600.)
    #text_thickness = 2
    #line_thickness = max(1, int(image.shape[1] / 500.))
    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    radius = max(5, int(im_w/140.))
    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
     # 遍历每一个目标边界框
    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh  #[top,left,weight,height]
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))


        #绘制轨迹，以box下边框中心点坐标为绘制坐标
        # print("start plot draw track..")
        # track_point =tuple(map(int, (x1 + w/2, y1 + h/2)))
        # print(track_point)
        # cv2.line(im, track_point, track_point, color, thickness=5, lineType=cv2.LINE_AA, shift=None)
        # 绘制边界框
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        # # 添加文字标签，显示目标ID和附加信息
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                     thickness=text_thickness)
        
        # #add for 检测聚集程度
        # if len(tlwhs)<3:
        #     continue
        # im = cluster_plot2(tlwhs,im)

    return im

def cluster_plot2(tlwhs, img):
    from sklearn.cluster import SpectralClustering

    # 假设tlwhs是一个包含多个tlwh列表的列表
    # tlwhs = [tlwhs1, tlwhs2, ...]

    # 创建一个空列表来存储每个聚类的中心点
    centers = []

    # 遍历每个tlwh列表
    for tlwh in tlwhs:
         # 假设tlwhs中的每个元素是一个tlwh列表
        x1, y1, w, h = tlwh
        center_x = x1 + w / 2
        center_y = y1 + h / 2
        centers.append([center_x, center_y])

    # 归一化中心点
    min_values = np.min(centers, axis=0)
    max_values = np.max(centers, axis=0)
    normalized_centers = (centers - min_values) / (max_values - min_values)

    # 设置聚类数量
    n_clusters = 3
    clustering = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0)
    labels = clustering.fit_predict(normalized_centers)

    # 创建一个蒙版并绘制多边形
    mask = np.ones_like(img)

    # 遍历每个聚类
    for i, label in enumerate(labels):
        # 获取所有属于当前聚类的中心点
        points = [centers[j] for j, l in enumerate(labels) if l == label]

        # 将中心点转换为多边形顶点
        points = np.array(points, dtype=np.int32)
        points = points.reshape(-1, 1, 2)  # 调整形状以匹配cv2.fillPoly的期望格式

        

        # 绘制多边形
        color = (_COLORS[i] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.fillPoly(mask, [points], color)

    # 将蒙版融合到原图
    img = cv2.addWeighted(img, 1, mask, 0.5, 0)

    # 保存图像
    cv2.imwrite("s.jpg", img)

    # 返回0
    return img


def cluster_plot(tlwhs,img):
    from sklearn.cluster import SpectralClustering #谱聚类
    centers = []
    for tlwh in tlwhs:
        x1, y1, w, h = tlwh
        center_x = x1 + w / 2
        center_y = y1 + h / 2
        centers.append([center_x, center_y])
    centers = np.array(centers)

    #归一化
    min_values = np.min(centers, axis=0)
    max_values = np.max(centers, axis=0)
    
    # 应用归一化公式
    normalized_centers = (centers - min_values) / (max_values - min_values)
    
    n_clusters = 3
    clustering = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0)
    labels = clustering.fit_predict(normalized_centers)
    
    clusters = {}
    for i, (tlwh, label) in enumerate(zip(tlwhs, labels)):
        if label not in clusters:
            clusters[label] = {'boxes': [], 'min_x': float('inf'), 'min_y': float('inf'),
                               'max_x': float('-inf'), 'max_y': float('-inf')}
        clusters[label]['boxes'].append(tlwh)
        clusters[label]['min_x'] = min(clusters[label]['min_x'], tlwh[0])
        clusters[label]['min_y'] = min(clusters[label]['min_y'], tlwh[1])
        clusters[label]['max_x'] = max(clusters[label]['max_x'], tlwh[0] + tlwh[2])
        clusters[label]['max_y'] = max(clusters[label]['max_y'], tlwh[1] + tlwh[3])
    
    # 绘制包围盒
    for label, data in clusters.items():
        cv2.rectangle(img, (int(data['min_x']), int(data['min_y'])), 
                      (int(data['max_x']), int(data['max_y'])), (0, 255, 0), 2)
    
    return img

_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
