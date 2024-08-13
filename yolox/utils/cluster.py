#谱聚类
from sklearn.cluster import SpectralClustering
import numpy as np
import cv2


def calculate_centers(tlwhs):
    """
    计算每个预测框的中心点坐标。
    
    参数:
        tlwhs (list of list): 目标边界框列表，格式为 [左上角X, 左上角Y, 宽度, 高度]。
        
    返回:
        centers (numpy.array): 包含所有中心点坐标的数组，形状为 (n_boxes, 2)。
    """
    centers = []
    for tlwh in tlwhs:
        x1, y1, w, h = tlwh
        center_x = x1 + w / 2
        center_y = y1 + h / 2
        centers.append([center_x, center_y])
    return np.array(centers)

def spectral_clustering(centers, n_clusters=2):
    """
    对中心点坐标进行谱聚类。
    
    参数:
        centers (numpy.array): 包含所有中心点坐标的数组。
        n_clusters (int): 聚类数量，默认为2。
        
    返回:
        labels (numpy.array): 每个中心点所属的聚类标签。
    """
    clustering = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0)
    print(clustering)
    labels = clustering.fit_predict(centers)
    return labels










if __name__ == "__main__":

    # 假设这是从plot_tracking函数中获得的tlwhs
    tlwhs = [[40, 328, 201, 297], [275, 265, 192, 272], [477, 408, 254, 242], 
             [401, 373, 261, 238], [470, 337, 123, 107], [149, 340, 255, 139], [7, 211, 143, 249],
               [327, 270, 258, 271], [300, 20, 277, 129], [289, 93, 209, 232]]  # 这里应该是具体的坐标列表

    # 计算中心点
    centers = calculate_centers(tlwhs)

    # 进行谱聚类
    n_clusters = 3  # 可以根据实际情况调整聚类数目
    cluster_labels = spectral_clustering(centers, n_clusters=n_clusters)

    # 打印或处理聚类结果
    for i, label in enumerate(cluster_labels):
        print(f"Center {i}: Cluster {label},{centers[i]}")
    # clusters = {}
    # for i, (tlwh, label) in enumerate(zip(tlwhs, cluster_labels)):
    #     if label not in clusters:
    #         clusters[label] = {'boxes': [], 'min_x': float('inf'), 'min_y': float('inf'),
    #                        'max_x': float('-inf'), 'max_y': float('-inf')}
    #     clusters[label]['boxes'].append(tlwh)
    #     clusters[label]['min_x'] = min(clusters[label]['min_x'], tlwh[0])
    #     clusters[label]['min_y'] = min(clusters[label]['min_y'], tlwh[1])
    #     clusters[label]['max_x'] = max(clusters[label]['max_x'], tlwh[0] + tlwh[2])
    #     clusters[label]['max_y'] = max(clusters[label]['max_y'], tlwh[1] + tlwh[3])
        
    #     # 假设这是我们的图像
    #     image = np.zeros((600, 800, 3), dtype=np.uint8)
    #     # 绘制包围盒
    #     for label, data in clusters.items():
    #         cv2.rectangle(image, (data['min_x'], data['min_y']), (data['max_x'], data['max_y']), (0, 255, 0), 2)

    #     # 显示图像
    # cv2.imwrite("a.jpg", image)
    
