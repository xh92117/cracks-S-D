import numpy as np
import torch
from typing import List, Tuple, Dict
import cv2
from sklearn.cluster import DBSCAN

class WCNMSFusion:
    """
    Weighted Cluster-NMS融合算法实现
    结合了WBF和Cluster-NMS的优点，适用于多模型融合
    """
    def __init__(self, iou_threshold: float = 0.5):
        """
        初始化融合器
        Args:
            iou_threshold: IOU阈值，用于聚类
        """
        self.iou_threshold = iou_threshold

    def calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        计算两个边界框的IOU
        Args:
            box1: 第一个边界框 [x1, y1, x2, y2]
            box2: 第二个边界框 [x1, y1, x2, y2]
        Returns:
            IOU值
        """
        # 计算交集区域
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # 计算交集面积
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # 计算并集面积
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0

    def weighted_cluster_nms(self, boxes: List[np.ndarray], 
                           scores: List[np.ndarray],
                           labels: List[np.ndarray],
                           weights: List[float]) -> Dict:
        """
        使用Weighted Cluster-NMS算法融合多个模型的预测框
        Args:
            boxes: 预测框列表
            scores: 置信度列表
            labels: 类别标签列表
            weights: 模型权重列表
        Returns:
            融合后的预测结果
        """
        # 合并所有框和分数
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for i in range(len(boxes)):
            if len(boxes[i]) == 0:
                continue
            all_boxes.extend(boxes[i])
            all_scores.extend(scores[i] * weights[i])  # 应用权重
            all_labels.extend(labels[i])
        
        if not all_boxes:
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'labels': np.array([])
            }
        
        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # 计算IOU矩阵
        n = len(all_boxes)
        iou_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                iou_matrix[i, j] = self.calculate_iou(all_boxes[i], all_boxes[j])
        
        # 使用DBSCAN聚类
        clustering = DBSCAN(
            eps=self.iou_threshold,
            min_samples=1,
            metric='precomputed'
        )
        labels = clustering.fit_predict(1 - iou_matrix)  # 将IOU转换为距离
        
        # 对每个簇进行加权平均
        unique_labels = np.unique(labels)
        fused_boxes = []
        fused_scores = []
        fused_labels = []
        
        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            cluster_boxes = all_boxes[cluster_indices]
            cluster_scores = all_scores[cluster_indices]
            cluster_labels = all_labels[cluster_indices]
            
            # 计算加权平均框
            weights = cluster_scores / np.sum(cluster_scores)
            fused_box = np.average(cluster_boxes, weights=weights, axis=0)
            fused_score = np.max(cluster_scores)
            fused_label = cluster_labels[np.argmax(cluster_scores)]
            
            fused_boxes.append(fused_box)
            fused_scores.append(fused_score)
            fused_labels.append(fused_label)
        
        return {
            'boxes': np.array(fused_boxes),
            'scores': np.array(fused_scores),
            'labels': np.array(fused_labels)
        }

    def fuse_predictions(self, det_pred: Dict, seg_pred: Dict,
                        det_weight: float = 0.6, seg_weight: float = 0.4) -> Dict:
        """
        融合检测和分割模型的预测结果
        Args:
            det_pred: 检测模型预测结果
            seg_pred: 分割模型预测结果
            det_weight: 检测模型权重
            seg_weight: 分割模型权重
        Returns:
            融合后的预测结果
        """
        boxes = [det_pred['boxes'], seg_pred['boxes']]
        scores = [det_pred['scores'], seg_pred['scores']]
        labels = [det_pred['labels'], seg_pred['labels']]
        weights = [det_weight, seg_weight]
        
        return self.weighted_cluster_nms(boxes, scores, labels, weights) 