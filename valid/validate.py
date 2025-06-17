import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from wbf_fusion import WCNMSFusion
import argparse
from pathlib import Path

def load_models(det_model_path: str, seg_model_path: str):
    """
    加载检测和分割模型
    Args:
        det_model_path: 检测模型路径
        seg_model_path: 分割模型路径
    Returns:
        检测模型和分割模型
    """
    det_model = YOLO(det_model_path)
    seg_model = YOLO(seg_model_path)
    return det_model, seg_model

def process_image(image_path: str, det_model, seg_model, wcnms_fusion, 
                 det_weight: float = 0.6, seg_weight: float = 0.4,
                 conf_threshold: float = 0.25):
    """
    处理单张图片
    Args:
        image_path: 图片路径
        det_model: 检测模型
        seg_model: 分割模型
        wcnms_fusion: WCNMS融合器
        det_weight: 检测模型权重
        seg_weight: 分割模型权重
        conf_threshold: 置信度阈值
    Returns:
        融合后的预测结果
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")

    # 检测模型预测
    det_results = det_model(image, conf=conf_threshold)[0]
    det_pred = {
        'boxes': det_results.boxes.xyxy.cpu().numpy(),
        'scores': det_results.boxes.conf.cpu().numpy(),
        'labels': det_results.boxes.cls.cpu().numpy()
    }

    # 分割模型预测
    seg_results = seg_model(image, conf=conf_threshold)[0]
    seg_pred = {
        'boxes': seg_results.boxes.xyxy.cpu().numpy(),
        'scores': seg_results.boxes.conf.cpu().numpy(),
        'labels': seg_results.boxes.cls.cpu().numpy()
    }

    # 融合预测结果
    fused_pred = wcnms_fusion.fuse_predictions(
        det_pred, seg_pred,
        det_weight=det_weight,
        seg_weight=seg_weight
    )

    return fused_pred, image

def visualize_results(image, fused_pred, output_path: str):
    """
    可视化预测结果
    Args:
        image: 原始图片
        fused_pred: 融合后的预测结果
        output_path: 输出图片路径
    """
    # 复制图片用于绘制
    vis_image = image.copy()
    
    # 绘制预测框
    for box, score, label in zip(fused_pred['boxes'], fused_pred['scores'], fused_pred['labels']):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 添加标签和置信度
        label_text = f"Class {int(label)}: {score:.2f}"
        cv2.putText(vis_image, label_text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存结果
    cv2.imwrite(output_path, vis_image)

def main():
    parser = argparse.ArgumentParser(description='验证WCNMS融合算法')
    parser.add_argument('--det-model', type=str, required=True, help='检测模型路径')
    parser.add_argument('--seg-model', type=str, required=True, help='分割模型路径')
    parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    parser.add_argument('--det-weight', type=float, default=0.6, help='检测模型权重')
    parser.add_argument('--seg-weight', type=float, default=0.4, help='分割模型权重')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IOU阈值')
    
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 加载模型
    det_model, seg_model = load_models(args.det_model, args.seg_model)

    # 初始化WCNMS融合器
    wcnms_fusion = WCNMSFusion(iou_threshold=args.iou_threshold)

    # 处理图片
    fused_pred, image = process_image(
        args.image, det_model, seg_model, wcnms_fusion,
        det_weight=args.det_weight,
        seg_weight=args.seg_weight,
        conf_threshold=args.conf_threshold
    )

    # 可视化结果
    output_path = os.path.join(args.output, f"result_{Path(args.image).stem}.jpg")
    visualize_results(image, fused_pred, output_path)

    print(f"处理完成，结果保存在: {output_path}")

if __name__ == '__main__':
    main() 