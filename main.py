from ultralytics import YOLO
import argparse
import os


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO11 训练脚本，支持自定义增强控制')
    
    # 训练模式
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'comparison'],
                        help='训练模式: single=单次训练, comparison=对比实验')
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default='ultralytics/cfg/models/11/yolo11-seg.yaml',
                        help='模型配置文件路径')
    parser.add_argument('--weights', type=str, default='yolo11n.pt',
                        help='预训练权重文件路径')
    
    # 数据相关参数
    parser.add_argument('--data', type=str, default='datasets/data.yaml',
                        help='数据配置文件路径')
    parser.add_argument('--epochs', type=int, default=300,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='图像尺寸')
    
    # 功能开关
    parser.add_argument('--custom-augment', action='store_true',
                        help='是否启用自定义增强算法')
    parser.add_argument('--cbam', action='store_true',
                        help='是否启用CBAM注意力机制')
    
    # 自定义增强参数
    parser.add_argument('--custom-augment-p', type=float, default=0.5,
                        help='自定义增强概率 (0.0-1.0)')
    parser.add_argument('--custom-augment-intensity', type=float, default=0.4,
                        help='自定义增强强度 (0.0-1.0)')
    parser.add_argument('--custom-augment-black-thresh', type=float, default=0.05,
                        help='黑色区域阈值 (0.0-1.0)')
    parser.add_argument('--custom-augment-white-thresh', type=float, default=0.1,
                        help='白色区域阈值 (0.0-1.0)')
    parser.add_argument('--custom-augment-sigma', type=int, default=5,
                        help='平滑系数')
    
    # 其他训练参数
    parser.add_argument('--device', type=str, default='',
                        help='训练设备 (cpu, 0, 1, 2, ...)')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载工作线程数')
    parser.add_argument('--project', type=str, default='runs/segment',
                        help='项目保存目录')
    parser.add_argument('--name', type=str, default='exp',
                        help='实验名称')
    
    return parser.parse_args()


def create_train_config(args):
    """创建训练配置字典"""
    config = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.imgsz,
        'device': args.device if args.device else None,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'save': True,
        'plots': True,
        'val': True,
        
        # 自定义增强参数
        'custom_augment': 1 if args.custom_augment else 0,
        'custom_augment_p': args.custom_augment_p,
        'custom_augment_intensity': args.custom_augment_intensity,
        'custom_augment_black_thresh': args.custom_augment_black_thresh,
        'custom_augment_white_thresh': args.custom_augment_white_thresh,
        'custom_augment_sigma': args.custom_augment_sigma,
        
        # 标准数据增强参数
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.1,
        'copy_paste': 0.1,
    }
    
    return config


def get_model_path(args):
    """根据参数选择模型配置文件"""
    if args.cbam:
        return 'ultralytics/cfg/models/11/yolo11-cbam.yaml'
    return 'ultralytics/cfg/models/11/yolo11.yaml'


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 根据参数选择模型配置
    model_path = get_model_path(args)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型配置文件 {model_path} 不存在")
        return
    
    # 检查数据配置文件是否存在
    if not os.path.exists(args.data):
        print(f"错误：数据配置文件 {args.data} 不存在")
        return
    
    # 创建YOLO模型
    print(f"加载模型配置：{model_path}")
    model = YOLO(model_path)
    
    # 加载预训练权重
    if args.weights and os.path.exists(args.weights):
        print(f"加载预训练权重：{args.weights}")
        model.load(args.weights)
    else:
        print("警告：未找到预训练权重文件，将从头开始训练")
    
    # 创建训练配置
    train_config = create_train_config(args)
    
    # 打印配置信息
    print("\n" + "="*50)
    print("训练配置信息:")
    print("="*50)
    print(f"训练模式: {args.mode}")
    print(f"模型配置: {model_path}")
    print(f"数据配置: {args.data}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"图像尺寸: {args.imgsz}")
    print(f"自定义增强: {'启用' if args.custom_augment else '禁用'}")
    print(f"CBAM注意力: {'启用' if args.cbam else '禁用'}")
    if args.custom_augment:
        print(f"  - 增强概率: {args.custom_augment_p}")
        print(f"  - 增强强度: {args.custom_augment_intensity}")
        print(f"  - 黑色阈值: {args.custom_augment_black_thresh}")
        print(f"  - 白色阈值: {args.custom_augment_white_thresh}")
        print(f"  - 平滑系数: {args.custom_augment_sigma}")
    print(f"设备: {args.device if args.device else '自动选择'}")
    print("="*50)
    
    # 开始训练
    try:
        print("\n开始训练...")
        results = model.train(**train_config)
        print("\n训练完成！")
        print(f"结果保存在: {results}")
        
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        return


if __name__ == '__main__':
    main()