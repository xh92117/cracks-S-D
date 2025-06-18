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
    parser.add_argument('--model', type=str, default='auto',
                        help='模型配置文件路径 (auto: 自动选择)')
    
    # 数据相关参数
    parser.add_argument('--data', type=str, default='/content/crack-detect-1/data.yaml',
                        help='数据配置文件路径')
    parser.add_argument('--epochs', type=int, default=300,
                        help='训练轮数')
    parser.add_argument('--patience', type=int, default=50,
                        help='早停耐心值')
    parser.add_argument('--batch', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='图像尺寸')
    
    # 功能开关
    parser.add_argument('--custom-augment', action='store_true',
                        help='是否启用自定义增强算法')
    parser.add_argument('--cbam', action='store_true',
                        help='是否启用CBAM注意力机制')
    parser.add_argument('--optimized', action='store_true',
                        help='是否使用优化配置 (降低增强强度和CBAM复杂度)')
    
    # 自定义增强参数 (Tanh增强配置)
    parser.add_argument('--custom-augment-p', type=float, default=0.5,
                        help='自定义增强概率 (0.0-1.0)')
    parser.add_argument('--custom-augment-tanh-strength', type=float, default=1.0,
                        help='Tanh映射强度 (<1=减弱, =1=正常, >1=增强)')
    parser.add_argument('--custom-augment-blend-alpha', type=float, default=0.5,
                        help='融合权重 (0.0=原图, 1.0=完全增强)')
    parser.add_argument('--custom-augment-clahe-limit', type=float, default=3.0,
                        help='CLAHE对比度限制 (1.0-8.0)')
    parser.add_argument('--custom-augment-preserve-color', action='store_true', default=True,
                        help='是否保持颜色信息 (推荐启用)')
    parser.add_argument('--custom-augment-sharpen-alpha', type=float, default=1.5,
                        help='锐化正权重参数')
    parser.add_argument('--custom-augment-sharpen-beta', type=float, default=-0.5,
                        help='锐化负权重参数')
    
    # 其他训练参数
    parser.add_argument('--device', type=str, default='',
                        help='训练设备 (cpu, 0, 1, 2, ...)')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载工作线程数')
    parser.add_argument('--project', type=str, default='runs',
                        help='项目保存目录')
    parser.add_argument('--name', type=str, default='exp',
                        help='实验名称')
    
    return parser.parse_args()


def create_train_config(args):
    """创建训练配置字典"""
    config = {
        'data': args.data,
        'epochs': args.epochs,
        'patience': args.patience,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device if args.device else None,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'save': True,
        'plots': True,
        'val': True,
        
        # 自定义增强参数 (新版配置驱动)
        'custom_augment': args.custom_augment,
        'custom_augment_p': args.custom_augment_p,
        'custom_augment_tanh_strength': getattr(args, 'custom_augment_tanh_strength', 1.0),
        'custom_augment_blend_alpha': getattr(args, 'custom_augment_blend_alpha', 0.5),
        'custom_augment_clahe_limit': getattr(args, 'custom_augment_clahe_limit', 3.0),
        'custom_augment_preserve_color': getattr(args, 'custom_augment_preserve_color', True),
        'custom_augment_sharpen_alpha': getattr(args, 'custom_augment_sharpen_alpha', 1.5),
        'custom_augment_sharpen_beta': getattr(args, 'custom_augment_sharpen_beta', -0.5),
        
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


def get_model_config(args):
    """根据参数选择模型配置文件"""
    if args.cbam:
        if hasattr(args, 'optimized') and args.optimized:
            return 'ultralytics/cfg/models/11/yolo11-cbam-optimized.yaml'
        else:
            return 'ultralytics/cfg/models/11/yolo11-cbam.yaml'
    else:
        return 'ultralytics/cfg/models/11/yolo11n.yaml'


def get_training_config(args):
    """根据参数选择训练配置文件"""
    if hasattr(args, 'optimized') and args.optimized:
        return 'ultralytics/cfg/crack-detection-optimized.yaml'
    else:
        return None  # 使用默认配置


def setup_training_config(args):
    """设置训练配置"""
    config = {}
    
    # 基础配置
    config.update({
        'data': args.data,
        'epochs': args.epochs,
        'patience': args.patience,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name
    })
    
    # 如果使用优化配置
    training_config_file = get_training_config(args)
    if training_config_file:
        import yaml
        try:
            with open(training_config_file, 'r', encoding='utf-8') as f:
                optimized_config = yaml.safe_load(f)
            
            # 合并优化配置，但保持命令行参数优先级
            for key, value in optimized_config.items():
                if key not in config and key not in ['task', 'mode', 'model', 'data']:
                    config[key] = value
        except Exception as e:
            print(f"警告: 无法加载优化配置文件 {training_config_file}: {e}")
    
    # 自定义增强配置 (新版Tanh增强)
    if args.custom_augment:
        custom_config = {
            'custom_augment': True,
            'custom_augment_p': getattr(args, 'custom_augment_p', 0.5),
            'custom_augment_tanh_strength': getattr(args, 'custom_augment_tanh_strength', 1.0),
            'custom_augment_blend_alpha': getattr(args, 'custom_augment_blend_alpha', 0.5),
            'custom_augment_clahe_limit': getattr(args, 'custom_augment_clahe_limit', 3.0),
            'custom_augment_preserve_color': getattr(args, 'custom_augment_preserve_color', True),
            'custom_augment_sharpen_alpha': getattr(args, 'custom_augment_sharpen_alpha', 1.5),
            'custom_augment_sharpen_beta': getattr(args, 'custom_augment_sharpen_beta', -0.5),
        }
        # 如果是优化模式，调整参数
        if hasattr(args, 'optimized') and args.optimized:
            custom_config.update({
                'custom_augment_p': 0.3,
                'custom_augment_tanh_strength': 0.7,
                'custom_augment_blend_alpha': 0.3,
                'custom_augment_clahe_limit': 2.0,
            })
            print("✅ 启用自定义Tanh增强算法 (优化模式)")
        else:
            print("✅ 启用自定义Tanh增强算法 (标准模式)")
        config.update(custom_config)
    else:
        config['custom_augment'] = False
        print("❌ 禁用自定义增强算法")
    
    return config


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 自动选择模型配置
    if args.model == 'auto':
        args.model = get_model_config(args)
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"错误：模型配置文件 {args.model} 不存在")
        return
    
    # 检查数据配置文件是否存在
    if not os.path.exists(args.data):
        print(f"错误：数据配置文件 {args.data} 不存在")
        return
    
    # 创建YOLO模型
    print(f"加载模型配置：{args.model}")
    model = YOLO(args.model)
    
    # 加载预训练权重
    if hasattr(args, 'weights') and args.weights and os.path.exists(args.weights):
        print(f"加载预训练权重：{args.weights}")
        model.load(args.weights)
    else:
        print("警告：未找到预训练权重文件，将从头开始训练")
    
    # 创建训练配置
    train_config = setup_training_config(args)
    
    # 打印配置信息
    print("\n" + "="*50)
    print("训练配置信息:")
    print("="*50)
    print(f"训练模式: {args.mode}")
    print(f"模型配置: {args.model}")
    print(f"数据配置: {args.data}")
    print(f"自定义增强: {'启用' if args.custom_augment else '禁用'}")
    print(f"CBAM注意力: {'启用' if args.cbam else '禁用'}")
    print(f"优化模式: {'启用' if hasattr(args, 'optimized') and args.optimized else '禁用'}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch}")
    print(f"图像尺寸: {args.imgsz}")
    if args.custom_augment:
        print(f"  - 增强概率: {getattr(args, 'custom_augment_p', 0.5)}")
        print(f"  - Tanh强度: {getattr(args, 'custom_augment_tanh_strength', 1.0)}")
        print(f"  - 融合权重: {getattr(args, 'custom_augment_blend_alpha', 0.5)}")
        print(f"  - CLAHE限制: {getattr(args, 'custom_augment_clahe_limit', 3.0)}")
        print(f"  - 保持颜色: {getattr(args, 'custom_augment_preserve_color', True)}")
        print(f"  - 锐化参数: {getattr(args, 'custom_augment_sharpen_alpha', 1.5)}, {getattr(args, 'custom_augment_sharpen_beta', -0.5)}")
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