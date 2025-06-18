import argparse
import os
import yaml
from ultralytics import YOLO

# 禁用 wandb，避免交互式提示
os.environ["WANDB_DISABLED"] = "true"

def main():
    """
    YOLO 训练启动器。
    通过四个核心命令行参数来配置并启动训练任务。
    """
    parser = argparse.ArgumentParser(description='极简 YOLO 训练启动器')
    
    # --- 定义四个核心配置参数 ---
    
    # 1. 数据集配置文件
    parser.add_argument('--data', type=str, required=True, 
                        help='数据集配置文件路径')
                        
    # 2. 模型架构配置文件
    parser.add_argument('--model', type=str, required=True, 
                        help='模型架构配置文件路径')
                        
    # 3. 训练超参数配置文件
    parser.add_argument('--hyp', type=str, default='/content/cracks-S-D/ultralytics/cfg/default.yaml', 
                        help='训练超参数配置文件路径')
                        
    # 4. 预训练权重文件
    parser.add_argument('--weights', type=str, default='yolo11n.pt', 
                        help='预训练权重路径')

    args = parser.parse_args()

    # --- 执行逻辑 ---

    # 1. 加载训练超参数
    try:
        with open(args.hyp, 'r') as f:
            train_kwargs = yaml.safe_load(f)
        print(f"信息：已从 {args.hyp} 加载训练超参数。")
    except Exception as e:
        print(f"错误：无法加载超参数文件 {args.hyp}。原因: {e}")
        return

    # 2. 将核心配置添加到训练参数中
    train_kwargs['data'] = args.data
    
    # 3. 初始化模型架构
    model = YOLO(args.model)
    print(f"信息：已从 {args.model} 加载模型架构。")

    # 4. 设置预训练权重 (如果指定了)
    if args.weights.lower() != 'none':
        train_kwargs['pretrained'] = args.weights
        print(f"信息：将使用预训练权重: {args.weights}")
    else:
        print("信息：将从头开始训练模型。")

    # 5. 开始训练
    try:
        print("\n--- 开始训练 ---")
        # 使用 **kwargs 将所有参数一次性传递给 train 方法
        model.train(**train_kwargs)
        print("\n--- 训练完成 ---")
    except Exception as e:
        print(f"\n训练过程中出现严重错误: {e}")

if __name__ == '__main__':
    # 确保已安装 pyyaml
    try:
        import yaml
    except ImportError:
        print("错误：PyYAML 未安装。请运行 'pip install pyyaml'。")
    else:
        main()
