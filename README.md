# YOLO11裂缝检测增强版

这个项目为YOLO11添加了专门针对裂缝检测优化的自定义图像增强算法和CBAM注意力机制，提供了完整的对比实验框架。

## 🚀 主要特性

- ✅ **自定义增强算法**: 专门针对裂缝检测优化的图像增强
- ✅ **CBAM注意力机制**: 集成卷积块注意力模块提升特征表示能力
- ✅ **对比实验框架**: 支持多种配置的自动化对比实验
- ✅ **完整集成**: 无缝集成到YOLO11训练流程中

## 📁 项目结构

```
yolo11cracks2.0/
├── ultralytics/
│   ├── data/
│   │   ├── custom_augment.py          # 自定义增强算法实现
│   │   └── augment.py                 # 修改后的YOLO增强模块
│   ├── nn/modules/
│   │   └── conv.py                    # CBAM注意力机制实现
│   └── cfg/models/11/
│       ├── yolo11.yaml                # 基础YOLO11配置
│       └── yolo11-cbam.yaml           # 带CBAM的YOLO11配置
├── train.py                           # 原始训练脚本
├── main.py                            # 新增主训练脚本(支持对比实验)
└── README.md                          # 项目说明文档
```

## 🔧 核心功能

### 1. 自定义增强算法 (Custom Augment)

专门为裂缝检测设计的图像增强算法，包含以下功能：
- **对比度增强**: 使用CLAHE算法增强局部对比度
- **边缘增强**: 通过Canny边缘检测强化裂缝特征
- **噪声添加**: 添加适量噪声提高模型鲁棒性
- **亮度调整**: 动态调整图像亮度适应不同光照条件
- **Tanh直方图均衡化**: 核心算法，动态优化图像对比度

### 2. CBAM注意力机制

卷积块注意力模块(Convolutional Block Attention Module)：
- **通道注意力**: 学习特征通道的重要性权重
- **空间注意力**: 关注图像中的重要空间位置
- **特征增强**: 提升裂缝特征的表示能力
- **轻量级设计**: 最小化计算开销

### 3. 对比实验框架

支持以下实验配置的自动化对比：
- 基础YOLO11 vs YOLO11+CBAM
- 启用/禁用自定义增强算法
- 不同模型尺寸对比
- 自动生成实验报告

## 🚀 快速开始

### 环境要求

```bash
pip install ultralytics
pip install opencv-python
pip install scipy
pip install pyyaml
```

### 基础训练

```bash
# 使用原始训练脚本
python train.py

# 使用新的主训练脚本 - 单次训练
python main.py --mode single --custom-augment --cbam

# 使用新的主训练脚本 - 对比实验
python main.py --mode comparison --epochs 100
```

### 详细使用方法

#### 1. 单次训练

```bash
# 基础YOLO11训练
python main.py --mode single --model yolo11n --data dataset/data.yaml

# 启用自定义增强
python main.py --mode single --custom-augment --name exp_custom_aug

# 使用CBAM注意力机制
python main.py --mode single --cbam --name exp_cbam

# 同时启用自定义增强和CBAM
python main.py --mode single --custom-augment --cbam --name exp_full
```

#### 2. 对比实验

```bash
# 运行完整对比实验
python main.py --mode comparison --epochs 300 --batch 32

# 快速测试(较少轮数)
python main.py --mode comparison --epochs 50 --batch 16
```

对比实验将自动运行以下4个配置：
1. `baseline_no_custom_aug`: 基础YOLO11，无自定义增强
2. `baseline_with_custom_aug`: 基础YOLO11，启用自定义增强
3. `cbam_no_custom_aug`: YOLO11+CBAM，无自定义增强
4. `cbam_with_custom_aug`: YOLO11+CBAM，启用自定义增强

#### 3. 传统训练方式

```python
from ultralytics import YOLO

# 基础模型训练
model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')
model.train(
    data='dataset/data.yaml',
    epochs=300,
    batch=32,
    custom_augment=1,  # 启用自定义增强
    custom_augment_p=0.5,
    amp=False
)

# CBAM模型训练
model_cbam = YOLO('ultralytics/cfg/models/11/yolo11-cbam.yaml')
model_cbam.train(
    data='dataset/data.yaml',
    epochs=300,
    batch=32,
    custom_augment=1,
    amp=False
)
```

## ⚙️ 配置参数

### 自定义增强参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `custom_augment` | 0 | 启用自定义增强 (0=禁用, 1=启用) |
| `custom_augment_p` | 0.5 | 应用增强的概率 |
| `custom_augment_black_thresh` | 0.05 | 黑色区域阈值 |
| `custom_augment_white_thresh` | 0.1 | 白色区域阈值 |
| `custom_augment_intensity` | 0.4 | 增强强度 |
| `custom_augment_sigma` | 5 | 平滑系数 |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epochs` | 300 | 训练轮数 |
| `batch` | 32 | 批次大小 |
| `imgsz` | 640 | 图像尺寸 |
| `optimizer` | 'SGD' | 优化器 |
| `amp` | False | 混合精度训练 |

## 📊 实验结果

运行对比实验后，将生成 `experiment_summary.yaml` 文件，包含：
- 各实验的训练状态
- 模型保存路径
- 实验配置信息
- 成功/失败统计

## 🔍 技术细节

### CBAM注意力机制集成

CBAM模块被集成在YOLO11的backbone中的关键位置：
- P2特征层后 (256通道)
- P3特征层后 (512通道) 
- P4特征层后 (512通道)
- P5特征层后 (1024通道)

### 自定义增强算法核心

```python
# 核心增强流程
def __call__(self, labels):
    if random.random() > self.p:
        return labels
    
    # 随机选择增强方法
    methods = [
        self.enhance_contrast,    # 对比度增强
        self.enhance_edges,       # 边缘增强
        self.add_noise,          # 噪声添加
        self.adjust_brightness   # 亮度调整
    ]
    
    enhanced = random.choice(methods)(labels['img'])
    labels['img'] = enhanced.astype(np.uint8)
    return labels
```

## 🛠️ 故障排除

### 常见问题

1. **训练时出现NaN loss**
   - 解决方案: 设置 `amp=False` 禁用混合精度训练

2. **CBAM模块未找到**
   - 确认 `ultralytics/nn/modules/conv.py` 中包含CBAM实现

3. **自定义增强未生效**
   - 检查 `custom_augment=1` 参数是否正确设置
   - 确认 `ultralytics/data/augment.py` 中的集成代码

4. **内存不足**
   - 减小 `batch_size` 参数
   - 降低 `imgsz` 图像尺寸

## 📈 性能优化建议

1. **数据集优化**
   - 确保标注质量
   - 平衡正负样本比例
   - 适当的数据增强强度

2. **训练策略**
   - 使用预训练权重
   - 适当的学习率调度
   - 早停策略避免过拟合

3. **模型选择**
   - 小数据集推荐使用yolo11n
   - 大数据集可尝试yolo11s/m
   - 根据精度要求选择是否使用CBAM

## 📝 更新日志

### v2.0 (当前版本)
- ✅ 集成CBAM注意力机制
- ✅ 新增对比实验框架
- ✅ 优化自定义增强算法
- ✅ 完善文档和使用说明

### v1.0
- ✅ 基础自定义增强算法
- ✅ YOLO11集成
- ✅ 基础训练脚本

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目基于AGPL-3.0许可证开源。 