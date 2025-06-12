# custom_augment.py
import cv2
import numpy as np
import random
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from ultralytics.utils import LOGGER  # 集成YOLO日志系统


class CustomAugment:
    """自定义数据增强类，专门用于裂缝检测"""
    
    def __init__(self, p=0.5, black_thresh=0.05, white_thresh=0.1, intensity=0.4, sigma=5):
        """
        初始化自定义增强器
        
        Args:
            p (float): 应用增强的概率
            black_thresh (float): 黑色区域阈值
            white_thresh (float): 白色区域阈值
            intensity (float): 增强强度
            sigma (float): 平滑系数
        """
        self.p = max(0.0, min(1.0, p))  # 确保概率在[0,1]范围内
        self.black_thresh = max(0.0, min(1.0, black_thresh))
        self.white_thresh = max(0.0, min(1.0, white_thresh))
        self.intensity = max(0.0, min(1.0, intensity))
        self.sigma = max(1, sigma)
        
    def enhance_contrast(self, img):
        """增强对比度"""
        try:
            if img is None or img.size == 0:
                return img
                
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        except Exception as e:
            LOGGER.warning(f"Enhanced contrast failed: {e}")
            return img
    
    def enhance_edges(self, img):
        """增强边缘特征"""
        try:
            if img is None or img.size == 0:
                return img
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return cv2.addWeighted(img, 1-self.intensity, edges, self.intensity, 0)
        except Exception as e:
            LOGGER.warning(f"Enhanced edges failed: {e}")
            return img
    
    def add_noise(self, img):
        """添加随机噪声"""
        try:
            if img is None or img.size == 0:
                return img
                
            noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
            return cv2.add(img, noise)
        except Exception as e:
            LOGGER.warning(f"Add noise failed: {e}")
            return img
    
    def adjust_brightness(self, img):
        """调整亮度"""
        try:
            if img is None or img.size == 0:
                return img
                
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            brightness_factor = 1 + random.uniform(-0.2, 0.2)
            hsv[:,:,2] = np.clip(hsv[:,:,2] * brightness_factor, 0, 255)
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        except Exception as e:
            LOGGER.warning(f"Adjust brightness failed: {e}")
            return img

    def _tanh_hist_equalization(self, img):
        """高级直方图均衡化算法"""
        try:
            if img is None or img.size == 0:
                return img
                
            # 转换为灰度处理
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

            # 动态Canny阈值
            otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            canny_low = max(0, int(otsu_thresh * 0.4))
            canny_high = min(255, int(otsu_thresh * 1.6))
            edges = cv2.Canny(gray, canny_low, canny_high)

            # 自适应形态学核
            min_dim = min(gray.shape)
            kernel_size = max(3, int(min_dim / 100))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            edge_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

            # 基础增强
            enhanced = cv2.addWeighted(gray, 1.5, cv2.GaussianBlur(gray, (0, 0), 3), -0.5, 0)
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)

            # 直方图分析
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
            smoothed_hist = gaussian_filter(hist, sigma=2)
            peaks, _ = find_peaks(smoothed_hist, prominence=np.max(smoothed_hist) * 0.1)

            # 动态强度计算
            mean_val = np.mean(gray)
            main_peak = peaks[np.argmax(smoothed_hist[peaks])] if len(peaks) > 0 else 128
            hist_skew = (mean_val - main_peak) / 255
            dynamic_intensity = 0.3 + 0.5 * abs(hist_skew)

            # Tanh映射
            x = np.linspace(0, 255, 256)
            mapped = 255 * (np.tanh((x - main_peak) / 128) + 1) / 2
            mapped = np.clip(mapped * dynamic_intensity + x * (1 - dynamic_intensity), 0, 255)

            # 应用LUT
            enhanced = cv2.LUT(enhanced, mapped.astype(np.uint8))

            # 边缘融合
            edge_strength = cv2.Sobel(enhanced, cv2.CV_64F, 1, 1, ksize=3)
            edge_weight = np.clip(cv2.normalize(edge_strength, None, 0, 1, cv2.NORM_MINMAX), 0, 1)
            final = cv2.addWeighted(gray, 0.3, enhanced, 0.7, 0)
            final = (final * (1 - edge_weight) + enhanced * edge_weight).astype(np.uint8)

            # CLAHE增强
            clahe = cv2.createCLAHE(clipLimit=3.0 + 2 * hist_skew, tileGridSize=(8, 8))
            final = clahe.apply(final)

            # 转换回彩色图像
            if len(img.shape) == 3:
                final = cv2.cvtColor(final, cv2.COLOR_GRAY2BGR)
                
            return final
        except Exception as e:
            LOGGER.warning(f"Tanh histogram equalization failed: {e}")
            return img
    
    def __call__(self, labels):
        """
        随机应用一种增强方法
        
        Args:
            labels (dict): YOLO数据字典，包含'img'和'instances'等键
            
        Returns:
            dict: 增强后的数据字典
        """
        if random.random() > self.p:
            return labels
        
        # 检查必要的键是否存在
        if 'img' not in labels or labels['img'] is None:
            return labels
            
        img = labels['img']
        if img.size == 0:
            return labels
            
        # 随机选择一种增强方法，包括新的高级方法
        methods = [
            self.enhance_contrast,
            self.enhance_edges,
            self.add_noise,
            self.adjust_brightness,
            self._tanh_hist_equalization  # 添加高级增强方法
        ]
        
        try:
            # 应用选中的增强方法
            enhanced = random.choice(methods)(img.copy())
            labels['img'] = enhanced.astype(np.uint8)
        except Exception as e:
            LOGGER.warning(f"Custom augmentation failed: {e}")
            # 如果增强失败，返回原图像
            return labels
            
        return labels
