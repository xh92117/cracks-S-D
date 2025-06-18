# custom_augment.py
import cv2
import numpy as np
import random
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from ultralytics.utils import LOGGER

class CustomAugment:
    """
    配置驱动的自定义裂缝增强类。
    所有参数从YOLO配置系统中读取，包括锐化参数。
    """
    
    def __init__(self, **kwargs):
        """
        从配置参数初始化增强器。
        """
        # 从配置中读取参数，提供合理的默认值
        self.enabled = kwargs.get('custom_augment', True)
        self.p = kwargs.get('custom_augment_p', 0.5)
        self.tanh_strength = kwargs.get('custom_augment_tanh_strength', 1.0)
        self.blend_alpha = kwargs.get('custom_augment_blend_alpha', 0.5)
        self.clahe_limit = kwargs.get('custom_augment_clahe_limit', 3.0)
        self.preserve_color = kwargs.get('custom_augment_preserve_color', True)
        # 【优化】将锐化参数也设为可配置
        self.sharpen_alpha = kwargs.get('custom_augment_sharpen_alpha', 1.5)
        self.sharpen_beta = kwargs.get('custom_augment_sharpen_beta', -0.5)
        
        if self.enabled:
            LOGGER.info(f"CustomAugment enabled: p={self.p}, preserve_color={self.preserve_color}, "
                       f"tanh_strength={self.tanh_strength}, blend_alpha={self.blend_alpha}, "
                       f"sharpen_alpha={self.sharpen_alpha}")
        else:
            LOGGER.info("CustomAugment disabled")
        
    def __call__(self, labels):
        """
        根据配置参数对图像应用Tanh增强算法。
        """
        if not self.enabled or random.random() > self.p:
            return labels
        
        img = labels.get('img')
        
        if img is None or img.size == 0 or len(img.shape) != 3 or img.shape[2] != 3:
            LOGGER.warning(f"Invalid image format: {img.shape if img is not None else 'None'}. Skipping.")
            return labels
        
        try:
            if self.preserve_color:
                enhanced_img = self._apply_color_preserving_enhancement(img)
            else:
                enhanced_img = self._apply_grayscale_enhancement(img)
            
            if enhanced_img is not None and enhanced_img.size > 0:
                labels['img'] = enhanced_img.astype(np.uint8)
            else:
                LOGGER.warning("Enhancement produced an invalid image, returning original.")
            
        except Exception as e:
            import traceback
            LOGGER.warning(f"CustomAugment failed: {e}. Returning original image.")
            LOGGER.debug(f"Traceback: {traceback.format_exc()}")
            
        return labels

    def _apply_color_preserving_enhancement(self, img):
        """
        在LAB色彩空间中处理，保持颜色信息。
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        enhanced_l = self._apply_tanh_enhancement(l_channel)
        enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    def _apply_grayscale_enhancement(self, img):
        """
        转换为灰度处理后转回BGR格式。
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        enhanced_gray = self._apply_tanh_enhancement(gray)
        return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    def _apply_tanh_enhancement(self, channel):
        """
        核心Tanh增强算法。
        """
        # 1. Unsharp Masking 锐化（使用可配置参数）
        blurred = cv2.GaussianBlur(channel, (0, 0), 3)
        sharpened = cv2.addWeighted(channel, self.sharpen_alpha, blurred, self.sharpen_beta, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        # 2. 直方图分析与自适应强度计算
        try:
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel()
            smoothed_hist = gaussian_filter(hist, sigma=2)
            peaks, _ = find_peaks(smoothed_hist, prominence=np.max(smoothed_hist) * 0.1)
            main_peak = peaks[np.argmax(smoothed_hist[peaks])] if len(peaks) > 0 else 128
            mean_val = np.mean(channel)
            hist_skew = (mean_val - main_peak) / 255
            dynamic_intensity = 0.3 + 0.5 * abs(hist_skew)
        except Exception:
            main_peak = 128
            dynamic_intensity = 0.5

        # 3. Tanh非线性映射
        x = np.linspace(0, 255, 256)
        tanh_map = 255 * (np.tanh((x - main_peak) / 128) + 1) / 2 
        final_intensity = np.clip(dynamic_intensity * self.tanh_strength, 0, 1)
        final_map = np.clip(tanh_map * final_intensity + x * (1 - final_intensity), 0, 255)

        # 4. LUT应用
        enhanced_tanh = cv2.LUT(sharpened, final_map.astype(np.uint8))
        
        # 5. 与原图融合
        blended = cv2.addWeighted(enhanced_tanh, self.blend_alpha, channel, 1 - self.blend_alpha, 0)

        # 6. CLAHE局部对比度增强
        clahe = cv2.createCLAHE(clipLimit=self.clahe_limit, tileGridSize=(8, 8))
        return clahe.apply(blended.astype(np.uint8))