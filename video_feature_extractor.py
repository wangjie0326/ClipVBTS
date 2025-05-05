import os
import torch
import numpy as np
from transformers import AutoProcessor, AutoModel
from PIL import Image
import cv2
from typing import List, Union, Tuple, Optional
import matplotlib.pyplot as plt

class VideoFeatureExtractor:
    """
    基于VideoCLIP-XL的视频特征提取器
    用于DIGIT视觉触觉传感器的视频数据分类
    """
    
    def __init__(
        self, 
        model_name: str = "alibaba-pai/VideoCLIP-XL", 
        device: str = None,
        frame_sample_rate: int = 4,
        offline_mode: bool = False
    ):
        """
        初始化视频特征提取器
        
        参数:
            model_name: 模型名称或路径
            device: 运行设备，如果为None则自动选择
            frame_sample_rate: 视频帧采样率
            offline_mode: 是否使用离线模式(不下载预训练模型)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.offline_mode = offline_mode
        self.frame_sample_rate = frame_sample_rate
        
        if not offline_mode:
            try:
                print(f"Loading model {model_name} on {self.device}...")
                # 设置代理为None，避免系统代理问题
                import os
                original_http_proxy = os.environ.get('HTTP_PROXY')
                original_https_proxy = os.environ.get('HTTPS_PROXY')
                
                try:
                    os.environ['HTTP_PROXY'] = ''
                    os.environ['HTTPS_PROXY'] = ''
                    
                    self.processor = AutoProcessor.from_pretrained(model_name)
                    self.model = AutoModel.from_pretrained(model_name).to(self.device)
                    self.model.eval()
                    print("Model loaded successfully!")
                finally:
                    # 恢复原始代理设置
                    if original_http_proxy:
                        os.environ['HTTP_PROXY'] = original_http_proxy
                    if original_https_proxy:
                        os.environ['HTTPS_PROXY'] = original_https_proxy
            except Exception as e:
                print(f"⚠️ 无法加载模型: {e}")
                print("⚠️ 切换至离线模式，仅提供视频帧提取功能")
                self.offline_mode = True
        
        if self.offline_mode:
            print("⚠️ 离线模式启用：仅提供视频帧提取功能，不进行特征提取")
    
    def extract_frames(
        self, 
        video_path: str, 
        max_frames: int = 8,
        resize_shape: Tuple[int, int] = (224, 224)
    ) -> List[Image.Image]:
        """
        从视频中提取帧
        
        参数:
            video_path: 视频文件路径
            max_frames: 最大帧数
            resize_shape: 调整大小的形状
            
        返回:
            frames: 帧列表
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 计算采样帧的索引
        if total_frames <= max_frames:
            # 如果视频帧数少于等于max_frames，则全部使用
            indices = list(range(0, total_frames, max(1, self.frame_sample_rate)))
        else:
            # 否则均匀采样max_frames帧
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # 将BGR转换为RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 调整大小
            frame = cv2.resize(frame, resize_shape)
            
            # 转换为PIL图像
            frame = Image.fromarray(frame)
            frames.append(frame)
        
        cap.release()
        
        print(f"提取了 {len(frames)} 帧，视频总帧数: {total_frames}")
        return frames

    def extract_features(
        self, 
        frames: List[Image.Image],
        return_dict: bool = False
    ) -> Union[torch.Tensor, dict]:
        """
        从视频帧中提取特征
        
        参数:
            frames: 视频帧列表
            return_dict: 是否返回字典格式的结果
            
        返回:
            features: 视频特征向量或包含所有输出的字典
        """
        if self.offline_mode:
            # 离线模式下，返回基于图像处理的简单特征
            print("⚠️ 离线模式：使用基本图像特征代替模型特征")
            
            # 转换图像为灰度并提取基本统计特征
            features_list = []
            for frame in frames:
                # 转为numpy数组
                frame_np = np.array(frame)
                
                # 转为灰度
                if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                    frame_gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
                else:
                    frame_gray = frame_np
                
                # 提取基本特征：平均值、标准差、最小值、最大值、中值
                mean = np.mean(frame_gray)
                std = np.std(frame_gray)
                min_val = np.min(frame_gray)
                max_val = np.max(frame_gray)
                median = np.median(frame_gray)
                
                # 计算简单的纹理特征（基于灰度共生矩阵）
                try:
                    from skimage.feature import graycomatrix, graycoprops
                    glcm = graycomatrix(frame_gray, [1], [0], 256, symmetric=True, normed=True)
                    contrast = graycoprops(glcm, 'contrast')[0, 0]
                    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
                    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                    energy = graycoprops(glcm, 'energy')[0, 0]
                    correlation = graycoprops(glcm, 'correlation')[0, 0]
                except ImportError:
                    # 如果没有scikit-image，使用简单替代
                    contrast = np.std(frame_gray) / np.mean(frame_gray) if np.mean(frame_gray) > 0 else 0
                    dissimilarity = contrast * 0.8
                    homogeneity = 1.0 / (1.0 + contrast)
                    energy = 1.0 / (1.0 + np.var(frame_gray)) if np.var(frame_gray) > 0 else 1.0
                    correlation = 0.5  # 默认值
                
                # 组合所有特征
                frame_features = np.array([
                    mean, std, min_val, max_val, median,
                    contrast, dissimilarity, homogeneity, energy, correlation
                ])
                
                features_list.append(frame_features)
            
            # 将所有帧的特征合并
            combined_features = np.vstack(features_list)
            mean_features = np.mean(combined_features, axis=0)
            
            # 扩展特征维度，用于替代模型输出
            expanded_features = np.tile(mean_features, 77)[:768]  # 扩展到模型输出的维度
            features_tensor = torch.FloatTensor(expanded_features).unsqueeze(0)
            
            if return_dict:
                # 创建一个类似模型输出的字典
                class SimpleOutput:
                    def __init__(self, features):
                        self.last_hidden_state = features.unsqueeze(1)
                return SimpleOutput(features_tensor)
            
            return features_tensor
            
        # 正常模式：使用预训练模型提取特征
        # 预处理帧
        inputs = self.processor(images=frames, return_tensors="pt").to(self.device)
        
        # 提取特征
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        if return_dict:
            return outputs
        
        # 获取视频特征表示（使用[CLS]标记的最后一层隐藏状态）
        video_features = outputs.last_hidden_state[:, 0, :]
        
        # 返回到CPU并转为numpy数组
        return video_features.cpu()
    
    def process_video(
        self, 
        video_path: str,
        max_frames: int = 8,
        return_dict: bool = False
    ) -> Union[torch.Tensor, dict]:
        """
        处理完整视频并提取特征
        
        参数:
            video_path: 视频文件路径
            max_frames: 最大帧数
            return_dict: 是否返回字典格式的结果
            
        返回:
            features: 视频特征向量或包含所有输出的字典
        """
        frames = self.extract_frames(video_path, max_frames)
        return self.extract_features(frames, return_dict)
    
    def process_batch_videos(
        self, 
        video_paths: List[str],
        max_frames: int = 8
    ) -> torch.Tensor:
        """
        批量处理视频并提取特征
        
        参数:
            video_paths: 视频文件路径列表
            max_frames: 每个视频的最大帧数
            
        返回:
            features: 批量视频特征向量
        """
        all_features = []
        
        for video_path in video_paths:
            features = self.process_video(video_path, max_frames)
            all_features.append(features)
            
        return torch.cat(all_features, dim=0)
    
    def visualize_frames(
        self, 
        video_path: str,
        max_frames: int = 8,
        figsize: Tuple[int, int] = (15, 3)
    ):
        """
        可视化视频帧
        
        参数:
            video_path: 视频文件路径
            max_frames: 最大帧数
            figsize: 图形大小
        """
        frames = self.extract_frames(video_path, max_frames)
        
        fig, axes = plt.subplots(1, len(frames), figsize=figsize)
        if len(frames) == 1:
            axes = [axes]
            
        for i, frame in enumerate(frames):
            axes[i].imshow(frame)
            axes[i].set_title(f"Frame {i}")
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()
        
    def similarity(self, feature1: torch.Tensor, feature2: torch.Tensor) -> float:
        """
        计算两个特征向量的余弦相似度
        
        参数:
            feature1: 第一个特征向量
            feature2: 第二个特征向量
            
        返回:
            similarity: 余弦相似度
        """
        return torch.nn.functional.cosine_similarity(feature1, feature2, dim=1).item()


# 使用示例
if __name__ == "__main__":
    # 初始化特征提取器
    extractor = VideoFeatureExtractor()
    
    # 处理单个视频
    video_path = "path/to/your/video.mp4"
    features = extractor.process_video(video_path)
    print(f"视频特征维度: {features.shape}")
    
    # 可视化视频帧
    extractor.visualize_frames(video_path)
    
    # 批量处理视频
    video_paths = ["path/to/video1.mp4", "path/to/video2.mp4"]
    batch_features = extractor.process_batch_videos(video_paths)
    print(f"批量视频特征维度: {batch_features.shape}")