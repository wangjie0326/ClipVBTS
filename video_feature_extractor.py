import os
import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Union, Tuple, Optional
import matplotlib.pyplot as plt
import glob
import warnings
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoFeatureExtractor:
    """
    基于VideoCLIP-XL的视频特征提取器
    用于DIGIT视觉触觉传感器的视频数据分类
    支持离线模式和本地模型加载
    """
    
    def __init__(
        self, 
        model_name: str = "alibaba-pai/VideoCLIP-XL", 
        device: str = None,
        frame_sample_rate: int = 4,
        offline_mode: bool = False,
        local_model_path: str = None,
        feature_dim: int = 768
    ):
        """
        初始化视频特征提取器
        
        参数:
            model_name: 模型名称或路径
            device: 运行设备，如果为None则自动选择
            frame_sample_rate: 视频帧采样率
            offline_mode: 是否使用离线模式(不下载预训练模型)
            local_model_path: 本地模型路径，如果提供则从本地加载模型
            feature_dim: 特征维度，仅在离线模式下使用
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.offline_mode = offline_mode
        self.frame_sample_rate = frame_sample_rate
        self.feature_dim = feature_dim
        self.model_name = model_name
        self.model = None
        self.processor = None
        
        # 检查是否使用本地模型
        if local_model_path and os.path.exists(local_model_path):
            self._load_local_model(local_model_path)
        elif not offline_mode:
            self._load_online_model(model_name)
        
        if self.offline_mode:
            logger.info("⚠️ 离线模式启用：仅提供视频帧提取功能，将使用简单图像特征代替模型特征")
            # 尝试加载scikit-image以提供更好的特征
            try:
                from skimage.feature import graycomatrix, graycoprops
                logger.info("✓ 已加载scikit-image，将使用更高级的纹理特征")
                self.has_skimage = True
            except ImportError:
                logger.info("⚠️ 未找到scikit-image，将使用基本图像特征")
                self.has_skimage = False
    
    def _load_local_model(self, local_model_path):
        """从本地加载模型"""
        try:
            logger.info(f"尝试从本地路径加载模型: {local_model_path}")
            
            # 检查本地模型文件
            if os.path.isdir(local_model_path):
                # 检查是否有必要的模型文件
                config_files = glob.glob(os.path.join(local_model_path, "*config*.json"))
                model_files = glob.glob(os.path.join(local_model_path, "*.bin")) + \
                             glob.glob(os.path.join(local_model_path, "*.safetensors"))
                
                if not config_files:
                    logger.warning(f"⚠️ 在 {local_model_path} 中未找到配置文件")
                    self.offline_mode = True
                    return
                    
                if not model_files and not os.path.exists(os.path.join(local_model_path, "pytorch_model.bin.index.json")):
                    logger.warning(f"⚠️ 在 {local_model_path} 中未找到模型文件")
                    self.offline_mode = True
                    return
            
            # 如果看起来有效，尝试加载模型
            from transformers import AutoProcessor, AutoModel
            
            # 禁用transformers警告
            warnings.filterwarnings("ignore")
            
            # 加载处理器和模型
            self.processor = AutoProcessor.from_pretrained(local_model_path)
            self.model = AutoModel.from_pretrained(local_model_path).to(self.device)
            self.model.eval()
            
            logger.info("✓ 从本地路径成功加载模型!")
            self.offline_mode = False
        except Exception as e:
            logger.error(f"❌ 从本地加载模型失败: {e}")
            self.offline_mode = True
    
    def _load_online_model(self, model_name):
        """从在线库加载模型"""
        try:
            logger.info(f"尝试从在线库加载模型 {model_name} 到 {self.device}...")
            
            # 禁用系统代理
            original_http_proxy = os.environ.get('HTTP_PROXY')
            original_https_proxy = os.environ.get('HTTPS_PROXY')
            
            try:
                # 清除代理环境变量
                os.environ['HTTP_PROXY'] = ''
                os.environ['HTTPS_PROXY'] = ''
                os.environ['http_proxy'] = ''
                os.environ['https_proxy'] = ''
                
                # 导入必要的库
                from transformers import AutoProcessor, AutoModel
                
                # 禁用transformers警告
                warnings.filterwarnings("ignore")
                
                # 尝试从缓存加载
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(self.device)
                self.model.eval()
                
                logger.info("✓ 模型加载成功!")
                self.offline_mode = False
            finally:
                # 恢复原始代理设置
                if original_http_proxy:
                    os.environ['HTTP_PROXY'] = original_http_proxy
                if original_https_proxy:
                    os.environ['HTTPS_PROXY'] = original_https_proxy
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            logger.warning("切换至离线模式，仅提供视频帧提取功能")
            self.offline_mode = True
    
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
        
        logger.info(f"提取了 {len(frames)} 帧，视频总帧数: {total_frames}")
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
            logger.info("使用离线特征提取器...")
            return self._extract_offline_features(frames, return_dict)
        else:
            # 在线模式：使用预训练模型提取特征
            return self._extract_online_features(frames, return_dict)
    
    def _extract_offline_features(self, frames, return_dict):
        """使用离线方式提取特征"""
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
            
            # 提取颜色特征
            if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                # 计算每个通道的均值和标准差
                r_mean = np.mean(frame_np[:,:,0])
                g_mean = np.mean(frame_np[:,:,1])
                b_mean = np.mean(frame_np[:,:,2])
                r_std = np.std(frame_np[:,:,0])
                g_std = np.std(frame_np[:,:,1])
                b_std = np.std(frame_np[:,:,2])
                
                # 颜色饱和度和亮度
                r_ratio = r_mean / (r_mean + g_mean + b_mean + 1e-10)
                g_ratio = g_mean / (r_mean + g_mean + b_mean + 1e-10)
                b_ratio = b_mean / (r_mean + g_mean + b_mean + 1e-10)
            else:
                r_mean = g_mean = b_mean = mean
                r_std = g_std = b_std = std
                r_ratio = g_ratio = b_ratio = 0.33
            
            # 计算简单的纹理特征
            if self.has_skimage:
                try:
                    from skimage.feature import graycomatrix, graycoprops
                    # 缩小灰度级别以提高计算速度
                    levels = 32
                    frame_gray_scaled = (frame_gray * (levels / 255)).astype(np.uint8)
                    glcm = graycomatrix(frame_gray_scaled, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels, symmetric=True, normed=True)
                    contrast = np.mean(graycoprops(glcm, 'contrast'))
                    dissimilarity = np.mean(graycoprops(glcm, 'dissimilarity'))
                    homogeneity = np.mean(graycoprops(glcm, 'homogeneity'))
                    energy = np.mean(graycoprops(glcm, 'energy'))
                    correlation = np.mean(graycoprops(glcm, 'correlation'))
                except Exception:
                    # 如果出错，回退到简单特征
                    contrast = np.std(frame_gray) / np.mean(frame_gray) if np.mean(frame_gray) > 0 else 0
                    dissimilarity = contrast * 0.8
                    homogeneity = 1.0 / (1.0 + contrast)
                    energy = 1.0 / (1.0 + np.var(frame_gray)) if np.var(frame_gray) > 0 else 1.0
                    correlation = 0.5  # 默认值
            else:
                # 如果没有scikit-image，使用简单替代
                contrast = np.std(frame_gray) / np.mean(frame_gray) if np.mean(frame_gray) > 0 else 0
                dissimilarity = contrast * 0.8
                homogeneity = 1.0 / (1.0 + contrast)
                energy = 1.0 / (1.0 + np.var(frame_gray)) if np.var(frame_gray) > 0 else 1.0
                correlation = 0.5  # 默认值
            
            # 计算边缘特征
            try:
                edges = cv2.Canny(frame_gray, 100, 200)
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            except Exception:
                edge_density = 0.1  # 默认值
            
            # 组合所有特征
            frame_features = np.array([
                mean, std, min_val, max_val, median,
                r_mean, g_mean, b_mean, r_std, g_std, b_std,
                r_ratio, g_ratio, b_ratio,
                contrast, dissimilarity, homogeneity, energy, correlation,
                edge_density
            ])
            
            features_list.append(frame_features)
        
        # 将所有帧的特征合并
        combined_features = np.vstack(features_list)
        
        # 计算统计量作为视频级特征
        mean_features = np.mean(combined_features, axis=0)
        std_features = np.std(combined_features, axis=0)
        max_features = np.max(combined_features, axis=0)
        min_features = np.min(combined_features, axis=0)
        
        # 组合所有统计量
        final_features = np.concatenate([mean_features, std_features, max_features, min_features])
        
        # 确保特征维度与预期一致
        if len(final_features) < self.feature_dim:
            # 如果特征不够长，通过复制扩展
            repetitions = int(np.ceil(self.feature_dim / len(final_features)))
            expanded_features = np.tile(final_features, repetitions)[:self.feature_dim]
        else:
            # 如果特征太长，截断
            expanded_features = final_features[:self.feature_dim]
        
        features_tensor = torch.FloatTensor(expanded_features).unsqueeze(0)
        
        if return_dict:
            # 创建一个类似模型输出的字典
            class SimpleOutput:
                def __init__(self, features):
                    self.last_hidden_state = features.unsqueeze(1)
            return SimpleOutput(features_tensor)
        
        return features_tensor
    
    def _extract_online_features(self, frames, return_dict):
        """使用预训练模型提取特征"""
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

    @staticmethod
    def find_local_models(search_paths=None):
        """
        查找本地的预训练模型
        
        参数:
            search_paths: 要搜索的路径列表
            
        返回:
            model_paths: 找到的模型路径列表
        """
        if search_paths is None:
            # 默认搜索路径
            search_paths = [
                "./models",  # 当前目录下的models文件夹
                os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),  # HuggingFace缓存目录
                os.path.join(os.path.expanduser("~"), ".huggingface"),  # HuggingFace配置目录
            ]
        
        model_paths = []
        
        for path in search_paths:
            if os.path.exists(path):
                # 查找可能的模型目录
                config_files = glob.glob(os.path.join(path, "**", "*config*.json"), recursive=True)
                for config_file in config_files:
                    model_dir = os.path.dirname(config_file)
                    model_paths.append(model_dir)
        
        return model_paths


# 使用示例
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="视频特征提取")
    parser.add_argument("--video", type=str, help="要处理的视频路径")
    parser.add_argument("--offline", action="store_true", help="使用离线模式")
    parser.add_argument("--local-model", type=str, help="本地模型路径")
    parser.add_argument("--find-models", action="store_true", help="查找本地模型")
    
    args = parser.parse_args()
    
    if args.find_models:
        models = VideoFeatureExtractor.find_local_models()
        print(f"找到 {len(models)} 个可能的模型:")
        for i, model_path in enumerate(models):
            print(f"{i+1}. {model_path}")
        sys.exit(0)
    
    # 初始化特征提取器
    extractor = VideoFeatureExtractor(
        offline_mode=args.offline,
        local_model_path=args.local_model
    )
    
    if args.video:
        # 处理单个视频
        features = extractor.process_video(args.video)
        print(f"视频特征维度: {features.shape}")
        
        # 可视化视频帧
        extractor.visualize_frames(args.video)
    else:
        print("请提供视频路径 (--video)")
