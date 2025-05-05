import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from typing import List, Dict, Tuple, Optional, Union

from video_feature_extractor import VideoFeatureExtractor

class VideoClassifier:
    """
    基于提取的视频特征进行分类的分类器
    适用于DIGIT视觉触觉传感器的视频数据分类
    """
    
    def __init__(
        self, 
        feature_extractor: Optional[VideoFeatureExtractor] = None,
        model_name: str = "alibaba-pai/VideoCLIP-XL",
        device: str = None,
        classifier_type: str = "mlp",
        offline_mode: bool = False
    ):
        """
        初始化视频分类器
        
        参数:
            feature_extractor: 视频特征提取器，如果为None则创建新的
            model_name: 模型名称或路径，用于创建新的特征提取器
            device: 运行设备，如果为None则自动选择
            classifier_type: 分类器类型，支持 'mlp', 'linear'
            offline_mode: 是否使用离线模式(不下载预训练模型)
        """
        if feature_extractor is None:
            self.feature_extractor = VideoFeatureExtractor(model_name, device, offline_mode=offline_mode)
        else:
            self.feature_extractor = feature_extractor
            
        self.device = self.feature_extractor.device
        self.classifier_type = classifier_type
        self.classifier = None
        self.classes = None
        self.feature_dim = None
        self.offline_mode = offline_mode or (hasattr(self.feature_extractor, 'offline_mode') and self.feature_extractor.offline_mode)
        
    def build_classifier(self, num_classes: int, feature_dim: int):
        """
        构建分类器模型
        
        参数:
            num_classes: 类别数
            feature_dim: 特征维度
        """
        self.feature_dim = feature_dim
        
        if self.classifier_type == "linear":
            self.classifier = torch.nn.Linear(feature_dim, num_classes).to(self.device)
        elif self.classifier_type == "mlp":
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(feature_dim, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(512, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(128, num_classes)
            ).to(self.device)
        else:
            raise ValueError(f"不支持的分类器类型: {self.classifier_type}")
            
    def extract_features_from_directory(
        self, 
        data_dir: str,
        max_frames: int = 8,
        cache_file: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从目录中提取视频特征
        目录结构应为:
        data_dir/
          ├── class1/
          │     ├── video1.mp4
          │     ├── video2.mp4
          │     └── ...
          ├── class2/
          │     ├── video1.mp4
          │     └── ...
          └── ...
          
        参数:
            data_dir: 数据目录
            max_frames: 每个视频的最大帧数
            cache_file: 缓存文件路径，如果提供则保存/加载特征
            
        返回:
            features: 特征数组
            labels: 标签数组
        """
        # 检查是否有缓存
        if cache_file and os.path.exists(cache_file):
            print(f"加载缓存的特征从 {cache_file}")
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                return cache_data['features'], cache_data['labels']
        
        # 获取类别
        self.classes = sorted([d for d in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, d))])
        print(f"发现类别: {self.classes}")
        
        features = []
        labels = []
        
        # 遍历每个类别
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            video_files = [f for f in os.listdir(class_dir) 
                          if f.endswith(('.mp4', '.avi', '.mov'))]
            
            print(f"处理类别 '{class_name}' 中的 {len(video_files)} 个视频...")
            
            # 遍历每个视频
            for video_file in tqdm(video_files):
                video_path = os.path.join(class_dir, video_file)
                try:
                    # 提取特征
                    feature = self.feature_extractor.process_video(video_path, max_frames)
                    features.append(feature.numpy())
                    labels.append(class_idx)
                except Exception as e:
                    print(f"处理视频 {video_path} 时出错: {e}")
                    continue
        
        features = np.vstack(features)
        labels = np.array(labels)
        
        # 保存到缓存
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({'features': features, 'labels': labels}, f)
            print(f"特征已缓存到 {cache_file}")
            
        return features, labels
    
    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        val_ratio: float = 0.2,
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        early_stopping: int = 5
    ) -> Dict:
        """
        训练分类器
        
        参数:
            features: 特征数组
            labels: 标签数组
            val_ratio: 验证集比例
            num_epochs: 训练轮数
            batch_size: 批大小
            learning_rate: 学习率
            weight_decay: 权重衰减
            early_stopping: 早停轮数
            
        返回:
            history: 训练历史
        """
        # 确保类别已设置
        if self.classes is None:
            self.classes = np.unique(labels)
            
        num_classes = len(self.classes)
        feature_dim = features.shape[1]
        
        # 构建分类器
        if self.classifier is None:
            self.build_classifier(num_classes, feature_dim)
            
        # 划分训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=val_ratio, stratify=labels, random_state=42
        )
        
        # 转换为张量
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        
        # 定义损失函数和优化器
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # 学习率调度
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # 训练历史
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # 早停计数器
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        # 训练循环
        for epoch in range(num_epochs):
            # 训练模式
            self.classifier.train()
            train_loss = 0.0
            train_preds = []
            train_true = []
            
            # 创建批次
            indices = torch.randperm(X_train.size(0))
            for start_idx in range(0, X_train.size(0), batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]
                
                # 前向传播
                outputs = self.classifier(X_batch)
                loss = criterion(outputs, y_batch)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 累积损失
                train_loss += loss.item() * X_batch.size(0)
                
                # 保存预测结果
                _, preds = torch.max(outputs, 1)
                train_preds.extend(preds.cpu().numpy())
                train_true.extend(y_batch.cpu().numpy())
            
            # 计算平均损失和准确率
            train_loss /= X_train.size(0)
            train_acc = accuracy_score(train_true, train_preds)
            
            # 评估模式
            self.classifier.eval()
            val_loss = 0.0
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for start_idx in range(0, X_val.size(0), batch_size):
                    end_idx = min(start_idx + batch_size, X_val.size(0))
                    X_batch = X_val[start_idx:end_idx]
                    y_batch = y_val[start_idx:end_idx]
                    
                    # 前向传播
                    outputs = self.classifier(X_batch)
                    loss = criterion(outputs, y_batch)
                    
                    # 累积损失
                    val_loss += loss.item() * X_batch.size(0)
                    
                    # 保存预测结果
                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(y_batch.cpu().numpy())
            
            # 计算平均损失和准确率
            val_loss /= X_val.size(0)
            val_acc = accuracy_score(val_true, val_preds)
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 保存历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= early_stopping:
                print(f"早停在第 {epoch+1} 轮")
                break
                
        return history
    
    def predict(self, features: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        预测类别
        
        参数:
            features: 特征数组或张量
            
        返回:
            predictions: 预测类别索引
        """
        if self.classifier is None:
            raise ValueError("分类器尚未训练")
            
        # 转换为张量
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features).to(self.device)
            
        # 评估模式
        self.classifier.eval()
        
        with torch.no_grad():
            outputs = self.classifier(features)
            _, predictions = torch.max(outputs, 1)
            
        return predictions.cpu().numpy()
    
    def predict_proba(self, features: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        预测类别概率
        
        参数:
            features: 特征数组或张量
            
        返回:
            probabilities: 预测类别概率
        """
        if self.classifier is None:
            raise ValueError("分类器尚未训练")
            
        # 转换为张量
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features).to(self.device)
            
        # 评估模式
        self.classifier.eval()
        
        with torch.no_grad():
            outputs = self.classifier(features)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        return probabilities.cpu().numpy()
    
    def predict_video(self, video_path: str, max_frames: int = 8) -> Tuple[int, str, float]:
        """
        预测单个视频的类别
        
        参数:
            video_path: 视频文件路径
            max_frames: 最大帧数
            
        返回:
            class_idx: 预测类别索引
            class_name: 预测类别名称
            confidence: 置信度
        """
        # 提取特征
        features = self.feature_extractor.process_video(video_path, max_frames)
        
        # 预测
        probs = self.predict_proba(features)
        class_idx = np.argmax(probs, axis=1)[0]
        confidence = probs[0, class_idx]
        
        # 获取类别名称
        class_name = self.classes[class_idx]
        
        return class_idx, class_name, confidence
    
    def evaluate(
        self, 
        features: np.ndarray, 
        labels: np.ndarray
    ) -> Dict:
        """
        评估分类器性能
        
        参数:
            features: 特征数组
            labels: 标签数组
            
        返回:
            metrics: 性能指标字典
        """
        # 预测
        predictions = self.predict(features)
        
        # 计算指标
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        # 每个类别的指标
        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        class_metrics = {}
        for i, class_name in enumerate(self.classes):
            class_metrics[class_name] = {
                'precision': class_precision[i],
                'recall': class_recall[i],
                'f1': class_f1[i]
            }
            
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'class_metrics': class_metrics
        }
    
    def plot_history(self, history: Dict):
        """
        绘制训练历史
        
        参数:
            history: 训练历史字典
        """
        plt.figure(figsize=(12, 5))
        
        # 绘制损失
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        # 绘制准确率
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
    def save(self, path: str):
        """
        保存模型
        
        参数:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            'classifier_state': self.classifier.state_dict() if self.classifier else None,
            'classes': self.classes,
            'feature_dim': self.feature_dim,
            'classifier_type': self.classifier_type
        }
        torch.save(save_dict, path)
        print(f"模型已保存到 {path}")
        
    def load(self, path: str):
        """
        加载模型
        
        参数:
            path: 加载路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.classes = checkpoint['classes']
        self.feature_dim = checkpoint['feature_dim']
        self.classifier_type = checkpoint.get('classifier_type', 'mlp')
        
        # 重建分类器
        self.build_classifier(len(self.classes), self.feature_dim)
        
        # 加载权重
        if checkpoint['classifier_state']:
            self.classifier.load_state_dict(checkpoint['classifier_state'])
        
        print(f"模型已从 {path} 加载")


# 使用示例
if __name__ == "__main__":
    # 初始化分类器
    classifier = VideoClassifier(classifier_type="mlp")
    
    # 从目录中提取特征
    data_dir = "path/to/data"
    features, labels = classifier.extract_features_from_directory(
        data_dir, 
        cache_file="cache/features.pkl"
    )
    
    # 训练分类器
    history = classifier.train(features, labels, num_epochs=30)
    
    # 绘制训练历史
    classifier.plot_history(history)
    
    # 评估分类器
    metrics = classifier.evaluate(features, labels)
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"加权精确率: {metrics['precision']:.4f}")
    print(f"加权召回率: {metrics['recall']:.4f}")
    print(f"加权F1分数: {metrics['f1']:.4f}")
    
    # 保存模型
    classifier.save("models/video_classifier.pth")
    
    # 预测单个视频
    video_path = "path/to/test/video.mp4"
    class_idx, class_name, confidence = classifier.predict_video(video_path)
    print(f"预测类别: {class_name} (置信度: {confidence:.4f})")