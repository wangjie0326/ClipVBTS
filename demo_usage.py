import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from video_feature_extractor import VideoFeatureExtractor
from video_classifier import VideoClassifier

def visualize_features(features, labels, classes, title="t-SNE特征可视化", method="tsne"):
    """可视化特征"""
    plt.figure(figsize=(10, 8))
    
    # 降维
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:  # PCA
        reducer = PCA(n_components=2, random_state=42)
        
    embedded = reducer.fit_transform(features)
    
    # 绘制每个类别的点
    for i, class_name in enumerate(classes):
        idx = labels == i
        plt.scatter(
            embedded[idx, 0], 
            embedded[idx, 1], 
            label=class_name,
            alpha=0.7
        )
    
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(alpha=0.3)
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)
    plt.show()

def run_training(data_dir, output_dir, max_frames=8, epochs=50):
    """训练视频分类器"""
    print(f"开始训练，数据目录: {data_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化分类器
    classifier = VideoClassifier(classifier_type="mlp")
    
    # 从目录中提取特征
    cache_file = os.path.join(output_dir, "features_cache.pkl")
    features, labels = classifier.extract_features_from_directory(
        data_dir, 
        max_frames=max_frames,
        cache_file=cache_file
    )
    
    print(f"提取的特征形状: {features.shape}, 标签形状: {labels.shape}")
    
    # 可视化原始特征
    visualize_features(
        features, 
        labels, 
        classifier.classes, 
        title="原始特征t-SNE可视化",
        method="tsne"
    )
    
    # 训练分类器
    print("开始训练分类器...")
    history = classifier.train(
        features, 
        labels, 
        num_epochs=epochs,
        batch_size=16,
        learning_rate=0.0005
    )
    
    # 绘制训练历史
    classifier.plot_history(history)
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=300)
    
    # 评估分类器
    metrics = classifier.evaluate(features, labels)
    print("\n=== 评估结果 ===")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"加权精确率: {metrics['precision']:.4f}")
    print(f"加权召回率: {metrics['recall']:.4f}")
    print(f"加权F1分数: {metrics['f1']:.4f}")
    
    # 保存每个类别的指标
    print("\n=== 每个类别的指标 ===")
    for class_name, class_metrics in metrics['class_metrics'].items():
        print(f"{class_name}: 精确率={class_metrics['precision']:.4f}, "
              f"召回率={class_metrics['recall']:.4f}, "
              f"F1={class_metrics['f1']:.4f}")
    
    # 保存模型
    model_path = os.path.join(output_dir, "digit_video_classifier.pth")
    classifier.save(model_path)
    print(f"\n模型已保存到: {model_path}")
    
    return classifier, features, labels

def run_inference(model_path, test_video, feature_extractor=None):
    """使用训练好的模型进行推理"""
    print(f"加载模型: {model_path}")
    
    # 初始化分类器
    classifier = VideoClassifier(feature_extractor)
    classifier.load(model_path)
    
    # 为特征提取器可视化帧
    print(f"可视化视频帧: {test_video}")
    classifier.feature_extractor.visualize_frames(test_video, max_frames=8)
    
    # 预测视频类别
    class_idx, class_name, confidence = classifier.predict_video(test_video)
    print(f"\n预测结果: {class_name} (置信度: {confidence:.4f})")
    
    # 显示前三个预测
    features = classifier.feature_extractor.process_video(test_video)
    probs = classifier.predict_proba(features)[0]
    
    # 获取前三个预测
    top3_idx = np.argsort(probs)[::-1][:3]
    print("\n前三个预测:")
    for idx in top3_idx:
        print(f"{classifier.classes[idx]}: {probs[idx]:.4f}")
    
    return class_name, confidence

def run_batch_inference(model_path, test_dir):
    """批量推理多个视频"""
    print(f"加载模型: {model_path}")
    
    # 初始化分类器
    classifier = VideoClassifier()
    classifier.load(model_path)
    
    # 获取测试目录中的所有视频
    videos = []
    for ext in ['mp4', 'avi', 'mov']:
        videos.extend(list(Path(test_dir).glob(f"*.{ext}")))
    
    print(f"找到 {len(videos)} 个测试视频")
    
    results = []
    for video_path in videos:
        print(f"\n处理视频: {video_path}")
        class_idx, class_name, confidence = classifier.predict_video(str(video_path))
        print(f"预测结果: {class_name} (置信度: {confidence:.4f})")
        results.append({
            'video': str(video_path),
            'prediction': class_name,
            'confidence': confidence
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='DIGIT视觉触觉传感器视频分类')
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 训练命令参数
    train_parser = subparsers.add_parser('train', help='训练分类器')
    train_parser.add_argument('--data', type=str, required=True, help='训练数据目录')
    train_parser.add_argument('--output', type=str, default='output', help='输出目录')
    train_parser.add_argument('--frames', type=int, default=8, help='每个视频的最大帧数')
    train_parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    
    # 推理命令参数
    infer_parser = subparsers.add_parser('infer', help='对单个视频进行推理')
    infer_parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    infer_parser.add_argument('--video', type=str, required=True, help='要预测的视频文件')
    
    # 批量推理命令参数
    batch_parser = subparsers.add_parser('batch', help='批量推理多个视频')
    batch_parser.add_argument('--model', type=str, required=True, help='模型文件路径')
    batch_parser.add_argument('--dir', type=str, required=True, help='包含视频的目录')
    
    # 特征提取命令参数
    extract_parser = subparsers.add_parser('extract', help='仅提取特征')
    extract_parser.add_argument('--video', type=str, required=True, help='要提取特征的视频')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        run_training(args.data, args.output, args.frames, args.epochs)
    
    elif args.command == 'infer':
        run_inference(args.model, args.video)
    
    elif args.command == 'batch':
        run_batch_inference(args.model, args.dir)
    
    elif args.command == 'extract':
        # 仅提取特征并显示
        extractor = VideoFeatureExtractor()
        frames = extractor.extract_frames(args.video, max_frames=8)
        features = extractor.extract_features(frames)
        
        print(f"提取的特征维度: {features.shape}")
        extractor.visualize_frames(args.video, max_frames=8)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()