"""
用于下载和管理HuggingFace模型的脚本
这个脚本可以帮助你在良好的网络环境中下载模型，然后在离线环境中使用
"""
import os
import sys
import argparse
from huggingface_hub import snapshot_download, hf_hub_download, login
import torch
from transformers import AutoProcessor, AutoModel

def download_model(model_name, cache_dir=None, use_auth_token=None):
    """
    下载完整的模型到指定目录
    
    参数:
        model_name: 模型名称或路径
        cache_dir: 缓存目录，默认为None (使用huggingface默认缓存)
        use_auth_token: HuggingFace API令牌，用于访问私有模型
    
    返回:
        model_dir: 模型目录路径
    """
    print(f"下载模型 {model_name} 到 {cache_dir if cache_dir else 'HuggingFace默认缓存目录'}")
    
    try:
        # 设置环境变量以禁用系统代理
        os.environ['HTTP_PROXY'] = ''
        os.environ['HTTPS_PROXY'] = ''
        os.environ['http_proxy'] = ''
        os.environ['https_proxy'] = ''
        
        # 下载模型
        model_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            token=use_auth_token,
            local_files_only=False,
            ignore_patterns=["*.safetensors", "*.bin", "*.h5"] if "--lite" in sys.argv else None
        )
        
        print(f"模型下载成功！保存在: {model_dir}")
        return model_dir
        
    except Exception as e:
        print(f"下载模型时出错: {e}")
        print("请检查您的网络连接或代理设置")
        return None

def verify_model(model_name, model_dir):
    """
    验证模型是否可以正确加载
    
    参数:
        model_name: 模型名称
        model_dir: 模型目录路径
    
    返回:
        is_valid: 模型是否有效
    """
    try:
        print(f"验证模型 {model_name} 是否可以加载...")
        
        # 尝试加载模型及处理器
        processor = AutoProcessor.from_pretrained(model_dir)
        model = AutoModel.from_pretrained(model_dir)
        
        print("✓ 模型验证成功！")
        return True
        
    except Exception as e:
        print(f"验证模型时出错: {e}")
        return False

def download_specific_files(model_name, files, output_dir, use_auth_token=None):
    """
    下载模型的特定文件
    
    参数:
        model_name: 模型名称
        files: 文件列表
        output_dir: 输出目录
        use_auth_token: HuggingFace API令牌
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for file in files:
        try:
            print(f"下载文件: {file}")
            local_file = hf_hub_download(
                repo_id=model_name,
                filename=file,
                cache_dir=output_dir,
                token=use_auth_token
            )
            print(f"文件已保存到: {local_file}")
        except Exception as e:
            print(f"下载文件 {file} 时出错: {e}")

def setup_auth():
    """
    设置HuggingFace认证
    """
    token = input("请输入您的HuggingFace令牌 (如果不需要，直接按Enter): ")
    if token:
        login(token=token)
        return token
    return None

def main():
    parser = argparse.ArgumentParser(description="HuggingFace模型下载工具")
    parser.add_argument("--model", type=str, default="alibaba-pai/VideoCLIP-XL", help="要下载的模型名称或路径")
    parser.add_argument("--cache-dir", type=str, default="./models", help="缓存目录")
    parser.add_argument("--auth", action="store_true", help="使用认证")
    parser.add_argument("--lite", action="store_true", help="轻量级下载(不包括大文件)")
    parser.add_argument("--verify", action="store_true", help="验证模型")
    
    args = parser.parse_args()
    
    # 设置认证
    token = setup_auth() if args.auth else None
    
    # 下载模型
    model_dir = download_model(args.model, args.cache_dir, token)
    
    # 验证模型
    if args.verify and model_dir:
        verify_model(args.model, model_dir)
    
    print("\n==== 使用说明 ====")
    print(f"您现在可以通过以下代码加载模型:")
    print(f"""
from transformers import AutoProcessor, AutoModel

# 使用本地模型路径
model_path = "{model_dir}"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)
    """)

if __name__ == "__main__":
    main()
