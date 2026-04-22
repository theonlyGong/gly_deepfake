"""
Deepfake检测单张图片预测脚本
使用 GlyFusionModelV2 进行单张图片推理
输出：预测类别、预测分数、预测时间
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import time

from model.gly_model_v2 import create_model


# 类别映射
CLASS_NAMES = {
    0: "Real",
    1: "Fake"
}


def get_transform(img_size=224):
    """获取单张图片的预处理变换（测试模式，无数据增强）"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def load_image(image_path, img_size=224):
    """
    加载并预处理单张图片

    Args:
        image_path: 图片文件路径
        img_size: 目标图像尺寸

    Returns:
        tensor: 预处理后的图像张量 (1, C, H, W)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 加载图片
    image = Image.open(image_path).convert('RGB')

    # 预处理
    transform = get_transform(img_size)
    image_tensor = transform(image)

    # 增加batch维度
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def predict_single(model, image_path, device, img_size=224):
    """
    对单张图片进行预测

    Args:
        model: 加载好的模型
        image_path: 图片路径
        device: 计算设备
        img_size: 图像尺寸

    Returns:
        dict: 包含预测类别、分数和时间的字典
    """
    # 加载并预处理图片
    image_tensor = load_image(image_path, img_size)
    image_tensor = image_tensor.to(device)

    # 开始计时
    start_time = time.perf_counter()

    # 模型推理
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)

    # 结束计时
    end_time = time.perf_counter()
    inference_time = (end_time - start_time) * 1000  # 转换为毫秒

    # 获取预测结果
    pred_class = torch.argmax(probs, dim=1).item()
    pred_score = probs[0][pred_class].item()

    # 获取各类别的分数
    class_scores = {
        CLASS_NAMES[0]: probs[0][0].item(),
        CLASS_NAMES[1]: probs[0][1].item()
    }

    return {
        'image_path': image_path,
        'predicted_class': CLASS_NAMES[pred_class],
        'predicted_label': pred_class,
        'confidence': pred_score,
        'class_scores': class_scores,
        'inference_time_ms': inference_time
    }


def load_model(model_path, device, **model_kwargs):
    """
    加载模型

    Args:
        model_path: 模型权重文件路径
        device: 计算设备
        **model_kwargs: 模型结构参数

    Returns:
        model: 加载好的模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from: {model_path}")

    # 创建模型并加载权重
    model = create_model(
        pretrained_path=model_path,
        **model_kwargs
    ).to(device)

    print("Model loaded successfully!")

    return model


def main():
    parser = argparse.ArgumentParser(description='Single Image Deepfake Detection')

    # 输入参数
    parser.add_argument('--image', type=str, default = './test_images/fake/pic1.jpg',
                        help='待预测的图片路径')
    parser.add_argument('--model_path', type=str,
                        default='./output/run_20260413_133506/checkpoints/best_model.pth',
                        help='模型权重文件路径')

    # 模型结构参数（应与训练时一致）
    parser.add_argument('--freq_dim', type=int, default=512,
                        help='FreqNet输出维度')
    parser.add_argument('--vit_dim', type=int, default=768,
                        help='ViT输出维度')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='融合层隐藏维度')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Cross-Attention头数')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout率')

    # 推理参数
    parser.add_argument('--img_size', type=int, default=224,
                        help='输入图像尺寸')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='推理设备')

    args = parser.parse_args()

    # 检查输入图片是否存在
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 模型参数
    model_kwargs = {
        'freq_dim': args.freq_dim,
        'vit_dim': args.vit_dim,
        'hidden_dim': args.hidden_dim,
        'num_heads': args.num_heads,
        'num_classes': 2,
        'dropout': args.dropout,
    }

    try:
        # 加载模型
        model = load_model(args.model_path, device, **model_kwargs)

        # 预热（可选，用于更准确的计时）
        print("\nWarming up model...")
        dummy_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)
        print("Warmup completed.\n")

        # 执行预测
        print(f"Predicting image: {args.image}")
        print("-" * 50)

        result = predict_single(model, args.image, device, args.img_size)

        # 输出结果
        print(f"\nPrediction Results:")
        print("=" * 50)
        print(f"Image Path:    {result['image_path']}")
        print(f"Predicted:     {result['predicted_class']} (Label: {result['predicted_label']})")
        print(f"Confidence:    {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"-" * 50)
        print(f"Class Scores:")
        print(f"  Real:        {result['class_scores']['Real']:.4f} ({result['class_scores']['Real']*100:.2f}%)")
        print(f"  Fake:        {result['class_scores']['Fake']:.4f} ({result['class_scores']['Fake']*100:.2f}%)")
        print(f"-" * 50)
        print(f"Inference Time: {result['inference_time_ms']:.2f} ms")
        print("=" * 50)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()