"""
Deepfake检测批量测试脚本
遍历samples_test目录，分别测试real和fake文件夹，统计整体准确性
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
CLASS_NAMES = {0: "Real", 1: "Fake"}
LABEL_FOLDERS = {"real": 0, "fake": 1}


def get_transform(img_size=224):
    """获取图片预处理变换"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def load_image(image_path, img_size=224):
    """加载并预处理单张图片"""
    image = Image.open(image_path).convert('RGB')
    transform = get_transform(img_size)
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)


def predict_single(model, image_path, device, img_size=224):
    """对单张图片进行预测，返回预测结果和推理时间"""
    image_tensor = load_image(image_path, img_size).to(device)

    start_time = time.perf_counter()
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
    end_time = time.perf_counter()

    pred_class = torch.argmax(probs, dim=1).item()
    pred_score = probs[0][pred_class].item()
    inference_time = (end_time - start_time) * 1000

    return {
        'predicted_class': pred_class,
        'predicted_label': CLASS_NAMES[pred_class],
        'confidence': pred_score,
        'real_score': probs[0][0].item(),
        'fake_score': probs[0][1].item(),
        'inference_time_ms': inference_time
    }


def load_model(model_path, device, **model_kwargs):
    """加载模型"""
    print(f"Loading model from: {model_path}")
    model = create_model(pretrained_path=model_path, **model_kwargs).to(device)
    print("Model loaded successfully!\n")
    return model


def test_folder(model, folder_path, true_label, device, img_size=224):
    """
    测试单个文件夹中的所有图片

    Args:
        model: 加载好的模型
        folder_path: 文件夹路径
        true_label: 真实标签 (0=real, 1=fake)
        device: 计算设备
        img_size: 图像尺寸

    Returns:
        results: 测试结果列表
    """
    results = []
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.gif')

    # 获取所有图片文件
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(image_extensions)]
    image_files.sort()

    folder_name = os.path.basename(folder_path)
    true_label_name = CLASS_NAMES[true_label]

    print(f"Testing {folder_name}/ ({len(image_files)} images, true label: {true_label_name})")
    print("-" * 60)

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        try:
            result = predict_single(model, img_path, device, img_size)
            result['file_name'] = img_file
            result['true_label'] = true_label
            result['correct'] = (result['predicted_class'] == true_label)
            results.append(result)

            # 只输出单张图片的置信度和类别
            status = "✓" if result['correct'] else "✗"
            print(f"{status} {img_file:<35} -> {result['predicted_label']:>4} ({result['confidence']:.4f})")

        except Exception as e:
            print(f"✗ {img_file:<35} -> Error: {e}")
            results.append({
                'file_name': img_file,
                'true_label': true_label,
                'predicted_class': None,
                'correct': False,
                'error': str(e)
            })

    return results


def calculate_metrics(results, label_name):
    """计算指定标签的指标"""
    total = len(results)
    correct = sum(1 for r in results if r.get('correct', False))
    accuracy = correct / total if total > 0 else 0

    # 计算平均推理时间
    valid_times = [r['inference_time_ms'] for r in results if 'inference_time_ms' in r]
    avg_time = sum(valid_times) / len(valid_times) if valid_times else 0

    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'avg_inference_time_ms': avg_time
    }


def print_summary(all_results, real_results, fake_results):
    """打印测试汇总结果"""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    # Real文件夹统计
    if real_results:
        real_metrics = calculate_metrics(real_results, 'real')
        print(f"\n[Real Folder] True Label: Real (0)")
        print(f"  Total Images:   {real_metrics['total']}")
        print(f"  Correct:        {real_metrics['correct']}")
        print(f"  Accuracy:       {real_metrics['accuracy']:.4f} ({real_metrics['accuracy']*100:.2f}%)")
        print(f"  Avg Time:       {real_metrics['avg_inference_time_ms']:.2f} ms")

    # Fake文件夹统计
    if fake_results:
        fake_metrics = calculate_metrics(fake_results, 'fake')
        print(f"\n[Fake Folder] True Label: Fake (1)")
        print(f"  Total Images:   {fake_metrics['total']}")
        print(f"  Correct:        {fake_metrics['correct']}")
        print(f"  Accuracy:       {fake_metrics['accuracy']:.4f} ({fake_metrics['accuracy']*100:.2f}%)")
        print(f"  Avg Time:       {fake_metrics['avg_inference_time_ms']:.2f} ms")

    # 整体统计
    total_images = len(all_results)
    total_correct = sum(1 for r in all_results if r.get('correct', False))
    overall_accuracy = total_correct / total_images if total_images > 0 else 0

    all_times = [r['inference_time_ms'] for r in all_results if 'inference_time_ms' in r]
    overall_avg_time = sum(all_times) / len(all_times) if all_times else 0
    total_time = sum(all_times) if all_times else 0

    print(f"\n[Overall Statistics]")
    print(f"  Total Images:   {total_images}")
    print(f"  Correct:        {total_correct}")
    print(f"  Failed:         {total_images - total_correct}")
    print(f"  Accuracy:       {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"  Avg Time/img:   {overall_avg_time:.2f} ms")
    print(f"  Total Time:     {total_time:.2f} ms ({total_time/1000:.2f} s)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Batch Test Deepfake Detection')

    parser.add_argument('--test_dir', type=str, default='./samples_test',
                        help='测试数据目录（包含real和fake子目录）')
    parser.add_argument('--model_path', type=str,
                        default='./output/run_20260413_133506/checkpoints/best_model.pth',
                        help='模型权重文件路径')

    # 模型结构参数
    parser.add_argument('--freq_dim', type=int, default=512)
    parser.add_argument('--vit_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.5)

    # 推理参数
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()

    # 检查测试目录
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory not found: {args.test_dir}")
        sys.exit(1)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

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

        # 预热
        print("Warming up model...")
        dummy_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)
        print("Warmup completed.\n")

        # 测试Real文件夹
        real_dir = os.path.join(args.test_dir, 'real')
        real_results = []
        if os.path.exists(real_dir):
            real_results = test_folder(model, real_dir, LABEL_FOLDERS['real'],
                                       device, args.img_size)
            print()
        else:
            print(f"Warning: real/ directory not found in {args.test_dir}\n")

        # 测试Fake文件夹
        fake_dir = os.path.join(args.test_dir, 'fake')
        fake_results = []
        if os.path.exists(fake_dir):
            fake_results = test_folder(model, fake_dir, LABEL_FOLDERS['fake'],
                                       device, args.img_size)
            print()
        else:
            print(f"Warning: fake/ directory not found in {args.test_dir}\n")

        # 汇总结果
        all_results = real_results + fake_results
        print_summary(all_results, real_results, fake_results)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()