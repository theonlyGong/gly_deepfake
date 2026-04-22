"""
ViT-L-14 模型测试脚本
实例化模型、加载权重并测试输出
"""
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.Vit_model.Vit import vit_l_14


def main():
    print("=" * 60)
    print("ViT-L-14 模型测试")
    print("=" * 60)

    # 1. 实例化模型
    print("\n[1] 实例化 ViT-L-14 模型...")
    model = vit_l_14()  # 特征提取模式
    print(f"    模型配置:")
    print(f"    - Image size: 224")
    print(f"    - Patch size: 14")
    print(f"    - Embed dim: 1024")
    print(f"    - Depth: 24")
    print(f"    - Num heads: 16")
    print(f"    [OK] Model instantiated successfully")

    # 2. 加载权重
    checkpoint_path = "checkpoints/ViT-L-14.pt"
    print(f"\n[2] 加载权重文件: {checkpoint_path}")

    if os.path.exists(checkpoint_path):
        try:
            model.load_from_checkpoint(checkpoint_path, strict=False)
            print(f"    [OK] Weights loaded successfully")
        except Exception as e:
            print(f"    [X] Weight loading failed: {e}")
            print(f"    将继续使用随机初始化权重进行测试...")
    else:
        print(f"    [!] Weight file not found: {checkpoint_path}")
        print(f"    将使用随机初始化权重进行测试...")

    # 3. 设置模型为评估模式
    model.eval()
    print(f"\n[3] 模型已设置为评估模式 (eval)")

    # 4. 创建测试输入
    batch_size = 1
    print(f"\n[4] 创建测试输入 tensor")
    print(f"    - Batch size: {batch_size}")
    print(f"    - Channels: 3")
    print(f"    - Height: 224")
    print(f"    - Width: 224")

    x = torch.randn(batch_size, 3, 224, 224)
    print(f"    [OK] Input tensor shape: {x.shape}")

    # 5. 前向传播
    print(f"\n[5] 执行前向传播...")
    with torch.no_grad():
        output = model(x)

    # 6. 输出结果
    print(f"\n[6] 输出结果")
    print(f"    {'=' * 40}")
    print(f"    输出 tensor shape: {output.shape}")
    print(f"    {'=' * 40}")
    print(f"\n    详细说明:")
    print(f"    - Batch size: {output.shape[0]}")
    print(f"    - Feature dim: {output.shape[1]}")
    print(f"\n    [OK] Test completed!")

    return output.shape


if __name__ == "__main__":
    output_shape = main()