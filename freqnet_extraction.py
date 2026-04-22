import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from model.freqnet_model.freqnet_exetractor import freqnet


def predict_single_image(image_path, model_path, loadSize=256, cropSize=224, device='cuda'):
    """

    Args:
        image_path: 图片路径
        model_path: 模型权重路径
        loadSize: 缩放尺寸 (默认256)
        cropSize: 裁剪尺寸 (默认224)
        device: 计算设备 ('cuda' 或 'cpu')

    Returns:
        prediction: 预测结果 (0-1之间的概率值)
        label: 预测标签 ('Real' 或 'Fake')
    """
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = freqnet()
    # print(model)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    # 定义预处理变换 (与测试时一致)
    transform = transforms.Compose([
        transforms.Resize((loadSize, loadSize)),
        transforms.CenterCrop(cropSize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 加载并预处理图片
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)  # 添加batch维度

    # 预测
    with torch.no_grad():
        output = model(img_tensor)
        # prediction = output.sigmoid().item()

    # 判断标签 (>0.5为Fake，否则为Real)
    # label = 'Fake' if prediction > 0.5 else 'Real'

    return output


def main():
    parser = argparse.ArgumentParser(description='单张图片深度伪造特征提取')
    parser.add_argument('--image_path', type=str, default='./test_images/real',
                        help='待检测图片的路径')
    parser.add_argument('--model_path', type=str, default='./checkpoints/freqnet_backbone_only.pth',
                        help='模型权重文件路径')
    parser.add_argument('--loadSize', type=int, default=256,
                        help='图片缩放尺寸 (默认: 256)')
    parser.add_argument('--cropSize', type=int, default=224,
                        help='图片裁剪尺寸 (默认: 224)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备: cuda 或 cpu (默认: cuda)')
    # parser.add_argument('--threshold', type=float, default=0.5,
    #                     help='判断真伪的阈值 (默认: 0.5)')

    opt = parser.parse_args()

    # 进行预测
    print(f"使用模型: {opt.model_path}")
    print("-" * 40)

    for img_path in os.listdir(opt.image_path):
        print("-"*40)
        img_path = opt.image_path+'/'+img_path
        print(f"正在检测图片: {img_path}")
        extracted_feature = predict_single_image(
            img_path,
            opt.model_path,
            opt.loadSize,
            opt.cropSize,
            opt.device
        )

        # 输出结果
        print(f"提取特征: {extracted_feature}")
        print(extracted_feature.shape)



if __name__ == '__main__':
    main()