"""
GlyFusionModel v2 - 整合FreqNet和ViT的完整模型
支持直接保存/加载完整模型权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ==================== FreqNet模块 ====================

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FreqNetExtractor(nn.Module):
    """FreqNet特征提取器 - 输出512维特征"""

    def __init__(self, block=Bottleneck, layers=[3, 4], zero_init_residual=False):
        super(FreqNetExtractor, self).__init__()

        # 频域卷积参数
        self.weight1 = nn.Parameter(torch.randn((64, 3, 1, 1)))
        self.bias1 = nn.Parameter(torch.randn((64,)))
        self.realconv1 = conv1x1(64, 64, stride=1)
        self.imagconv1 = conv1x1(64, 64, stride=1)

        self.weight2 = nn.Parameter(torch.randn((64, 64, 1, 1)))
        self.bias2 = nn.Parameter(torch.randn((64,)))
        self.realconv2 = conv1x1(64, 64, stride=1)
        self.imagconv2 = conv1x1(64, 64, stride=1)

        self.weight3 = nn.Parameter(torch.randn((256, 256, 1, 1)))
        self.bias3 = nn.Parameter(torch.randn((256,)))
        self.realconv3 = conv1x1(256, 256, stride=1)
        self.imagconv3 = conv1x1(256, 256, stride=1)

        self.weight4 = nn.Parameter(torch.randn((256, 256, 1, 1)))
        self.bias4 = nn.Parameter(torch.randn((256,)))
        self.realconv4 = conv1x1(256, 256, stride=1)
        self.imagconv4 = conv1x1(256, 256, stride=1)

        self.inplanes = 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def hfreqWH(self, x, scale):
        """高频空间滤波"""
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1])
        b, c, h, w = x.shape
        x[:, :, h // 2 - h // scale:h // 2 + h // scale, w // 2 - w // scale:w // 2 + w // scale] = 0.0
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)
        return x

    def hfreqC(self, x, scale):
        """高频通道滤波"""
        x = torch.fft.fft(x, dim=1, norm="ortho")
        x = torch.fft.fftshift(x, dim=1)
        b, c, h, w = x.shape
        x[:, c // 2 - c // scale:c // 2 + c // scale, :, :] = 0.0
        x = torch.fft.ifftshift(x, dim=1)
        x = torch.fft.ifft(x, dim=1, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)
        return x

    def forward(self, x):
        # HFRI
        x = self.hfreqWH(x, 4)
        x = F.conv2d(x, self.weight1, self.bias1, stride=1, padding=0)
        x = F.relu(x, inplace=True)

        # HFRFC
        x = self.hfreqC(x, 4)

        # FCL
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1])
        x = torch.complex(self.realconv1(x.real), self.imagconv1(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        # HFRFS
        x = self.hfreqWH(x, 4)
        x = F.conv2d(x, self.weight2, self.bias2, stride=2, padding=0)
        x = F.relu(x, inplace=True)

        # HFRFC
        x = self.hfreqC(x, 4)

        # FCL
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1])
        x = torch.complex(self.realconv2(x.real), self.imagconv2(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        x = self.maxpool(x)
        x = self.layer1(x)

        # HFRFS
        x = self.hfreqWH(x, 4)
        x = F.conv2d(x, self.weight3, self.bias3, stride=1, padding=0)
        x = F.relu(x, inplace=True)

        # FCL
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1])
        x = torch.complex(self.realconv3(x.real), self.imagconv3(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        # HFRFS
        x = self.hfreqWH(x, 4)
        x = F.conv2d(x, self.weight4, self.bias4, stride=2, padding=0)
        x = F.relu(x, inplace=True)

        # FCL
        x = torch.fft.fft2(x, norm="ortho")
        x = torch.fft.fftshift(x, dim=[-2, -1])
        x = torch.complex(self.realconv4(x.real), self.imagconv4(x.imag))
        x = torch.fft.ifftshift(x, dim=[-2, -1])
        x = torch.fft.ifft2(x, norm="ortho")
        x = torch.real(x)
        x = F.relu(x, inplace=True)

        x = self.layer2(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


# ==================== ViT模块 ====================

class CLIPAttention(nn.Module):
    """Multi-head attention as used in CLIP"""

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        return x


class CLIPMLP(nn.Module):
    """MLP as used in CLIP"""

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.c_fc = nn.Linear(embed_dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x


class CLIPResidualAttentionBlock(nn.Module):
    """Transformer block as used in CLIP"""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CLIPAttention(embed_dim, num_heads)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = CLIPMLP(embed_dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CLIPTransformer(nn.Module):
    """Transformer encoder as used in CLIP visual encoder"""

    def __init__(self, embed_dim: int, num_layers: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.resblocks = nn.ModuleList([
            CLIPResidualAttentionBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.resblocks:
            x = block(x)
        return x


class ViTExtractor(nn.Module):
    """ViT-L/14特征提取器 - 输出768维特征（带proj）或1024维（不带proj）"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        output_dim: Optional[int] = 768,  # CLIP projection输出维度
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Patch embedding
        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        num_patches = (img_size // patch_size) ** 2

        # CLS token
        self.class_embedding = nn.Parameter(torch.empty(embed_dim))
        self.positional_embedding = nn.Parameter(torch.empty(num_patches + 1, embed_dim))

        # Layer norms
        self.ln_pre = nn.LayerNorm(embed_dim)
        self.ln_post = nn.LayerNorm(embed_dim)

        # Transformer
        self.transformer = CLIPTransformer(embed_dim, num_layers, num_heads, mlp_ratio)

        # Projection
        if output_dim is not None:
            self.proj = nn.Parameter(torch.empty(output_dim, embed_dim))
        else:
            self.proj = None

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.class_embedding, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')

        for block in self.transformer.resblocks:
            nn.init.xavier_uniform_(block.attn.in_proj_weight)
            nn.init.zeros_(block.attn.in_proj_bias)
            nn.init.xavier_uniform_(block.attn.out_proj.weight)
            nn.init.zeros_(block.attn.out_proj.bias)
            nn.init.xavier_uniform_(block.mlp.c_fc.weight)
            nn.init.zeros_(block.mlp.c_fc.bias)
            nn.init.xavier_uniform_(block.mlp.c_proj.weight)
            nn.init.zeros_(block.mlp.c_proj.bias)
            nn.init.ones_(block.ln_1.weight)
            nn.init.zeros_(block.ln_1.bias)
            nn.init.ones_(block.ln_2.weight)
            nn.init.zeros_(block.ln_2.bias)

        nn.init.ones_(self.ln_pre.weight)
        nn.init.zeros_(self.ln_pre.bias)
        nn.init.ones_(self.ln_post.weight)
        nn.init.zeros_(self.ln_post.bias)

        if self.proj is not None:
            nn.init.normal_(self.proj, std=self.embed_dim ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        # Add CLS token
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)

        # Add positional embedding
        x = x + self.positional_embedding.to(x.dtype)

        # Pre-transformer norm
        x = self.ln_pre(x)

        # Transformer
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)

        # Post-transformer norm (only on CLS token)
        x = self.ln_post(x[:, 0, :])

        # Optional projection
        if self.proj is not None:
            x = x @ self.proj.T

        return x


# ==================== Cross-Attention融合模块 ====================

class CrossAttentionFusion(nn.Module):
    """使用8头Cross-Attention进行特征融合"""

    def __init__(self, freq_dim=512, vit_dim=768, hidden_dim=512, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        self.freq_dim = freq_dim
        self.vit_dim = vit_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 投影层
        self.vit_proj = nn.Linear(vit_dim, hidden_dim)
        self.freq_proj = nn.Linear(freq_dim, hidden_dim)

        # Multi-head Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )

    def forward(self, freq_feat, vit_feat):
        B = freq_feat.size(0)

        # 投影特征
        freq_proj = self.freq_proj(freq_feat)
        vit_proj = self.vit_proj(vit_feat)

        # 增加序列维度
        freq_proj = freq_proj.unsqueeze(1)
        vit_proj = vit_proj.unsqueeze(1)

        # Cross-Attention
        attn_out, _ = self.cross_attn(
            query=freq_proj,
            key=vit_proj,
            value=vit_proj
        )

        # 残差连接和LayerNorm
        freq_proj = self.norm1(freq_proj + attn_out)

        # FFN
        ffn_out = self.ffn(freq_proj)
        fused_feat = self.norm2(freq_proj + ffn_out)

        # 移除序列维度
        fused_feat = fused_feat.squeeze(1)

        return fused_feat


# ==================== 完整模型 ====================

class GlyFusionModelV2(nn.Module):
    """
    GlyFusion模型v2 - 整合FreqNet和ViT的Deepfake检测模型
    支持直接保存/加载完整模型权重
    """

    def __init__(
        self,
        freq_dim=512,
        vit_dim=768,
        hidden_dim=512,
        num_heads=8,
        num_classes=2,
        dropout=0.5,
        pretrained_path=None  # 可以直接从output目录加载完整权重
    ):
        super(GlyFusionModelV2, self).__init__()

        # 两个backbone
        self.freqnet = FreqNetExtractor()
        self.vit = ViTExtractor(output_dim=vit_dim)

        # Cross-Attention融合模块
        self.fusion_module = CrossAttentionFusion(
            freq_dim=freq_dim,
            vit_dim=vit_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 初始化新添加的层
        self._initialize_new_weights()

        # 加载预训练权重（如果提供）
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    def _initialize_new_weights(self):
        """初始化融合模块和分类器的权重"""
        for m in self.fusion_module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def load_pretrained(self, checkpoint_path, strict=True):
        """
        从output目录加载完整模型权重

        Args:
            checkpoint_path: 权重文件路径 (.pth)
            strict: 是否严格匹配所有键
        """
        import warnings
        print(f"Loading pretrained model from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # 处理不同格式的checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                print(f"  Found 'model_state_dict' key")
                if 'epoch' in checkpoint:
                    print(f"  Checkpoint from epoch: {checkpoint.get('epoch', 'unknown')}")
                if 'metrics' in checkpoint:
                    metrics = checkpoint.get('metrics', {})
                    if isinstance(metrics, dict) and 'f1' in metrics:
                        print(f"  Checkpoint F1 score: {metrics['f1']:.4f}")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # 加载权重
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)

        if missing_keys:
            print(f"  Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys}")

        print(f"  Model loaded successfully!")

    def freeze_backbones(self):
        """冻结backbone参数，只训练融合和分类部分"""
        for param in self.freqnet.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False
        print("Backbones frozen. Only fusion_module and classifier will be trained.")

    def unfreeze_backbones(self):
        """解冻backbone参数"""
        for param in self.freqnet.parameters():
            param.requires_grad = True
        for param in self.vit.parameters():
            param.requires_grad = True
        print("Backbones unfrozen. All parameters will be trained.")

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入图像
        Returns:
            output: (B, 2) 分类输出（logits）
        """
        # 提取两个分支的特征
        freq_feat = self.freqnet(x)  # (B, 512)
        vit_feat = self.vit(x)       # (B, 768)

        # Cross-Attention融合
        fused_feat = self.fusion_module(freq_feat, vit_feat)  # (B, 512)

        # 分类
        output = self.classifier(fused_feat)  # (B, 2)

        return output

    def get_features(self, x):
        """
        获取中间特征（用于分析或可视化）
        Returns:
            dict: 包含各阶段特征的字典
        """
        freq_feat = self.freqnet(x)
        vit_feat = self.vit(x)
        fused_feat = self.fusion_module(freq_feat, vit_feat)
        output = self.classifier(fused_feat)

        return {
            'freq_feat': freq_feat,
            'vit_feat': vit_feat,
            'fused_feat': fused_feat,
            'logits': output
        }

    def save_checkpoint(self, path, epoch=None, optimizer_state=None, metrics=None):
        """
        保存完整checkpoint（包含模型权重、优化器状态、训练信息等）
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'epoch': epoch,
            'metrics': metrics
        }
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")


def create_model(pretrained_path=None, num_classes=2, **kwargs):
    """
    创建GlyFusion模型

    Args:
        pretrained_path: 预训练权重路径（从output目录）
        num_classes: 分类类别数
        **kwargs: 其他模型参数
    """
    model = GlyFusionModelV2(
        num_classes=num_classes,
        pretrained_path=pretrained_path,
        **kwargs
    )
    return model


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建模型（不加载预训练权重）
    model = create_model().to(device)

    # 测试输入
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224).to(device)

    print("\nModel structure:")
    print(model)

    print("\nForward pass test:")
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output: {output}")

    # 获取中间特征
    print("\nFeature extraction test:")
    with torch.no_grad():
        features = model.get_features(x)
        for name, feat in features.items():
            if isinstance(feat, torch.Tensor):
                print(f"{name}: {feat.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 测试保存/加载
    print("\nSave/Load test:")
    test_path = "test_checkpoint.pth"
    model.save_checkpoint(test_path, epoch=0, metrics={'f1': 0.95})

    # 加载权重
    model2 = create_model(pretrained_path=test_path).to(device)
    print("Model loaded successfully!")

    # 清理
    import os
    os.remove(test_path)
    print(f"Removed test file: {test_path}")