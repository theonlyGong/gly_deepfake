import torch
import torch.nn as nn
import torch.nn.functional as F

from model.freqnet_model.freqnet_exetractor import FreqNet, freqnet
from model.Vit_model.Vit import ViT_L_14, vit_l_14


class CrossAttentionFusion(nn.Module):
    """
    使用8头Cross-Attention进行特征融合
    FreqNet特征作为Query，ViT特征作为Key和Value
    """
    def __init__(self, freq_dim=512, vit_dim=1024, hidden_dim=512, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        self.freq_dim = freq_dim
        self.vit_dim = vit_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 将ViT特征投影到与FreqNet相同的维度
        self.vit_proj = nn.Linear(vit_dim, hidden_dim)

        # 将FreqNet特征投影到hidden_dim
        self.freq_proj = nn.Linear(freq_dim, hidden_dim)

        # Multi-head Cross-Attention
        # Query来自FreqNet，Key和Value来自ViT
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
        """
        Args:
            freq_feat: (B, 512) FreqNet特征
            vit_feat: (B, 1024) ViT特征
        Returns:
            fused_feat: (B, hidden_dim) 融合后的特征
        """
        B = freq_feat.size(0)

        # 投影特征
        freq_proj = self.freq_proj(freq_feat)  # (B, hidden_dim)
        vit_proj = self.vit_proj(vit_feat)     # (B, hidden_dim)

        # 增加序列维度用于attention: (B, 1, hidden_dim)
        freq_proj = freq_proj.unsqueeze(1)
        vit_proj = vit_proj.unsqueeze(1)

        # Cross-Attention: Query来自FreqNet，Key和Value来自ViT
        # 这样FreqNet特征可以关注ViT特征中的相关信息
        attn_out, _ = self.cross_attn(
            query=freq_proj,      # (B, 1, hidden_dim)
            key=vit_proj,         # (B, 1, hidden_dim)
            value=vit_proj        # (B, 1, hidden_dim)
        )

        # 残差连接和LayerNorm
        freq_proj = self.norm1(freq_proj + attn_out)

        # FFN
        ffn_out = self.ffn(freq_proj)
        fused_feat = self.norm2(freq_proj + ffn_out)

        # 移除序列维度: (B, hidden_dim)
        fused_feat = fused_feat.squeeze(1)

        return fused_feat


class GlyFusionModel(nn.Module):
    """
    融合FreqNet和ViT的Deepfake检测模型
    使用8头Cross-Attention进行特征融合
    """
    def __init__(
        self,
        freq_dim=512,
        vit_dim=768,  # CLIP ViT-L/14 proj 输出维度
        hidden_dim=512,
        num_heads=8,
        num_classes=2,
        dropout=0.5,
        freqnet_checkpoint=None,
        vit_checkpoint=None
    ):
        super(GlyFusionModel, self).__init__()

        # 加载两个 backbone
        self.freqnet = freqnet()
        # ViT-L-14 CLIP 模型的 proj 输出维度是 768
        self.vit = vit_l_14(output_dim=768)

        # 加载预训练权重
        if freqnet_checkpoint is not None:
            self.load_freqnet_checkpoint(freqnet_checkpoint)
        if vit_checkpoint is not None:
            self.load_vit_checkpoint(vit_checkpoint)

        # 冻结 backbone（可选，根据训练需求）
        # self._freeze_backbones()

        # Cross-Attention融合模块
        self.fusion_module = CrossAttentionFusion(
            freq_dim=freq_dim,
            vit_dim=vit_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )

        # 分类头：输入是融合特征，输出是2维（二分类）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self._initialize_weights()

    def _freeze_backbones(self):
        """冻结backbone参数，只训练融合和分类部分"""
        for param in self.freqnet.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False

    def load_freqnet_checkpoint(self, checkpoint_path, strict=True):
        """
        加载FreqNet预训练权重

        Args:
            checkpoint_path: 权重文件路径
            strict: 是否严格匹配所有键
        """
        import warnings
        print(f"Loading FreqNet checkpoint from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # 处理不同格式的checkpoint
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # 加载权重
        missing_keys, unexpected_keys = self.freqnet.load_state_dict(state_dict, strict=strict)

        if missing_keys:
            print(f"  Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys}")

        print(f"  FreqNet checkpoint loaded successfully!")

    def load_vit_checkpoint(self, checkpoint_path, strict=True):
        """
        加载ViT预训练权重（支持CLIP格式，处理proj转置）

        Args:
            checkpoint_path: 权重文件路径
            strict: 是否严格匹配所有键
        """
        import warnings
        print(f"Loading ViT checkpoint from: {checkpoint_path}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # 尝试加载为 TorchScript
                scripted_model = torch.jit.load(checkpoint_path, map_location='cpu')
                state_dict = scripted_model.state_dict()
            except Exception:
                # 回退到普通加载
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict):
                    if 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    elif 'model' in checkpoint:
                        state_dict = checkpoint['model']
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

        # 筛选 visual encoder 的权重并移除 'visual.' 前缀
        visual_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('visual.'):
                new_k = k.replace('visual.', '')
                visual_state_dict[new_k] = v

        # 处理 proj 的转置：CLIP中是 [embed_dim, output_dim]，模型中是 [output_dim, embed_dim]
        if 'proj' in visual_state_dict:
            proj_weight = visual_state_dict['proj']
            if proj_weight.shape[0] == self.vit.embed_dim and proj_weight.shape[1] == self.vit.output_dim:
                # 需要转置: [1024, 768] -> [768, 1024]
                visual_state_dict['proj'] = proj_weight.T
                print(f"  Transposed proj weight from {proj_weight.shape} to {visual_state_dict['proj'].shape}")

        # 加载权重
        missing_keys, unexpected_keys = self.vit.load_state_dict(visual_state_dict, strict=strict)

        if missing_keys:
            print(f"  Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys}")

        print(f"  ViT checkpoint loaded successfully!")

    def _initialize_weights(self):
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

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入图像
        Returns:
            output: (B, 2) 分类输出（logits）
        """
        # 提取两个分支的特征
        freq_feat = self.freqnet(x)  # (B, 512)
        vit_feat = self.vit(x)       # (B, 1024)

        # Cross-Attention融合
        fused_feat = self.fusion_module(freq_feat, vit_feat)  # (B, hidden_dim)

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
            'freq_feat': freq_feat,      # (B, 512)
            'vit_feat': vit_feat,        # (B, 1024)
            'fused_feat': fused_feat,    # (B, hidden_dim)
            'logits': output             # (B, 2)
        }


def gly_fusion_model(freqnet_checkpoint=None, vit_checkpoint=None, **kwargs):
    """创建融合模型

    Args:
        freqnet_checkpoint: FreqNet预训练权重路径
        vit_checkpoint: ViT预训练权重路径
        **kwargs: 其他模型参数
    """
    return GlyFusionModel(
        freqnet_checkpoint=freqnet_checkpoint,
        vit_checkpoint=vit_checkpoint,
        **kwargs
    )


if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 预训练权重路径（根据实际情况修改）
    FREQNET_CHECKPOINT = "../checkpoints/freqnet_backbone_only.pth"
    VIT_CHECKPOINT = "../checkpoints/ViT-L-14.pt"

    # 创建模型并加载预训练权重
    model = gly_fusion_model(
        freqnet_checkpoint=FREQNET_CHECKPOINT,
        vit_checkpoint=VIT_CHECKPOINT
    ).to(device)

    # 或者：不加载预训练权重（从头训练）
    # model = gly_fusion_model().to(device)

    # 测试输入
    batch_size = 1
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
            print(f"{name}: {feat.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")