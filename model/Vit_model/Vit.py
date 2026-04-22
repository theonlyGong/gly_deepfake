"""
ViT-L-14 Model compatible with OpenAI CLIP weights
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

__all__ = ['ViT_L_14', 'vit_l_14']


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

        # Compute Q, K, V using in_proj
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
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


class ViT_L_14(nn.Module):
    """
    Vision Transformer Large (ViT-L/14) - CLIP compatible

    This implementation matches the structure of OpenAI CLIP ViT-L/14
    to enable direct loading of pretrained weights.

    Config:
    - Image size: 224x224
    - Patch size: 14x14
    - Embed dim: 1024
    - Layers: 24
    - Num heads: 16
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        output_dim: Optional[int] = None,  # For CLIP projection
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Patch embedding (conv1 in CLIP)
        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

        # Calculate number of patches
        num_patches = (img_size // patch_size) ** 2

        # CLS token (class_embedding in CLIP)
        self.class_embedding = nn.Parameter(torch.empty(embed_dim))

        # Positional embedding
        self.positional_embedding = nn.Parameter(torch.empty(num_patches + 1, embed_dim))

        # Pre-transformer layer norm (ln_pre in CLIP)
        self.ln_pre = nn.LayerNorm(embed_dim)

        # Transformer
        self.transformer = CLIPTransformer(embed_dim, num_layers, num_heads, mlp_ratio)

        # Post-transformer layer norm (ln_post in CLIP)
        self.ln_post = nn.LayerNorm(embed_dim)

        # Optional projection layer (proj in CLIP)
        if output_dim is not None:
            self.proj = nn.Parameter(torch.empty(output_dim, embed_dim))
        else:
            self.proj = None

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.class_embedding, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        # Initialize conv1
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Initialize transformer blocks
        for block in self.transformer.resblocks:
            # Attention projection
            nn.init.xavier_uniform_(block.attn.in_proj_weight)
            nn.init.zeros_(block.attn.in_proj_bias)
            nn.init.xavier_uniform_(block.attn.out_proj.weight)
            nn.init.zeros_(block.attn.out_proj.bias)
            # MLP
            nn.init.xavier_uniform_(block.mlp.c_fc.weight)
            nn.init.zeros_(block.mlp.c_fc.bias)
            nn.init.xavier_uniform_(block.mlp.c_proj.weight)
            nn.init.zeros_(block.mlp.c_proj.bias)
            # Layer norms
            nn.init.ones_(block.ln_1.weight)
            nn.init.zeros_(block.ln_1.bias)
            nn.init.ones_(block.ln_2.weight)
            nn.init.zeros_(block.ln_2.bias)

        # Layer norms
        nn.init.ones_(self.ln_pre.weight)
        nn.init.zeros_(self.ln_pre.bias)
        nn.init.ones_(self.ln_post.weight)
        nn.init.zeros_(self.ln_post.bias)

        # Projection
        if self.proj is not None:
            nn.init.normal_(self.proj, std=self.embed_dim ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images
        Returns:
            features: (B, embed_dim) or (B, output_dim) if proj is not None
        """
        # Patch embedding
        x = self.conv1(x)  # (B, embed_dim, H//14, W//14)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, embed_dim, num_patches)
        x = x.permute(0, 2, 1)  # (B, num_patches, embed_dim)

        # Add CLS token
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)  # (B, num_patches+1, embed_dim)

        # Add positional embedding
        x = x + self.positional_embedding.to(x.dtype)

        # Pre-transformer norm
        x = self.ln_pre(x)

        # Transformer expects (N, B, C) format
        x = x.permute(1, 0, 2)  # (num_patches+1, B, embed_dim)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # (B, num_patches+1, embed_dim)

        # Post-transformer norm (only on CLS token)
        x = self.ln_post(x[:, 0, :])

        # Optional projection
        if self.proj is not None:
            x = x @ self.proj.T

        return x

    def load_from_checkpoint(self, checkpoint_path: str, strict: bool = True):
        """
        Load weights from CLIP checkpoint

        Args:
            checkpoint_path: Path to the CLIP checkpoint file
            strict: Whether to strictly enforce that all keys match
        """
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Try loading as TorchScript
                scripted_model = torch.jit.load(checkpoint_path, map_location='cpu')
                state_dict = scripted_model.state_dict()
            except Exception:
                # Fall back to regular loading
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

        # Filter to only visual encoder weights and remove 'visual.' prefix
        visual_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('visual.'):
                new_k = k.replace('visual.', '')
                visual_state_dict[new_k] = v

        # Load state dict
        missing_keys, unexpected_keys = self.load_state_dict(visual_state_dict, strict=strict)

        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

        return self


def vit_l_14(**kwargs) -> ViT_L_14:
    """
    Create ViT-L/14 model with CLIP-compatible configuration
    """
    default_config = {
        'img_size': 224,
        'patch_size': 14,
        'embed_dim': 1024,
        'num_layers': 24,
        'num_heads': 16,
        'mlp_ratio': 4.0,
    }
    default_config.update(kwargs)
    return ViT_L_14(**default_config)


# Backward compatibility
ViT = ViT_L_14
vit = vit_l_14