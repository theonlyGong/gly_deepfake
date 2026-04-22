import torch

# 检查output中保存的权重格式
ckpt = torch.load('F:/Deepfake_detect/gly_deepfake/output/run_20260413_133506/checkpoints/best_model.pth', map_location='cpu', weights_only=False)
print('Keys in checkpoint:', ckpt.keys())
print()
if 'model_state_dict' in ckpt:
    print('State dict keys (first 30):')
    for k in list(ckpt['model_state_dict'].keys())[:30]:
        print(f'  {k}')
    print(f'\nTotal params: {len(ckpt["model_state_dict"].keys())}')