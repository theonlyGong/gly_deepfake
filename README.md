### Instructions

这个项目参考于2025年的两篇论文，我只采取了他们的特征提取部分，load了他们的参数作为预训练权重。后续加入了混合的数据集（FF++ & Celeb-DF & GAN-based test data）进行训练简单观测了一下效果。如果感兴趣可以后续使用其他高质量公开数据集进行微调！！！

- 你需要自己创建一个checkpoints文件夹，checkpoints 文件夹是预训练模型（包含FreqNet的特征提取层+ViT-L的特征提取层即可）
  可从以下链接进行下载：通过Baidu网盘分享的文件: https://pan.baidu.com/s/15f5xOzYvCFNql01gUx48nw?pwd=6666 提取码: 6666
- output文件夹内是训练的模型参数，按照时间makefile生成的，里面的best_model.pt文件就是训练的模型参数，可以直接```torch.load()```
  best_model.pt文件链接：通过网盘分享的文件：run_20260413_133506链接: https://pan.baidu.com/s/1NUnq-_EAyTKWPZ44abu1Aw?pwd=7777 提取码: 7777
- ./model里面的 gly_model_v2.py是重写的模型结构class文件，直接实例化这个文件里的class。
- gly_train_v2.py是可以直接跑的训练代码
- gly_single.py和gly_batch_test.py分别是单张预测和文件夹内图片预测的统计脚本。
- freqnet_extraction和vit_extraction.py分别是下面两个论文的特征提取的流程脚本。

### 以上项目参考了：

空间频率域检测方法 (2025) - Deepfake Detection via Spatial-Frequency Domain Analysis
论文链接：[Deepfake Detection via Spatial-Frequency Attention Network | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/11182301)
Github链接：[GitHub - chuangchuangtan/FreqNet-DeepfakeDetection · GitHub](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection)

and

Fatformer (2024)：
论文链接：[Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Forgery-aware_Adaptive_Transformer_for_Generalizable_Synthetic_Image_Detection_CVPR_2024_paper.pdf)
Github链接：[GitHub - Michel-liu/FatFormer: [CVPR 2024\] The official repo for Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection · GitHub](https://github.com/Michel-liu/FatFormer)

初步做了60个epoch的训练，结果如下：

| 数据类型 | 数据量 | Precision | Recall | F1_score |  Acc   |
| :------: | :----: | :-------: | :----: | :------: | :----: |
| 训练数据 | 48113  |  83.46%   | 96.00% |  89.29%  | 82.90% |
| 验证数据 | 12028  |  83.63%   | 97.00% |  90.04%  | 83.80% |

推理时间：10 -16ms （A100显卡），占用GPU显存在16G左右。
