import os
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from Fourier import FourierProcessor


class ChangeDetectionDataset(Dataset):
    """变换检测数据集，用于加载.mat文件并提取傅里叶变换后的图像对"""

    def __init__(self, sample_dir, gt_dir, transform=None, label_transform=None, key_name='mat'):
        """
        初始化数据集。

        参数:
        - sample_dir: .mat 文件目录（包含变换前后的图像对）
        - gt_dir: 掩膜图像目录（变化掩膜）
        - transform: 图像预处理变换
        - label_transform: 掩膜图像的预处理变换
        - key_name: .mat 文件中变量的名称，默认为 'mat'
        """
        self.sample_dir = sample_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.label_transform = label_transform
        self.key_name = key_name

        # 获取所有.mat文件
        self.mat_files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.mat')])

        # 确保文件数量一致
        assert len(self.mat_files) == len(os.listdir(gt_dir)), "样本图像和掩膜图像的数量不一致！"

    def __len__(self):
        """返回数据集大小"""
        return len(self.mat_files)

    def __getitem__(self, idx):
        """
        获取一个样本数据和对应的标签。

        参数:
        - idx: 索引

        返回:
        - 一个字典，包含 'x1', 'x2' 和 'gt'，它们是网络输入和标签
        """
        # 获取.mat文件路径
        mat_file = self.mat_files[idx]
        mat_path = os.path.join(self.sample_dir, mat_file)

        # 加载.mat文件并处理傅里叶变换
        processor = FourierProcessor(mat_path, key_name=self.key_name)
        x1, x2 = processor.process()  # x1是原图，x2是经过反傅里叶变换后的图像

        # 将x1和x2从numpy数组转换为PIL图像
        x1 = Image.fromarray(x1.astype(np.uint8))  # 将numpy数组转换为PIL图像
        x2 = Image.fromarray(x2.astype(np.uint8))  # 将numpy数组转换为PIL图像

        # 获取对应的掩膜图像（假设.mat和.png文件同名）
        gt_file = mat_file.replace('.mat', '.png')  # 假设掩膜图像是和 .mat 文件同名的 .png 图像
        gt_path = os.path.join(self.gt_dir, gt_file)

        # 检查掩膜图像是否存在
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"掩膜图像 {gt_file} 未找到，路径：{gt_path}")

        # 读取掩膜图像并转换为灰度图
        gt = Image.open(gt_path).convert('L')  # 转为灰度图（L模式）

        # 将灰度图像转换为二值化图像
        gt = np.array(gt)  # 转为 numpy 数组
        gt = (gt > 128).astype(np.float32)  # 大于128的像素值为1，其它为0
        gt = Image.fromarray(gt)  # 转回为PIL图像

        # 如果需要，进行图像变换（如调整大小、归一化、转换为tensor等）
        if self.transform:
            x1 = self.transform(x1)
            x2 = self.transform(x2)

        # 对标签进行单独的变换
        if self.label_transform:
            gt = self.label_transform(gt)

        return {'x1': x1, 'x2': x2, 'gt': gt}



# 使用示例：
if __name__ == "__main__":
    # 数据路径设置
    sample_dir = "data/sample"  # 存储 .mat 文件的目录
    gt_dir = "data/gt"  # 存储掩膜图像的目录

    # 设置图像预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转换为单通道
        transforms.Resize((256, 256)),  # 调整大小（根据需要调整）
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到 [-1, 1]
    ])

    # 设置标签变换
    label_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 标签大小也调整一致
        transforms.ToTensor(),  # 转换为 Tensor
    ])

    # 创建数据集实例
    dataset = ChangeDetectionDataset(sample_dir=sample_dir, gt_dir=gt_dir, transform=transform, label_transform=label_transform)

    # 创建DataLoader实例
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 测试数据加载并可视化
    for batch in dataloader:
        x1 = batch['x1']
        x2 = batch['x2']
        gt = batch['gt']

        # 将Tensor转换为NumPy数组以便于可视化
        x1_np = x1.squeeze().numpy()  # 去除批次维度
        x2_np = x2.squeeze().numpy()  # 去除批次维度
        gt_np = gt.squeeze().numpy()  # 去除批次维度

        # 可视化 x1, x2 和 gt
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        # 显示原图 x1
        axes[0].imshow(x1_np, cmap='gray')
        axes[0].set_title('Original Image (x1)')
        axes[0].axis('off')

        # 显示经过反傅里叶变换后的图像 x2
        axes[1].imshow(x2_np, cmap='gray')
        axes[1].set_title('Filtered Image (x2)')
        axes[1].axis('off')

        # 显示掩膜图像 gt
        axes[2].imshow(gt_np, cmap='gray')
        axes[2].set_title('Ground Truth (gt)')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        break  # 只显示一个batch
