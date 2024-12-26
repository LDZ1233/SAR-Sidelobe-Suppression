import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.io import savemat
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ChangeDetectionDataset  # 导入数据集
from net1 import ChangeDetectionNetwork  # 假设 ChangeDetectionNetwork 类定义在 net1.py 文件中


# 可视化预测图
def visualize_prediction(x1, x2, gt, output, x1_modified, batch_idx):
    """
    可视化图像预测结果
    """
    # 将 Tensor 转换为 numpy 数组，并去掉单通道维度
    x1_np = x1[0].cpu().numpy().squeeze()  # 选择第一个样本并去掉通道维度
    x2_np = x2[0].cpu().numpy().squeeze()  # 选择第一个样本并去掉通道维度
    gt_np = gt[0].cpu().numpy().squeeze()  # 选择第一个样本并去掉通道维度
    output_np = output[0].cpu().detach().numpy().squeeze()  # 选择第一个样本并去掉通道维度
    x1_modified_np = x1_modified[0].cpu().numpy().squeeze()  # 选择第一个样本并去掉通道维度

    # 创建子图展示结果
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # 显示原图 x1
    axes[0].imshow(x1_np, cmap='gray')
    axes[0].set_title('Original Image (x1)')
    axes[0].axis('off')

    # 显示经过反傅里叶变换后的图像 x2
    axes[1].imshow(x2_np, cmap='gray')
    axes[1].set_title('Filtered Image (x2)')
    axes[1].axis('off')

    # 显示真实标签图像 gt
    axes[2].imshow(gt_np, cmap='gray')
    axes[2].set_title('Ground Truth (gt)')
    axes[2].axis('off')

    # 显示预测输出图像
    axes[3].imshow(output_np, cmap='gray')
    axes[3].set_title('Model Prediction')
    axes[3].axis('off')

    # 显示修改后的 x1
    axes[4].imshow(x1_modified_np, cmap='gray')
    axes[4].set_title('Modified Image (x1_modified)')
    axes[4].axis('off')

    # 保存图像
    plt.tight_layout()
    plt.savefig(f"prediction_batch{batch_idx}.png")
    plt.show()


# 预测过程
def predict(model, dataloader, device, output_dir="predictions"):
    model.eval()  # 设置为评估模式
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录，如果不存在的话
    with torch.no_grad():  # 关闭梯度计算
        for batch_idx, batch in enumerate(dataloader):
            x1 = batch['x1'].to(device)
            x2 = batch['x2'].to(device)
            gt = batch['gt'].to(device)

            # 前向传播，得到模型的输出
            output = model(x1, x2)

            # 二值化输出图像（通常0.5是合适的阈值）
            output_bin = (output > 0.5).float()  # 0.5作为阈值，超过则为1，不超过为0

            # 这里是替换操作：当output_bin为1时，x1中的值将被x2中的值替换
            x1_modified = torch.where(output_bin == 1, x2, x1)

            # 可视化预测结果
            visualize_prediction(x1, x2, gt, output, x1_modified, batch_idx)

            # 保存为 .mat 格式
            output_np = output_bin[0].cpu().numpy().squeeze()  # 选择第一个样本并去掉通道维度
            gt_np = gt[0].cpu().numpy().squeeze()  # 选择第一个样本并去掉通道维度

            # 保存预测掩膜和真实标签为.mat文件
            mat_dict = {
                'pred_mask': output_np,
                'gt_mask': gt_np,
                'x1_modified': x1_modified[0].cpu().numpy().squeeze()  # 保存修改后的 x1
            }
            mat_filename = os.path.join(output_dir, f"prediction_batch_{batch_idx}.mat")
            savemat(mat_filename, mat_dict)  # 保存为 .mat 文件
            print(f"Saved prediction and ground truth to {mat_filename}")


# 主函数
def main():
    # 设置路径
    sample_dir = "data/test/sample"  # 你的样本目录
    gt_dir = "data/test/gt"  # 你的掩膜目录
    model_path = "model_epoch_150.pth"  # 加载的训练好的模型路径

    # 超参数
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转换为单通道
        transforms.Resize((256, 256)),  # 调整大小（根据需要调整）
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到 [-1, 1]
    ])

    label_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 标签大小也调整一致
        transforms.ToTensor(),  # 转换为 Tensor
    ])

    # 创建数据集和数据加载器
    train_dataset = ChangeDetectionDataset(sample_dir=sample_dir, gt_dir=gt_dir, transform=transform,
                                           label_transform=label_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # 加载模型
    model = ChangeDetectionNetwork(input_channels=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 切换为评估模式

    # 进行预测并保存结果
    predict(model, train_loader, device, output_dir="predictions")


if __name__ == "__main__":
    main()
