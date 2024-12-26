import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import ChangeDetectionDataset  # 导入数据集
from net1 import ChangeDetectionNetwork  # 假设 ChangeDetectionNetwork 类定义在 net1.py 文件中
from torchvision import transforms


# 训练过程
def train(model, dataloader, criterion, optimizer, device):
    model.train()  # 设为训练模式
    running_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        x1 = batch['x1'].to(device)
        x2 = batch['x2'].to(device)
        gt = batch['gt'].to(device)

        optimizer.zero_grad()

        # 前向传播
        output = model(x1, x2)

        # 计算损失
        loss = criterion(output, gt)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f"Training Loss: {epoch_loss:.4f}")
    return epoch_loss


# 主训练函数
def main():
    # 设置路径
    sample_dir = "data/sample"  # 你的样本目录
    gt_dir = "data/gt"  # 你的掩膜目录

    # 超参数
    batch_size = 4
    num_epochs = 300
    learning_rate = 1e-3
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
    train_dataset = ChangeDetectionDataset(sample_dir=sample_dir, gt_dir=gt_dir, transform=transform, label_transform=label_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 模型、损失函数和优化器
    model = ChangeDetectionNetwork(input_channels=1).to(device)
    criterion = nn.BCELoss()  # 使用二进制交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        # 训练
        train_loss = train(model, train_loader, criterion, optimizer, device)

        # 每5个epoch保存一次模型
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")


if __name__ == "__main__":
    main()
