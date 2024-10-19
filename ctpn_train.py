import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

# 1. 自定义数据集类
class CTPNDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_names = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".jpg", ".txt"))

        # 加载图像
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 加载标签（你需要根据CTPN标签格式解析）
        with open(label_path, 'r') as f:
            label = f.read().strip()
            # 处理标签以适应模型，标签的格式可能需要自定义解析

        return image, label

# 2. CTPN模型定义
class CTPN(nn.Module):
    def __init__(self):
        super(CTPN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 输入为3通道图像
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 降采样

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 再次降采样

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 降采样
        )

        # RNN部分用于预测文本框的边界信息
        self.rnn = nn.LSTM(256, 128, bidirectional=True, batch_first=True)

        # 输出层，预测文本框的回归值和类别
        self.fc = nn.Linear(128 * 2, 5)  # 输出5个值：包括文本框坐标和类别

    def forward(self, x):
        # CNN前向传播
        x = self.cnn(x)

        # 将特征序列化为RNN输入
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, -1)  # 调整为 [batch_size, width, features]

        # RNN前向传播
        x, _ = self.rnn(x)

        # 全连接层映射为输出
        x = self.fc(x)

        return x

# 3. 训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)  # 标签处理后需要是适应模型格式的张量

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}")

# 4. 评估函数
def evaluate_model(model, dataloader):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # 评估模型性能，例如准确率等
            # 这里需要根据标签和预测结果计算性能指标

# 5. 主函数
if __name__ == "__main__":
    # 超参数设置
    num_epochs = 10
    batch_size = 16
    learning_rate = 0.001
    # 检查是否有可用的GPU，如果没有则使用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("TRAIN CTPN FOR:",device)

    # 数据转换
    transform = transforms.Compose([
        transforms.Resize((512, 512))
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载训练集和测试集
    train_dataset = CTPNDataset("./data_sets/ctpn/train/img", "./data_sets/ctpn/train/label", transform)
    test_dataset = CTPNDataset("./data_sets/ctpn/test/img", "./data_sets/ctpn/test/label", transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型、损失函数和优化器
    model = CTPN().to(device)
    criterion = nn.CrossEntropyLoss()  # 你需要根据标签格式选择合适的损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, num_epochs)

    # 评估模型
    evaluate_model(model, test_loader)
