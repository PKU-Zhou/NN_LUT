import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的两层神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(1, 64)  # 第一个全连接层，输入1个特征，输出64个特征
        self.fc2 = nn.Linear(64, 1)  # 第二个全连接层，输入64个特征，输出1个特征

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 使用ReLU激活函数
        x = self.fc2(x)
        return x

# 创建网络实例
net = SimpleNet()

# 检查是否有可用的GPU，如果有则使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 训练数据和标签
x_train = torch.tensor([[[i]] for i in range(1, 100)], dtype=torch.float32).to(device)
y_train = torch.tensor([[[i**0.5]] for i in range(1, 100)], dtype=torch.float32).to(device)

# 训练网络
num_epochs = 5000
for epoch in range(num_epochs):
    # 前向传播
    outputs = net(x_train)
    loss = criterion(outputs, y_train)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新权重

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# 训练完成后保存模型权重
torch.save(net.state_dict(), './weights/sqrt_weights.pth')  # 保存模型权重到文件

# 测试网络
x_test = torch.tensor([[[10]]], dtype=torch.float32).to(device)
y_pred = net(x_test)
print(f'Predicted square root of 10: {y_pred.item():.4f}')