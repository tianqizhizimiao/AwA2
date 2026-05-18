## 索引
1. 模块定义与解析
2. 初始化一个模型
3. 前向传播
4. 训练
5. 手写数字识别项目实践开发

## 1. 模块定义与解析

- 这是一个python的第三方库, 使用py的导入语法规则进行调用
- 开发者能够更简洁高效的进行神经网络架构搭建

## 2. 初始化一个模型
```python
import torch  # 导入

class Net(torch.nn.Module):  # 必须继承torch.nn.Module
    def __init__(self):
        super(Net, self).__init__()

        # 在这里定义网络结构
        self.fc1 = torch.nn.Linear(6, 4)  # 定义一个全连接层, 6输入4输出
        self.r1 = torch.nn.ReLU()             # 定义一个激活函数
        self.fc2 = torch.nn.Linear(4, 2)   # 定义一个全连接层, 4输入2输出

    def forward(self, x):  # 前向传播 此函数必须定义
        """
        输入：torch.Tensor，形状应为 (batch_size, 6)
        输出：torch.Tensor，形状应为 (batch_size, 2)
        """
        y = self.fc1(x)
        x = self.r1(y)
        y = self.fc2(x)
        return(y)
```

## 3. 前向传播
```python
# 初始化刚刚定义的结构
net = Net()

# 初始化一段数据进行输入
input_data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 构造一段长度为 6 的列表 即 6维的向量
input_data = torch.tensor(input_data, dtype=torch.float32)  # 将列表转换为torch.tensor张量对象, 并设置数据类型为32位浮点数

net.eval()  # 将模型设置为评估模式
with torch.no_grad():
    output = net(input_data)  # 进行前向传播, 模型会自动调用forward方法
    
    # output 即模型的输出
    print(len(list(output)))  # 在此长度应为 2, 即 2 维的向量   [y1, y2]
```

## 4. 训练
```python
net = Net()  # 初始化模型

"""
定义一个训练目标
让模型学会根据输入结构输出对应的输出
1. 当输入为 [1.0, 1.0, 1.0, 0.0, 0.0, 0.0] 时 输出 [0.0, 1.0]
2. 当输入为 [0.0, 0.0, 0.0, 1.0, 1.0, 1.0] 时 输出 [1.0, 0.0]
3. 当输入为 [0.0, 1.0, 1.0, 1.0, 1.0, 0.0] 时 输出 [0.5, 0.5]
"""

# 定义训练数据
train_data = [
    [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0]],
    [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0], [1.0, 0.0]],
    [[0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [0.5, 0.5]],
]

class Dataset(torch.utils.data.Dataset):  # 数据集对象
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx][0], dtype=torch.float32),
                torch.tensor(self.data[idx][1], dtype=torch.float32))

loss_fn = torch.nn.MSELoss()  # 初始化损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)  # 初始化优化器, 步长设置为0.01
dataset = Dataset(train_data)

for epoch in range(500):  # 训练100轮
    net.train()  # 设置为训练模式
    for value, labels in dataset:
        optimizer.zero_grad()  # 清零梯度
        output = net(value)  # 前向传播
        loss = loss_fn(output, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新梯度


# 进行验证

net.eval()  # 设置为评估模式
with torch.no_grad():
    input_data = torch.tensor(
        [0.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=torch.float32
    )
    output = net(input_data)
    print(list(output))  # 输出应接近 [0.5, 0.5]
```
