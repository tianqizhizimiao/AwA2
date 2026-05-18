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
        self.fc1 = torch.nn.Linear(400, 100)  # 定义一个全连接层, 400输入100输出
        self.r1 = torch.nn.ReLU()             # 定义一个激活函数
        self.fc2 = torch.nn.Linear(100, 20)   # 定义一个全连接层, 100输入20输出

    def forward(self, x):  # 前向传播 此函数必须定义
        """
        输入：torch.Tensor，形状应为 (batch_size, 400)
        输出：torch.Tensor，形状应为 (batch_size, 20)
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
input_data = [0.0 for n in range(400)]  # 构造一段长度为 400 的列表 即 400维的向量
input_data = torch.tensor(input_data, dtype=torch.float32)  # 将列表转换为torch.tensor张量对象

net.eval()  # 将模型设置为评估模式
output = net(input_data)  # 进行前向传播, 模型会自动调用forward方法

# output 即模型的输出
print(len(list(output)))  # 在此长度应为 20, 即 20 维的向量 
```
