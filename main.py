import numpy as np
import matplotlib.pyplot as plt


# 搭建我的神经网络
class MyNeuralNetwork:
    def __init__(self, inodes, hnodes, onodes):
        # 输入层节点数
        self.inodes = inodes
        # 隐藏层节点数
        self.hnodes = hnodes
        # 输出层节点数
        self.onodes = onodes
        # 学习率
        self.lr = 0.05
        # 权重初始化
   
        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5
    
    def sigmoid(self, x):
        """
        sigmoid激活函数
        """
        return 1 / (1 + np.exp(-x))
    def sigmoid_derivative(self, x):
        """
        sigmoid导数
        """
        return x * (1 - x)
    
    def forward(self, inputs):
        """
        return hidden layer output and final output
        """
        self.hidden_inputs = np.dot(self.wih, inputs)
        self.hidden_outputs = self.sigmoid(self.hidden_inputs)
        self.final_inputs = np.dot(self.who, self.hidden_outputs)
        self.final_outputs = self.sigmoid(self.final_inputs)
        return self.hidden_outputs, self.final_outputs
    
    def compute_loss(self, inputs, targets):
        _, fo = self.forward(inputs)
        return np.mean((targets - fo) ** 2)
    
    def backward(self, inputs, targets, hidden_outputs, final_outputs):
        """
        反向传播
        """
        # 计算输出层误差
        output_errors = targets - final_outputs
        # 计算隐藏层误差
        hidden_errors = np.dot(self.who.T, output_errors)
        
        # 更新权重
        self.who += self.lr * np.dot(output_errors * self.sigmoid_derivative(self.final_outputs), self.hidden_outputs.T)
        self.wih += self.lr * np.dot(hidden_errors * self.sigmoid_derivative(self.hidden_outputs), inputs.T)
    
    def train(self, inputs, targets):
        """
        训练神经网络
        """
        # 前向传播
        hidden_outputs, final_outputs = self.forward(inputs)
        # 反向传播
        self.backward(inputs, targets, hidden_outputs, final_outputs)
        
    def predict(self, inputs):
        """
        预测
        """
        _, final_outputs = self.forward(inputs)
        return np.argmax(final_outputs)
    
    def fit(self):
        # 1000 张图片
        train_size = 2500
        inputs = np.zeros((train_size, self.inodes, 1))
        for i in range(train_size):
            # 读取图片
            img = plt.imread(f'./train/{i}.png').reshape(self.inodes, 1) / 255.0
            inputs[i] = img
            if i % 100 == 0:
                print(f'Processing training images: {i}')
        
        # targets
        targets = np.zeros((train_size, self.onodes , 1))
        labels = np.loadtxt('./train_labs.txt', dtype=int)
        for i in range(train_size):
            label = labels[i,1]
            targets[i, label] = 1.0
        
        # 训练神经网络
        # print(targets[0].shape, inputs[0].shape)
        for epoch in range(10):
            for i in range(train_size):
                self.train(inputs[i], targets[i])
                if i % 100 == 0:
                    loss = self.compute_loss(inputs[i], targets[i])
                    print(f'Epoch {epoch}, Image {i}, Loss: {loss:.4f}')

    def test(self):
        """
        测试神经网络
        """
        # 1000 张图片
        test_size = 1000
        inputs = np.zeros((test_size, self.inodes, 1))
        for i in range(test_size):
            # 读取图片
            img = plt.imread(f'./test/{i}.png').reshape(self.inodes, 1) / 255.0
            inputs[i] = img
            if i % 100 == 0:
                print(f'Processing test images: {i}')
            
        # targets
        targets = np.zeros((test_size, self.onodes , 1))
        labels = np.loadtxt('./test_labs.txt', dtype=int)
        for i in range(test_size):
            label = labels[i,1]
            targets[i, label] = 1.0
        
        # 测试神经网络
        correct = 0
        for i in range(test_size):
            pred = self.predict(inputs[i])
            if pred == np.argmax(targets[i]):
                correct += 1
            if i % 100 == 0:
                print(f'Testing image {i}, Predicted: {pred}, Actual: {np.argmax(targets[i])}')
        
        print(f'Test accuracy: {correct / test_size:.2%}')


# 测试神经网络
# 创建一个神经网络实例
nn = MyNeuralNetwork(inodes=784, hnodes=334, onodes=10)

img = plt.imread('./train/0.png').reshape(784, 1) / 255.0
print(nn.predict(img))

nn.fit()
print(nn.predict(img))