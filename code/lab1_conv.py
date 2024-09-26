import numpy as np

# 定義卷積層
class Conv2D:
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # 初始化權重和偏差
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size)
        self.bias = np.zeros((output_channels, 1))

    def forward(self, input_data):
        batch_size, input_channels, input_height, input_width = input_data.shape
        output_height = int((input_height - self.kernel_size + 2 * self.padding) / self.stride) + 1
        output_width = int((input_width - self.kernel_size + 2 * self.padding) / self.stride) + 1

        # 填充輸入數據
        padded_input = np.pad(input_data, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # 初始化輸出
        output_data = np.zeros((batch_size, self.output_channels, output_height, output_width))

        # 卷積運算
        for b in range(batch_size):
            for c_out in range(self.output_channels):
                for i in range(0, output_height, self.stride):
                    for j in range(0, output_width, self.stride):
                        input_slice = padded_input[b, :, i:i+self.kernel_size, j:j+self.kernel_size]
                        output_data[b, c_out, i, j] = np.sum(input_slice * self.weights[c_out]) + self.bias[c_out]

        return output_data

# 定義ReLU激活函數
class ReLU:
    def forward(self, input_data):
        return np.maximum(0, input_data)

# 定義模型結構
class SimpleCNN:
    def __init__(self):
        self.conv1 = Conv2D(input_channels=1, output_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(input_channels=16, output_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()

    def forward(self, input_data):
        x = self.conv1.forward(input_data)
        x = self.relu1.forward(x)
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        return x

# 測試
if __name__ == "__main__":
    # 假設輸入數據是28x28的RGB圖像，並且有10個
    input_data = np.random.randn(10, 3, 28, 28)

    # 建立模型
    model = SimpleCNN()

    # 執行前向傳播
    output_data = model.forward(input_data)

    # 輸出結果
    print("輸入數據形狀:", input_data.shape)
    print("輸出數據形狀:", output_data.shape)
