
import torch
import torchvision.transforms as transforms
from PIL import Image
from modle import FewShot
import cv2
import numpy as np
import matplotlib.pyplot as plt
# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建一个空模型
model = FewShot()

# 加载模型的状态字典
model_state_dict = torch.load('model_state_dict_9.pth', map_location=device)

# 将状态字典加载到模型中
model.load_state_dict(model_state_dict)

# 将模型移动到指定的设备
model = model.to(device)

# 设置模型为评估模式
model.eval()

print("模型已加载")

# 加载并预处理手写数字图像
image_path = 'myhand/72f67a21041872127638177e1e06c7f.png'  # 替换为您自己的图像路径

# 读取输入图像
input_image = cv2.imread(image_path)

# 将输入图像转换为灰度图像
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# 创建转换操作
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 将灰度图像应用转换操作
image = transform(gray_image)

# 添加批次维度
image = image.unsqueeze(0)

# 将图像移动到设备上
image = image.to(device)

# 使用模型进行推理
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output.data, 1)

# 获取预测结果
prediction = predicted.item()
print('预测结果:', prediction)