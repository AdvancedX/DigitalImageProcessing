import torch
from torchvision import transforms
from PIL import Image
import os
from model import LR


# 创建逻辑回归模型实例
input_size = 320 * 320 * 3  # 注意与训练时的输入大小一致
num_classes = 10  # 假设有7个类别
model = LR(input_size, num_classes)
model_weights = 'runs/model_epoch_2.ckpt'
# 加载模型权重
model.load_state_dict(torch.load(model_weights))
model.eval()  # 设置为评估模式

# 图像预处理函数
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(image_path)
    image = transform(image)
    return image

# 预测函数
def predict_image(image_path):
    model.eval()
    with torch.no_grad():
        image = preprocess_image(image_path)
        image = image.reshape(1, -1)  # 添加 batch 维度
        output = model(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# 标签名称映射关系
class_mapping = {0: 'bus', 1: 'fuel', 2: 'mpv', 3: 'pickup', 4: 'sedan', 5: 'suv', 6: 'truck'}


# 预测单张图像
# image_path_to_predict = r'E:\VehicleAttrs_V6\train\bus\00000002.jpg'
# predicted_class = predict_image(image_path_to_predict)
# print(f'The predicted class is: {predicted_class}')

# 预测文件夹
folder_path = r'E:\Python\pythonProject\val-1\val\pickup'
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # 只处理图像文件
        image_path = os.path.join(folder_path, filename)
        # 进行预测
        predicted_class = predict_image(image_path)
        # 输出结果
        print(f'File: {filename}, Predicted Class: {class_mapping[predicted_class]}')