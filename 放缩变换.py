import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('2023-04-13 123047.png')

# 检查图像是否正确读取
if image is None:
    print('Failed to read image')
    sys.exit(1)

# 定义缩放比例
scale_percent = 10  # 缩放比例为原图的50%

# 计算缩放后的图像尺寸
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# 进行图像缩放
resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_LINEAR)

# 使用matplotlib显示原始图像和缩放后的图像
plt.figure(figsize=(10, 4))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(122), plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)), plt.title('Resized Image')
plt.show()
