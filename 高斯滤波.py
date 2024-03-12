import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('2023-04-13 123047.png')

# 检查图像是否正确读取
if image is None:
    print('Failed to read image')
    sys.exit(1)

# 定义滤波器的大小（核大小）
kernel_size = (5, 5)  # 这里使用 5x5 大小的滤波器

# 应用高斯滤波
blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

# 使用matplotlib显示原始图像和高斯滤波后的图像
plt.figure(figsize=(10, 4))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(122), plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)), plt.title('Blurred Image')
plt.show()
