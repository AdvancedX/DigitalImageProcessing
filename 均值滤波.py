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
kernel_size = (3, 3)  # 这里使用 3*3 大小的滤波器

# 应用均值滤波
blurred_image = cv2.blur(image, kernel_size)

# 使用matplotlib显示原始图像和均值滤波后的图像
plt.figure(figsize=(10, 4))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(122), plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB)), plt.title('Blurred Image')
plt.show()
