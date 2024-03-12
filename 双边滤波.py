import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('2023-04-13 123047.png')

# 检查图像是否正确读取
if image is None:
    print('Failed to read image')
    sys.exit(1)

# 定义滤波器的大小和参数
diameter = 15  # 滤波器直径
sigma_color = 75  # 颜色空间的标准差
sigma_space = 75  # 坐标空间的标准差

# 应用双边滤波
filtered_image = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)

# 使用matplotlib显示原始图像和双边滤波后的图像
plt.figure(figsize=(10, 4))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(122), plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)), plt.title('Filtered Image')
plt.show()
