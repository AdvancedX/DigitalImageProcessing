import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('2023-04-13 123047.png')

# 检查图像是否正确读取
if image is None:
    print('Failed to read image')
    sys.exit(1)

# 定义旋转角度（单位：度）
angle = 45

# 计算图像中心点坐标
(h, w) = image.shape[:2]
center = (w // 2, h // 2)

# 构建旋转矩阵
M = cv2.getRotationMatrix2D(center, angle, 1.0)

# 进行仿射变换（图像旋转）
rotated_image = cv2.warpAffine(image, M, (w, h))

# 使用matplotlib显示原始图像和旋转后的图像
plt.figure(figsize=(10, 4))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(122), plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB)), plt.title('Rotated Image')
plt.show()
