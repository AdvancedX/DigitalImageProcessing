import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('2023-04-13 123047.png', cv2.IMREAD_GRAYSCALE)

# 检查图像是否正确读取
if image is None:
    print('Failed to read image')
    sys.exit(1)

# 使用拉普拉斯算子进行边缘检测
edges = cv2.Laplacian(image, cv2.CV_64F)

# 转换为uint8类型，并取绝对值
edges = cv2.convertScaleAbs(edges)

# 使用matplotlib显示原始图像和拉普拉斯算子处理后的图像
plt.figure(figsize=(10, 4))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(edges, cmap='gray'), plt.title('Edges Detected by Laplacian')
plt.show()
