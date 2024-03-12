import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('2023-04-13 123047.png', cv2.IMREAD_GRAYSCALE)

# 检查图像是否正确读取
if image is None:
    print('Failed to read image')
    sys.exit(1)

# 使用Canny算子进行边缘检测
edges = cv2.Canny(image, 100, 200)  # 参数分别为低阈值和高阈值

# 使用matplotlib显示原始图像和Canny算子处理后的图像
plt.figure(figsize=(10, 4))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(edges, cmap='gray'), plt.title('Edges Detected by Canny')
plt.show()
