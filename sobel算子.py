import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('2023-04-13 123047.png', cv2.IMREAD_GRAYSCALE)

# 检查图像是否正确读取
if image is None:
    print('Failed to read image')
    sys.exit(1)

# 使用Sobel算子计算图像的水平方向和垂直方向的梯度
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向梯度
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向梯度

# 计算梯度的绝对值，并将结果转换为8位灰度图像
sobel_x = np.uint8(np.absolute(sobel_x))
sobel_y = np.uint8(np.absolute(sobel_y))

# 合并水平和垂直方向的梯度
sobel_combined = cv2.bitwise_or(sobel_x, sobel_y)

# 使用matplotlib显示原始图像和Sobel算子处理后的图像
plt.figure(figsize=(10, 8))
plt.subplot(221), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(222), plt.imshow(sobel_x, cmap='gray'), plt.title('Sobel X')
plt.subplot(223), plt.imshow(sobel_y, cmap='gray'), plt.title('Sobel Y')
plt.subplot(224), plt.imshow(sobel_combined, cmap='gray'), plt.title('Sobel Combined')
plt.tight_layout()
plt.show()
