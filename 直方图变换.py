import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread(r'2023-04-13 123047.png', cv2.IMREAD_GRAYSCALE)

# 检查图像是否正确读取
if image is None:
    print('Failed to read image')
    sys.exit(1)

# 应用直方图均衡化
equalized_image = cv2.equalizeHist(image)

# 使用matplotlib显示原始图像和直方图均衡化后的图像
plt.figure(figsize=(10, 4))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(equalized_image, cmap='gray'), plt.title('Equalized Image')
plt.show()
