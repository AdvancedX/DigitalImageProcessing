import numpy as np
import cv2
import matplotlib.pyplot as plt


# 读取图像
image = cv2.imread(r'2023-04-13 123047.png')
# 将图像转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 对数变换
c = 1  # 可调整的参数
log_transformed = np.uint8(255 * np.log(1 + gray_image.astype(float)) / np.log(256))
# 显示原始灰度图和对数变换后的图像
plt.figure(figsize=(10, 4))
plt.subplot(121), plt.imshow(gray_image, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(log_transformed, cmap='gray'), plt.title('Edges Detected by log')
plt.show()
