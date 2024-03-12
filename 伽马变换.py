import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread(r'2023-04-13 123047.png')

# 检查图像是否正确读取
if image is None:
    print('Failed to read image')
    sys.exit(1)

# 将图像转换为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 伽马值
gamma = 0.5  # 可调整的参数

# 伽马变换
gamma_transformed = np.uint8(255 * np.power(gray_image / 255.0, gamma))

# 使用matplotlib显示原始灰度图和伽马变换后的图像
plt.figure(figsize=(10, 4))
plt.subplot(121), plt.imshow(gray_image, cmap='gray'), plt.title('Original Gray Image')
plt.subplot(122), plt.imshow(gamma_transformed, cmap='gray'), plt.title('Gamma Transformed Gray Image')
plt.show()
