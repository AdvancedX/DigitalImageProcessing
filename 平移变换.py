import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('2023-04-13 123047.png')

# 检查图像是否正确读取
if image is None:
    print('Failed to read image')
    sys.exit(1)

# 定义平移距离（单位：像素）
dx = 100  # 水平方向平移距离
dy = 100  # 垂直方向平移距离

# 构建平移矩阵
M = np.float32([[1, 0, dx],   # 水平方向平移
                [0, 1, dy]])  # 垂直方向平移

# 进行仿射变换（图像平移）
shifted_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

# 使用matplotlib显示原始图像和平移后的图像
plt.figure(figsize=(10, 4))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(122), plt.imshow(cv2.cvtColor(shifted_image, cv2.COLOR_BGR2RGB)), plt.title('Shifted Image')
plt.show()

# 保存平移后的图像
cv2.imwrite(r'after.jpg',shifted_image)
