import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    img0 = cv2.imread('2023-04-13 123047.png',0)

    # 检查图像是否正确读取
    if img0 is None:
        print('Failed to read image')
        sys.exit(1)

    img = np.array(img0, dtype=np.float32)
    mean = np.mean(img)
    img = img - mean
    img = img*2 + mean*1.5 #修对比度和亮度
    img = img/255.

    # 使用matplotlib显示原始图像和处理后的图像
    plt.figure(figsize=(10, 4))
    plt.subplot(121), plt.imshow(img0, cmap='gray'), plt.title('Original Image')
    plt.subplot(122), plt.imshow(img, cmap='gray'), plt.title('Processed Image')
    plt.show()
