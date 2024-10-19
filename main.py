import numpy as np
import cv2
import matplotlib.pyplot as plt

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    根据指定宽度或高度调整图像大小
    """
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is not None:
        r = width / float(w)
        dim = (width, int(h * r))

    elif height is not None:
        r = height / float(h)
        dim = (int(w * r), height)

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
def order_points(pts):
    """
    对输入的四个点进行排序，返回左上、右上、右下和左下顺序的点。
    :param pts: 输入的四个点的坐标
    :return: 排序后的四个点，依次为左上、右上、右下、左下
    """
    rect = np.zeros((4, 2), dtype="float32")

    # 计算每个点的和，左上角的点具有最小的和，右下角的点具有最大的和
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 左上角
    rect[2] = pts[np.argmax(s)]  # 右下角

    # 计算每个点的差值，右上角的点具有最小的差值，左下角的点具有最大的差值
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 右上角
    rect[3] = pts[np.argmax(diff)]  # 左下角

    return rect

def four_point_transform(image, pts):
    """
    对图像进行四点透视变换
    :param image: 输入图像
    :param pts: 四个顶点的坐标
    :return: 变换后的图像
    """
    pts = order_points(pts.astype("float32"))
    # 获取四个点
    (top_left, top_right, bottom_right, bottom_left) = pts

    # 计算新图像的宽度和高度
    widthA = np.linalg.norm(bottom_right - bottom_left)
    widthB = np.linalg.norm(top_right - top_left)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(top_right - bottom_right)
    heightB = np.linalg.norm(top_left - bottom_left)
    maxHeight = max(int(heightA), int(heightB))

    # 定义目标图像的四个顶点
    dest = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(pts, dest)

    # 执行透视变换
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
# 读取输入图像
image = cv2.imread('./img/demo2.jpg')
orig = image.copy()

# 计算缩放比例
ratio = image.shape[0] / 500.0
# 调整图像大小
image = resize(image, height=500)

# 图像预处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度化
blur = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊
edged = cv2.Canny(blur, 30, 200)  # 边缘检测

# 显示边缘检测结果
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.subplot(1, 2, 2)
plt.title("Edge Image")
plt.imshow(edged, cmap="gray")
plt.axis("off")
plt.show()

# 轮廓检测
cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# 按照轮廓面积降序排序，并只取前5个
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
screenCnt = None

# 遍历轮廓
for i in cnts:
    peri = cv2.arcLength(i, True)  # 计算周长
    approx = cv2.approxPolyDP(i, 0.02 * peri, True)  # 多边形逼近
    
    # 如果逼近的多边形有4个顶点，假定它是矩形
    if len(approx) == 4:
        screenCnt = approx
        break

# 如果找到矩形轮廓
if screenCnt is not None:
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Contour Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
else:
    print("未找到矩形轮廓")
# 透视变换
# 进行四点透视变换，确保点的类型为 float32
print(screenCnt.reshape(4, 2)* ratio)
print("透视变化")
warped = four_point_transform(orig, screenCnt.reshape(4, 2)* ratio)
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
plt.title("Warped Image")
plt.axis("off")
plt.show()
# 二值化
# _, binary = cv2.threshold(warped, 127, 255, cv2.THRESH_BINARY)

# # 显示原始图像和二值化图像

# plt.subplot(1, 2, 2)
# plt.title("Binary Image")
# plt.imshow(binary, cmap='gray')
# plt.axis("off")

# plt.show()