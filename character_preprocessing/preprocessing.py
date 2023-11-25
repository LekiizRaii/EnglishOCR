import cv2


# Đọc hình ảnh từ máy tính của bạn
uploaded = files.upload()

# Lấy tên của tệp hình ảnh đã tải lên
image_file = list(uploaded.keys())[0]

# Đọc hình ảnh bằng OpenCV
image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)  # Chuyển thành hình ảnh xám

# Áp dụng phương pháp nhị phân hóa Otsu
_, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Invert màu sắc (đảo ngược)
inverted_image = cv2.bitwise_not(binary_image)

# Hiển thị hình ảnh ban đầu và hình ảnh đã binarization
from matplotlib import pyplot as plt
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(122), plt.imshow(inverted_image, cmap='gray')
plt.title('Otsu\'s Binary Image')
plt.show()
img = inverted_image

#skeletonization
size = np.size(inverted_image)
skel = np.zeros(inverted_image.shape, np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

while True:
    open_image = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, element)
    temp = cv2.subtract(inverted_image, open_image)
    eroded = cv2.erode(inverted_image, element)
    skel = cv2.bitwise_or(skel, temp)
    inverted_image = eroded.copy()

    if cv2.countNonZero(inverted_image) == 0:
        break

# Hiển thị hình ảnh ban đầu và hình ảnh đã được skeletonized
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Otsu\'s Binary Image')
plt.axis('off')
plt.subplot(122), plt.imshow(skel, cmap='gray')
plt.title('Skeletonized Image')
plt.axis('off')
plt.show()