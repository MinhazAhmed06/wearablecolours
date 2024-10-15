import cv2
import numpy as np
from collections import Counter

# detecting skin pixels
min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)

image = cv2.imread("image.jpg")
imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)

skinYCrCb = cv2.bitwise_and(image, image, mask = skinRegionYCrCb)

# major colourcode
def pixel_dict(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    flat_pixels = [tuple(pixel) for line in image for pixel in line]
    color_counts = Counter(flat_pixels)
    top_colors = color_counts.most_common(5)
    return top_colors

print(pixel_dict(skinYCrCb))



cv2.imshow("ycrcb.png", np.hstack([image,skinYCrCb]))
cv2.waitKey(0)
cv2.destroyAllWindows()

print(skinYCrCb)