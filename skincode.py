import cv2
import numpy as np
from collections import Counter

def pixel_dict(x):
    # detecting skin pixels
    min_YCrCb = np.array([0,133,77],np.uint8)
    max_YCrCb = np.array([235,173,127],np.uint8)

    image = cv2.imread(x)

    imageYCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
    skinYCrCb = cv2.bitwise_and(image, image, mask = skinRegionYCrCb)

    # major colourcode
    image = cv2.cvtColor(skinYCrCb,cv2.COLOR_BGR2RGB)
    flat_pixels = [tuple(pixel) for line in image for pixel in line]
    color_counts = Counter(flat_pixels)
    top_colors = color_counts.most_common(2)
    if top_colors[0][0] == (0,0,0):
        return '#%02x%02x%02x' % top_colors[1][0]
    else:
        return '#%02x%02x%02x' % top_colors[0][0]


print(pixel_dict("image.jpg")) #image location



# cv2.imshow("ycrcb.png", np.hstack([image,skinYCrCb]))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# print(skinYCrCb)