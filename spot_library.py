import cv2
import math
import numpy as np

#enhance detail of the foreground
def enhance_detail(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    clahe = cv2.createCLAHE(clipLimit = 1.0)

    channels = cv2.split(hsv_img)
    channels[0] = clahe.apply(channels[0])
    channels[1] = clahe.apply(channels[1])
    channels[2] = clahe.apply(channels[2])

    hsv_img = cv2.merge(channels)
    img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    img = cv2.detailEnhance(img, sigma_s=0.2, sigma_r=0.2)

    return img

# clean foreground mask
def morphological_transform(image):
    blur = cv2.GaussianBlur(image, (1,1), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    transformed = cv2.erode(th, kernel, iterations=1)
    transformed = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((5, 5), np.uint8)
    transformed = cv2.morphologyEx(transformed, cv2.MORPH_CLOSE, kernel)
    return transformed


def preprocessing(frame, backSub):
    # extract and enhance foreground
    fg_mask = backSub.apply(frame)
    fg_mask = morphological_transform(fg_mask)
    extracted_fg = cv2.bitwise_and(frame, frame, mask=fg_mask)
    extracted_fg = enhance_detail(extracted_fg)

    # extract and enhance background
    bg_mask = 255 - fg_mask
    
    extracted_bg = cv2.bitwise_and(frame, frame, mask=bg_mask)
    #suppress edges from the background
    extracted_bg = cv2.edgePreservingFilter(extracted_bg, flags=1, sigma_s=50, sigma_r=0.2)

    #combine fg and bg
    image = cv2.add(extracted_fg, extracted_bg)
    return extracted_fg, extracted_bg, image


# get distance between teo 
def get_distance(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2

    d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return d

# function to get the slope of the line, will decide the significance of the direction change
def get_slope(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2

    if (x1 == x2): return 0
    return (y2-y1)/(x2-x1)


# return direction between two points, distance threshold set to 3 pixels at least
def get_direction(prevPoint, currentPoint):
    dX = currentPoint[0] - prevPoint[0]
    dY = prevPoint[1] - currentPoint[1]
    (dirX, dirY) = ("", "")
    if np.abs(dX) > 3:
        #The sign function returns -1 if dX < 0, 0 if dX==0, 1 if dX > 0. nan is returned for nan
        dirX = "E" if np.sign(dX) == 1 else "W"
    if np.abs(dY) > 3:
        dirY = "N" if np.sign(dY) == 1 else "S"
    if dirX != "" and dirY != "":
        direction = "{}{}".format(dirY, dirX)
    else:
        direction = dirX if dirX != "" else dirY
    return direction
