import cv2
import numpy as np
from scipy import special


def RGBGaussianMeans(src, RGBmeans, RGBstds, prob, Kr, areaThresh):

    [Bmeans, Gmeans, Rmeans] = RGBmeans
    [Bstd, Gstd, Rstd] = RGBstds
    mean = np.array([Bmeans, Gmeans, Rmeans])
    std = np.array([Bstd, Gstd, Rstd])
    mask = cv2.inRange(src, mean - (std * special.erfinv(prob / 100.0)), mean + (std * special.erfinv(prob / 100.0)))
    wy, wx, d = np.shape(src)
    erode = cv2.erode(mask, (Kr, Kr), iterations=1)
    blur = cv2.GaussianBlur(erode, (5, 5), 0)
    dilate = cv2.dilate(blur, (Kr, Kr), iterations=1)
    res_img = np.multiply(src, np.reshape((mask > 0), (wy, wx, 1)))

    # draw contour of target
    (contours, hierarchy) = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > areaThresh):
            x, y, w, h = cv2.boundingRect(contour)
            res_img = cv2.drawContours(res_img, contour, -1, (0, 255, 255), 3)
            # img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(res_img, "Area: " + str(area), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))

    return res_img