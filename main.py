import cv2
import numpy as np
from matplotlib import pyplot as plt


def read_data():
    left = "tsukuba-imL.png"
    right = "tsukuba-imR.png"
    img_left = cv2.cvtColor(cv2.imread(left), cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(cv2.imread(right), cv2.COLOR_BGR2GRAY)
    return left, right


if __name__ == "__main__":

    imgL = cv2.imread('tsukuba_l.png',0)
    imgR = cv2.imread('tsukuba_r.png',0)

    stereo = cv2.createStereoBM(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgL,imgR)
    plt.imshow(disparity,'gray')
    plt.show()
