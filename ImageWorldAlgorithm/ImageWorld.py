"""
@Author: P_k_y
@Time: 2021/1/19
"""

import cv2


class ImageWorld:

    def __init__(self, map_image_file_name, width=500, height=500):
        try:
            img = cv2.imread(map_image_file_name)
        except IOError:
            raise IOError("[ERROR] Can't Open File %s!" % map_image_file_name)
        img = cv2.resize(img, (width, height))
        self.map_origin = img
        image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
        self.map_array = img_binary
