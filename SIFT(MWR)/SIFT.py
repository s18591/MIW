import argparse
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

def main(args):
    img1 = cv2.imread(args.first_image)
    img2 = cv2.imread(args.second_image)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # sift
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
    plt.imshow(img3), plt.show()

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--first_image', type=str, required=True)
    parser.add_argument('-s', '--second_image', type=str, required=True)
    return parser.parse_args()



if __name__ == '__main__':
    main(parse_argument())