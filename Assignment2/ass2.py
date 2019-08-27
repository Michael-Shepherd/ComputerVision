import matplotlib.pyplot as plt
from skimage import data, filters
import scipy.ndimage as ndimage
import numpy as np
from PIL import Image
from math import floor, ceil
import cv2
import os
import math
import pandas as pd
import random
import signal


class Timeout:
    """Timeout class using ALARM signal."""

    class Timeout(Exception):
        pass

    def __init__(self, sec):
        self.sec = sec

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        signal.alarm(0)  # disable alarm

    def raise_timeout(self, *args):
        raise Timeout.Timeout()


def H_matrix(theta, s, x0, y0, t0=None, t1=None):
    return np.array(
        [
            [
                s * math.cos(theta),
                -s * math.sin(theta),
                -(math.cos(theta) * x0 + math.sin(theta) * y0 - x0),
            ],
            [
                s * math.sin(theta),
                s * math.cos(theta),
                -(math.sin(theta) * x0 + math.cos(theta) * y0 - x0),
            ],
            [0, 0, 1],
        ]
    )


def H_matrix_t(theta, s, x0, y0, t0, t1):
    return np.array(
        [
            [s * math.cos(theta), -s * math.sin(theta), t1],
            [s * math.sin(theta), s * math.cos(theta), t1],
            [0, 0, 1],
        ]
    )


def apply_homography(a, h):
    r, c, col = np.shape(a)
    # Determine size of output image by forward-transforming the four corners of A
    p1_h, p2_h, p3_h, p4_h = (
        np.array([[0], [0], [1]], dtype=np.float),
        np.array([[c - 1], [0], [1]], dtype=np.float),
        np.array([[0], [r - 1], [1]], dtype=np.float),
        np.array([[c - 1], [r - 1], [1]], dtype=np.float),
    )

    # Apply homography on each point
    p1, p2, p3, p4 = np.dot(h, p1_h), np.dot(h, p2_h), np.dot(h, p3_h), np.dot(h, p4_h)
    p1, p2, p3, p4 = p1 / p1[2, 0], p2 / p2[2, 0], p3 / p3[2, 0], p4 / p4[2, 0]

    # Get new min and max from each axis
    min_x = floor(min([p1[0, 0], p2[0, 0], p3[0, 0], p4[0, 0]]))
    max_x = ceil(max([p1[0, 0], p2[0, 0], p3[0, 0], p4[0, 0]]))
    min_y = floor(min([p1[1, 0], p2[1, 0], p3[1, 0], p4[1, 0]]))
    max_y = ceil(max([p1[1, 0], p2[1, 0], p3[1, 0], p4[1, 0]]))

    # Obtain new row and column
    nr, nc = max_y - min_y + 1, max_x - min_x + 1
    b = np.zeros([int(nr), int(nc), int(col)]) + 255
    h_inv = np.linalg.inv(h)
    for y in range(int(nr)):
        for x in range(int(nc)):
            p = np.array([[x + min_x], [y + min_y], [1]])
            pp = np.dot(h_inv, p)

            xp, yp = pp[0, 0] / pp[2, 0], pp[1, 0] / pp[2, 0]

            xpf, ypf = int(floor(xp)), int(floor(yp))
            xpc, ypc = xpf + 1, ypf + 1

            if (xpf >= 0) and (xpc < c) and (ypf >= 0) and (ypc < r):
                b[y, x, :] = (
                    (xpc - xp) * (ypc - yp) * a[ypf, xpf, :]
                    + (xpc - xp) * (yp - ypf) * a[ypc, xpf, :]
                    + (xp - xpf) * (ypc - yp) * a[ypf, xpc, :]
                    + (xp - xpf) * (yp - ypf) * a[ypc, xpc, :]
                )

    return b.astype(np.uint8)


def one():
    A = plt.imread("arc.jpg")
    plt.imshow(A)
    plt.show()

    theta = np.pi / 3
    s = 1.5
    H = H_matrix(theta, s, 0, 0)
    B = apply_homography(A, H)
    plt.imshow(B)
    plt.show()

    theta = 2 * np.pi / 3
    s = 0.5
    H = H_matrix(theta, s, 0, 0)
    B = apply_homography(A, H)
    plt.imshow(B)
    plt.show()

    theta = 0.2 * np.pi
    s = 0.7
    H = H_matrix(theta, s, 0, 0)
    B = apply_homography(A, H)
    plt.imshow(B)
    plt.show()


def find_homography(pts_src, pts_dest):
    """
    Find Homography from src to dest
    """
    x1 = pts_src[0][0]
    x2 = pts_src[1][0]
    x3 = pts_src[2][0]
    x4 = pts_src[3][0]
    y1 = pts_src[0][1]
    y2 = pts_src[1][1]
    y3 = pts_src[2][1]
    y4 = pts_src[3][1]
    x_1 = pts_dest[0][0]
    x_2 = pts_dest[1][0]
    x_3 = pts_dest[2][0]
    x_4 = pts_dest[3][0]
    y_1 = pts_dest[0][1]
    y_2 = pts_dest[1][1]
    y_3 = pts_dest[2][1]
    y_4 = pts_dest[3][1]

    A = np.array(
        [
            [x1, y1, 1, 0, 0, 0, -x_1 * x1, -x_1 * y1, -x_1],
            [0, 0, 0, x1, y1, 1, -y_1 * x1, -y_1 * y1, -y_1],
            [x2, y2, 1, 0, 0, 0, -x_2 * x2, -x_2 * y2, -x_2],
            [0, 0, 0, x2, y2, 1, -y_2 * x2, -y_2 * y2, -y_2],
            [x3, y3, 1, 0, 0, 0, -x_3 * x3, -x_3 * y3, -x_3],
            [0, 0, 0, x3, y3, 1, -y_3 * x3, -y_3 * y3, -y_3],
            [x4, y4, 1, 0, 0, 0, -x_4 * x4, -x_4 * y4, -x_4],
            [0, 0, 0, x4, y4, 1, -y_4 * x4, -y_4 * y4, -y_4],
        ]
    )

    _, _, V = np.linalg.svd(A)
    h = V.T[:, V.shape[1] - 1]

    H = np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]])
    return H


def apply_homography_1(a, h, target):
    r, c, col = np.shape(a)
    # Determine size of output image by forward-transforming the four corners of A
    p1_h, p2_h, p3_h, p4_h = (
        np.array([[0], [0], [1]], dtype=np.float),
        np.array([[c - 1], [0], [1]], dtype=np.float),
        np.array([[0], [r - 1], [1]], dtype=np.float),
        np.array([[c - 1], [r - 1], [1]], dtype=np.float),
    )

    # Apply homography on each point
    p1, p2, p3, p4 = np.dot(h, p1_h), np.dot(h, p2_h), np.dot(h, p3_h), np.dot(h, p4_h)
    p1, p2, p3, p4 = p1 / p1[2, 0], p2 / p2[2, 0], p3 / p3[2, 0], p4 / p4[2, 0]

    # Get new min and max from each axis
    min_x = floor(min([p1[0, 0], p2[0, 0], p3[0, 0], p4[0, 0]]))
    max_x = ceil(max([p1[0, 0], p2[0, 0], p3[0, 0], p4[0, 0]]))
    min_y = floor(min([p1[1, 0], p2[1, 0], p3[1, 0], p4[1, 0]]))
    max_y = ceil(max([p1[1, 0], p2[1, 0], p3[1, 0], p4[1, 0]]))

    # Obtain new row and column
    nr, nc = max_y - min_y + 1, max_x - min_x + 1

    # Initialize new array with white background
    b = np.zeros([int(nr), int(nc), int(col)]) + 255

    # Pre-compute inverse of H
    h_inv = np.linalg.inv(h)

    # Define function on coordinates
    for y in range(int(nr)):
        for x in range(int(nc)):
            p = np.array([[x + min_x], [y + min_y], [1]])
            pp = np.dot(h_inv, p)

            # De-homogenize
            xp, yp = pp[0, 0] / pp[2, 0], pp[1, 0] / pp[2, 0]

            # Interpolate
            xpf, ypf = int(floor(xp)), int(floor(yp))
            xpc, ypc = xpf + 1, ypf + 1

            # Placing the pixels in new image
            if (xpf >= 0) and (xpc < c) and (ypf >= 0) and (ypc < r):
                b[y, x, :] = (
                    (xpc - xp) * (ypc - yp) * a[ypf, xpf, :]
                    + (xpc - xp) * (yp - ypf) * a[ypc, xpf, :]
                    + (xp - xpf) * (ypc - yp) * a[ypf, xpc, :]
                    + (xp - xpf) * (yp - ypf) * a[ypc, xpc, :]
                )
    x_min = int(min_x)
    y_min = int(min_y)
    tmp = b.astype(np.uint8)
    base = np.zeros_like(target)
    mask = tmp[:, :, :] == [255, 255, 255]
    for r in range(0, mask.shape[0]):
        for c in range(0, mask.shape[1]):
            if mask[r, c, 1] == False:
                base[r + y_min][c + x_min] = tmp[r, c, :]
    mask = base == [0, 0, 0]
    un_mask = np.invert(mask)
    base[mask] = target[mask]
    return base


def two():
    poster_small = cv2.resize(plt.imread("poster.jpg"), None, fx=0.2, fy=0.2)
    poster_medium = cv2.resize(plt.imread("poster.jpg"), None, fx=2, fy=2)
    poster_large = cv2.resize(plt.imread("poster.jpg"), None, fx=10, fy=10)
    building = plt.imread("griest.jpg")
    plt.imshow(poster_medium)
    plt.show()
    plt.imshow(building)
    plt.show()

    building_points = np.asarray([[104, 247], [315, 120], [32, 622], [313, 564]])
    # Small homograhpy
    points_small = np.asarray(
        [
            [0, 0],
            [poster_small.shape[1] - 1, 0],
            [0, poster_small.shape[0] - 1],
            [poster_small.shape[1] - 1, poster_small.shape[0] - 1],
        ]
    )
    H_small = find_homography(points_small, building_points)
    # Medium homograhpy
    points_medium = np.asarray(
        [
            [0, 0],
            [poster_medium.shape[1] - 1, 0],
            [0, poster_medium.shape[0] - 1],
            [poster_medium.shape[1] - 1, poster_medium.shape[0] - 1],
        ]
    )
    H_medium = find_homography(points_medium, building_points)
    # Large homograhpy
    points_large = np.asarray(
        [
            [0, 0],
            [poster_large.shape[1] - 1, 0],
            [0, poster_large.shape[0] - 1],
            [poster_large.shape[1] - 1, poster_large.shape[0] - 1],
        ]
    )
    H_large = find_homography(points_large, building_points)
    small = apply_homography_1(poster_small, H_small, building)
    medium = apply_homography_1(poster_medium, H_medium, building)
    large = apply_homography_1(poster_large, H_large, building)


def show_2():
    plt.imshow(small)
    plt.show()
    plt.imshow(medium)
    plt.show()
    plt.imshow(large)
    plt.show()


def apply_homography_2(a, h, target):
    r, c, col = np.shape(a)
    # Determine size of output image by forward-transforming the four corners of A
    p1_h, p2_h, p3_h, p4_h = (
        np.array([[0], [0], [1]], dtype=np.float),
        np.array([[c - 1], [0], [1]], dtype=np.float),
        np.array([[0], [r - 1], [1]], dtype=np.float),
        np.array([[c - 1], [r - 1], [1]], dtype=np.float),
    )

    # Apply homography on each point
    p1, p2, p3, p4 = np.dot(h, p1_h), np.dot(h, p2_h), np.dot(h, p3_h), np.dot(h, p4_h)
    p1, p2, p3, p4 = p1 / p1[2, 0], p2 / p2[2, 0], p3 / p3[2, 0], p4 / p4[2, 0]

    # Get new min and max from each axis
    min_x = floor(min([p1[0, 0], p2[0, 0], p3[0, 0], p4[0, 0]]))
    max_x = ceil(max([p1[0, 0], p2[0, 0], p3[0, 0], p4[0, 0]]))
    min_y = floor(min([p1[1, 0], p2[1, 0], p3[1, 0], p4[1, 0]]))
    max_y = ceil(max([p1[1, 0], p2[1, 0], p3[1, 0], p4[1, 0]]))

    # Obtain new row and column
    nr, nc = max_y - min_y + 1, max_x - min_x + 1

    # Initialize new array with black and then some background
    b = np.zeros([int(nr), int(nc), int(col)]) + 256

    # Pre-compute inverse of H
    h_inv = np.linalg.inv(h)

    pts_src = []
    pts_dst = []
    # Define function on coordinates
    for y in range(int(nr)):
        for x in range(int(nc)):
            p = np.array([[x + min_x], [y + min_y], [1]])
            pp = np.dot(h_inv, p)

            # De-homogenize
            xp, yp = pp[0, 0] / pp[2, 0], pp[1, 0] / pp[2, 0]

            # Interpolate
            xpf, ypf = int(floor(xp)), int(floor(yp))
            xpc, ypc = xpf + 1, ypf + 1

            # Placing the pixels in new image
            if (xpf >= 0) and (xpc < c) and (ypf >= 0) and (ypc < r):
                b[y, x, :] = (
                    (xpc - xp) * (ypc - yp) * a[ypf, xpf, :]
                    + (xpc - xp) * (yp - ypf) * a[ypc, xpf, :]
                    + (xp - xpf) * (ypc - yp) * a[ypf, xpc, :]
                    + (xp - xpf) * (yp - ypf) * a[ypc, xpc, :]
                )
                pts_src.append([yp, xp])
                pts_dst.append([y + min_y, x + min_x])
    x_min = int(min_x)
    y_min = int(min_y)
    tmp = b.astype(np.uint8)
    base = np.zeros_like(target)
    mask = tmp[:, :, :] == [256, 256, 256]
    for r in range(0, mask.shape[0]):
        for c in range(0, mask.shape[1]):
            if mask[r, c, 1] == False:
                base[r + y_min][c + x_min] = tmp[r, c, :]
    mask = base == [0, 0, 0]
    un_mask = np.invert(mask)
    base[mask] = target[mask]
    #     return base.astype(np.uint8), pts_src, pts_dst, int(min_x),int(min_x)
    return base.astype(np.uint8)


def draw_matches(matches, test_image, template):
    mask = np.zeros_like(test_image, dtype=bool)
    mask[: template.shape[0], : template.shape[1], :] = True
    un_mask = np.invert(mask)
    out_image = np.zeros_like(test_image)
    out_image[un_mask] = test_image[un_mask]
    for x in range(0, template.shape[0]):
        for y in range(0, template.shape[1]):
            out_image[x, y, :] = template[x, y, :]
    # Draw points
    for m in matches:
        cv2.circle(out_image, (m[0], m[1]), 3, colour, thickness=1, lineType=8, shift=0)
        cv2.circle(out_image, (m[2], m[3]), 3, colour, thickness=1, lineType=8, shift=0)
        cv2.line(out_image, (m[0], m[1]), (m[2], m[3]), colour, thickness=2, lineType=8)

    # Show Images
    plt.imshow(out_image)
    plt.show()


def get_good_matches(matches, threshold):
    end_points = matches[:, 2:4]
    start_points = matches[:, 0:2]
    good_matches = []
    for i in range(0, len(matches)):
        for j in range(i, len(matches)):
            #             p1 = end_points[random.randrange(0, len(matches), 1)]
            #             p2 = end_points[random.randrange(0, len(matches), 1)]
            p1 = end_points[i]
            p2 = end_points[j]
            p_1 = start_points[i]
            p_2 = start_points[j]
            distance = np.abs(np.cross(p2 - p1, p1 - end_points)) / np.linalg.norm(
                p2 - p1
            )
            distance_1 = np.abs(
                np.cross(p_2 - p_1, p_1 - start_points)
            ) / np.linalg.norm(p_2 - p_1)
            distance = distance[:] < threshold
            distance_1 = distance_1[:] < threshold / 2
            wololo = np.logical_and(distance, distance_1)
            tmp_matches = matches[wololo]
            if len(tmp_matches) > len(good_matches):
                good_matches = tmp_matches
    return good_matches


def draw_given_matches():
    template = plt.imread("fifa/template.jpg")
    corners = np.asarray(
        [
            [0, 0],
            [template.shape[1] - 1, 0],
            [0, template.shape[0] - 1],
            [template.shape[1] - 1, template.shape[0] - 1],
        ]
    )
    colour = (0, 200, 10)
    for i in range(1, 13):
        matches = pd.read_csv(
            f"fifa/fifasiftmatches/siftmatches_{i}.txt", sep="\s+", header=None
        )
        matches = np.asarray(matches, dtype=int)
        test_image = plt.imread(f"fifa/{i}.jpg")
        draw_matches(matches, test_image, template)


def draw_good_matches():
    for i in range(1, 13):
        matches = pd.read_csv(
            f"fifa/fifasiftmatches/siftmatches_{i}.txt", sep="\s+", header=None
        )
        matches = np.asarray(matches, dtype=int)
        test_image = plt.imread(f"fifa/{i}.jpg")
        matches = get_good_matches(matches, 10)
        draw_matches(matches, test_image, template)


def estimate_homography(matches):
    """
    1. repeat many times:
        1.1 choose 4 matches randomly
        1.2 calculate H (as we did in Lecture 10)
        1.3 map every (x i , y i ) with H, and compare with its match (x i 0 , y i 0 );
            those close enough form the consensus set
    2. pick the largest consensus set found: this is our set of inliers
    3. re-estimate H in a least-squares sense, using the entire set of inliers
    """
    test_matches = np.float32(matches[:, 0:2]).reshape(-1, 1, 2)
    test_results = np.float32(matches[:, 2:4]).reshape(-1, 1, 2)
    best_H = []
    best_count = 0
    for i in range(0, 1000):
        r1 = random.randrange(0, len(matches), 1)
        r2 = random.randrange(0, len(matches), 1)
        r3 = random.randrange(0, len(matches), 1)
        r4 = random.randrange(0, len(matches), 1)
        ran_matches = np.array([matches[r1], matches[r2], matches[r3], matches[r4]])
        H = find_homography(ran_matches[:, 0:2], ran_matches[:, 2:4])
        count = 0
        trans = cv2.perspectiveTransform(test_matches, H)
        for i in range(0, len(trans)):
            start = (trans[i][0][0], trans[i][0][0])
            end = (test_results[i][0][0], test_results[i][0][0])
            dist = math.sqrt(
                (start[0] - end[0] * 1.0) ** 2 + (start[1] - end[1] * 1.0) ** 2
            )
            if dist < 5:
                count += 1
            if count > best_count:
                best_count = count
                best_H = H

    return best_H


def identify_logo():
    template = plt.imread("fifa/template.jpg")
    corners = np.asarray(
        [
            [0, 0],
            [template.shape[1] - 1, 0],
            [0, template.shape[0] - 1],
            [template.shape[1] - 1, template.shape[0] - 1],
        ]
    )

    for i in range(1, 13):
        matches = pd.read_csv(
            f"fifa/fifasiftmatches/siftmatches_{i}.txt", sep="\s+", header=None
        )
        matches = np.asarray(matches, dtype=int)
        test_image = plt.imread(f"fifa/{i}.jpg")
        matches = get_good_matches(matches, 40)
        H = estimate_homography(matches)
        corners = np.float32(corners).reshape(-1, 1, 2)
        corners_1 = cv2.perspectiveTransform(corners, H)
        cv2.line(
            test_image,
            (corners_1[0][0][0], corners_1[0][0][1]),
            (corners_1[1][0][0], corners_1[1][0][1]),
            colour,
            10,
        )
        cv2.line(
            test_image,
            (corners_1[0][0][0], corners_1[0][0][1]),
            (corners_1[2][0][0], corners_1[2][0][1]),
            colour,
            10,
        )
        cv2.line(
            test_image,
            (corners_1[3][0][0], corners_1[3][0][1]),
            (corners_1[2][0][0], corners_1[2][0][1]),
            colour,
            10,
        )
        cv2.line(
            test_image,
            (corners_1[3][0][0], corners_1[3][0][1]),
            (corners_1[1][0][0], corners_1[1][0][1]),
            colour,
            10,
        )

        plt.imshow(test_image)
        plt.show()
