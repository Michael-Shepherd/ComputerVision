import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
import sys

lego1 = plt.imread("CVassignment3_files/lego1.jpg")
lego2 = plt.imread("CVassignment3_files/lego2.jpg")
height = 9.6
width = 16
length = 32
colour = (255, 255, 0)
x_col = (255, 0, 0)
y_col = (0, 255, 0)
z_col = (0, 0, 255)

pts_x = [[width, 0, 0]] + [[width + x * length, 0, 0] for x in range(1, 8)]
pts_y = [[0, x * length, 0] for x in range(1, 7)]
pts_z = [[0, 0, x * height] for x in range(0, 14)]
pts_z.reverse()
pts_y.reverse()
pts = pts_z + pts_y + pts_x
pts = np.asarray(pts)
print(len(pts))
pts


pts_1 = np.array(
    [
        [1004, 812],
        [1004, 884],
        [1004, 960],
        [1000, 1032],
        [1004, 1108],
        [1004, 1184],
        [996, 1260],
        [1004, 1336],
        [1004, 1408],
        [1004, 1484],
        [1004, 1560],
        [1000, 1632],
        [996, 1704],
        [996, 1776],
        [244, 1448],
        [360, 1504],
        [476, 1552],
        [600, 1608],
        [728, 1664],
        [860, 1716],
        [1104, 1764],
        [1308, 1728],
        [1508, 1688],
        [1700, 1652],
        [1888, 1616],
        [2072, 1580],
        [2256, 1544],
        [2428, 1512],
    ]
)


pts_2 = np.array(
    [
        [1568, 770],
        [1568, 856],
        [1576, 948],
        [1568, 1028],
        [1572, 1108],
        [1564, 1196],
        [1568, 1280],
        [1564, 1360],
        [1560, 1440],
        [1556, 1524],
        [1560, 1608],
        [1556, 1684],
        [1560, 1768],
        [1556, 1848],
        [208, 1584],
        [416, 1624],
        [628, 1668],
        [852, 1716],
        [1084, 1752],
        [1316, 1804],
        [1624, 1816],
        [1756, 1744],
        [1884, 1676],
        [2004, 1620],
        [2124, 1556],
        [2232, 1492],
        [2336, 1432],
        [2436, 1380],
    ]
)


def decomposeP(P):
    # The input P is assumed to be a 3−by−4 homogeneous camera matrix.
    # The function returns a homogeneous 3−by−3 calibration matrix K,
    # a 3−by−3 rotation matrix R and a 3−by−1 vector c such that
    # K*R*[eye(3), −c] = P.
    W = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    # calculate K and R (up to sign)
    t = np.matmul(W, P[:, 0:3])
    Qt, Rt = np.linalg.qr(t.T)
    K = np.matmul(W, np.matmul(Rt.T, W))
    R = np.matmul(W, Qt.T)

    # correct for negative focal length(s) if necessary
    D = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if K[0, 0] < 0:
        D[0, 0] = -1
    if K[1, 1] < 0:
        D[1, 1] = -1
    if K[2, 2] < 0:
        D[2, 2] = -1
    K = np.matmul(K, D)
    R = np.matmul(D, R)

    # calculate c
    c = -1 * np.dot(R.T, np.dot(np.linalg.inv(K), P[:, 3]))
    return K, R, c


def scale(ma):
    """
    Take numpy matrix ma of size m,n and divide it by ma[m,n]
    """
    a = ma.shape
    if len(a) == 1:
        return ma / ma[a[0] - 1]
    return ma / ma[a[0] - 1, a[1] - 1]


def plot_lego_2d():
    lego1 = plt.imread("CVassignment3_files/lego1.jpg")
    lego2 = plt.imread("CVassignment3_files/lego2.jpg")

    for i in range(0, len(pts_2)):
        cv2.circle(lego1, (pts_1[i][0], pts_1[i][1]), 20, colour, thickness=-1)
        cv2.circle(lego2, (pts_2[i][0], pts_2[i][1]), 20, colour, thickness=-1)

    x = [(996, 1776), (244, 1448)]
    y = [(996, 1776), (1004, 812)]
    z = [(996, 1776), (2428, 1512)]
    cv2.line(lego1, x[0], x[1], y_col, 5)
    cv2.line(lego1, y[0], y[1], z_col, 5)
    cv2.line(lego1, z[0], z[1], x_col, 5)
    x = [(1556, 1848), (1568, 770)]
    y = [(1556, 1848), (208, 1584)]
    z = [(1556, 1848), (2436, 1380)]
    cv2.line(lego2, x[0], x[1], z_col, 5)
    cv2.line(lego2, y[0], y[1], y_col, 5)
    cv2.line(lego2, z[0], z[1], x_col, 5)

    plt.imshow(lego1)
    plt.show()
    plt.imshow(lego2)
    plt.show()


def plot_lego_3d():
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter([x[0] for x in pts], [x[1] for x in pts], [x[2] for x in pts])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


def get_P(x, X):
    # 2d => x, 3d => X
    A = []
    for i in range(0, len(x)):
        yX = x[i][1] * X[i]
        xX = x[i][0] * X[i]
        A.append(
            [0, 0, 0, 0, -X[i][0], -X[i][1], -X[i][2], -1, yX[0], yX[1], yX[2], x[i][1]]
        )
        A.append(
            [X[i][0], X[i][1], X[i][2], 1, 0, 0, 0, 0, -xX[0], -xX[1], -xX[2], -x[i][0]]
        )
    A = np.asarray(A)
    _, _, V = np.linalg.svd(A)
    p = V.T[:, V.shape[1] - 1]
    P = np.array(
        [[p[0], p[1], p[2], p[3]], [p[4], p[5], p[6], p[7]], [p[8], p[9], p[10], p[11]]]
    )
    return P


def check_P(image, P, points):
    for x in points:
        x = P.dot(np.array([x[0], x[1], x[2], 1]))
        cv2.circle(
            image, (int(x[0] / x[2]), int(x[1] / x[2])), 20, colour, thickness=-1
        )
    plt.imshow(image)
    plt.show()


def plot_cameras_3d(c1, R1, c2, R2):
    starts = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    ends = np.array([[100, 0, 0], [0, 100, 0], [0, 0, 100]])

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter([x[0] for x in pts], [x[1] for x in pts], [x[2] for x in pts])
    ax.scatter(c1[0], c1[1], c1[2], color="m", label="Camera 1")
    ax.scatter(c2[0], c2[1], c2[2], color="c", label="Camera 2")

    starts_1 = [np.matmul(R1, x) + c1 for x in starts]
    ends_1 = [np.matmul(R1, x) + c1 for x in ends]

    starts_x = [x[0] for x in starts_1]
    starts_y = [x[1] for x in starts_1]
    starts_z = [x[2] for x in starts_1]
    ends_x = [x[0] for x in ends_1]
    ends_y = [x[1] for x in ends_1]
    ends_z = [x[2] for x in ends_1]

    ax.plot(
        [starts_x[0], ends_x[0]],
        [starts_y[0], ends_y[0]],
        zs=[starts_z[0], ends_z[0]],
        color="r",
        label="x1",
    )
    ax.plot(
        [starts_x[1], ends_x[1]],
        [starts_y[1], ends_y[1]],
        zs=[starts_z[1], ends_z[1]],
        color="g",
        label="y1",
    )
    ax.plot(
        [starts_x[2], ends_x[2]],
        [starts_y[2], ends_y[2]],
        zs=[starts_z[2], ends_z[2]],
        color="b",
        label="z1",
    )

    starts_2 = [np.matmul(R2, x) + c2 for x in starts]
    ends_2 = [np.matmul(R2, x) + c2 for x in ends]

    starts_x = [x[0] for x in starts_2]
    starts_y = [x[1] for x in starts_2]
    starts_z = [x[2] for x in starts_2]
    ends_x = [x[0] for x in ends_2]
    ends_y = [x[1] for x in ends_2]
    ends_z = [x[2] for x in ends_2]
    ax.plot(
        [starts_x[0], ends_x[0]],
        [starts_y[0], ends_y[0]],
        zs=[starts_z[0], ends_z[0]],
        color="r",
        label="x2",
    )
    ax.plot(
        [starts_x[1], ends_x[1]],
        [starts_y[1], ends_y[1]],
        zs=[starts_z[1], ends_z[1]],
        color="g",
        label="y2",
    )
    ax.plot(
        [starts_x[2], ends_x[2]],
        [starts_y[2], ends_y[2]],
        zs=[starts_z[2], ends_z[2]],
        color="b",
        label="z2",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend()
    plt.show()


def one_a():
    P1 = get_P(pts_1, pts)
    P1_d = P1 / P1[2, 3]
    lego1 = plt.imread("CVassignment3_files/lego1.jpg")
    plot_lego_3d()
    check_P(lego1, P1, pts)
    print(P1)


def one_b():
    K1, R1, c1 = decomposeP(P1_d)
    R1_looks_right = np.linalg.inv(R1)
    print(scale(K1))
    print(R1)
    print(c1)


def two_a():
    P2 = get_P(pts_2, pts)
    P2_d = P2 / P2[2, 3]
    lego2 = plt.imread("CVassignment3_files/lego2.jpg")
    plot_lego_3d()
    check_P(lego2, P2, pts)
    print(P2)
    print(P2_d)
    K2, R2, c2 = decomposeP(P2_d)
    R2_looks_right = np.linalg.inv(R2)

    print(scale(K2))
    print(R2)
    print(c2)


def two_b():
    dist = np.sqrt(np.sum((c1 - c2) ** 2, axis=0))
    vec = [0, 0, 1]
    v1 = np.dot(R1_looks_right, vec)
    v2 = np.dot(R2_looks_right, vec)
    m1 = 1
    m2 = 1
    angle = np.arccos(abs(np.dot(v1, v2)))

    print(f"Distance: {dist}mm")
    print(f"Angle: {np.rad2deg(angle)}{chr(176)}")


def three_a():
    p1_4 = P1_d[..., 3] / P1_d[..., 3][2]
    p2_4 = P2_d[..., 3] / P2_d[..., 3][2]
    lego1 = plt.imread("CVassignment3_files/lego1.jpg")
    lego2 = plt.imread("CVassignment3_files/lego2.jpg")
    cv2.circle(lego1, (int(p1_4[0]), int(p1_4[1])), 20, colour, thickness=-1)
    plt.imshow(lego1)
    plt.show()
    cv2.circle(lego2, (int(p2_4[0]), int(p2_4[1])), 20, colour, thickness=-1)
    plt.imshow(lego2)
    plt.show()


def three_b():
    lego1 = plt.imread("CVassignment3_files/lego1.jpg")
    lego2 = plt.imread("CVassignment3_files/lego2.jpg")

    p1_1 = scale(P1)[..., 0] / scale(P1)[..., 0][2]
    p2_1 = scale(P2)[..., 0] / scale(P2)[..., 0][2]
    p1_2 = scale(P1)[..., 1] / scale(P1)[..., 1][2]
    p2_2 = scale(P2)[..., 1] / scale(P2)[..., 1][2]
    p1_3 = scale(P1)[..., 2] / scale(P1)[..., 2][2]
    p2_3 = scale(P2)[..., 2] / scale(P2)[..., 2][2]

    x = [(int(p1_4[0]), int(p1_4[1])), (int(p1_1[0]), int(p1_1[1]))]
    y = [(int(p1_4[0]), int(p1_4[1])), (int(p1_2[0]), int(p1_2[1]))]
    z = [(int(p1_4[0]), int(p1_4[1])), (int(-p1_3[0]), int(-p1_3[1]))]
    cv2.line(lego1, x[0], x[1], x_col, 10)
    cv2.line(lego1, y[0], y[1], y_col, 10)
    cv2.line(lego1, z[0], z[1], z_col, 10)

    x = [(int(p2_4[0]), int(p2_4[1])), (int(p2_1[0]), int(p2_1[1]))]
    y = [(int(p2_4[0]), int(p2_4[1])), (int(p2_2[0]), int(p2_2[1]))]
    z = [(int(p2_4[0]), int(p2_4[1])), (int(p2_3[0]), int(-p2_3[1]))]
    cv2.line(lego2, x[0], x[1], x_col, 10)
    cv2.line(lego2, y[0], y[1], y_col, 10)
    cv2.line(lego2, z[0], z[1], z_col, 10)

    plt.imshow(lego1)
    plt.show()
    plt.imshow(lego2)
    plt.show()


def four_a():
    C = [c1[0], c1[1], c1[2], 1]
    C_1 = [c2[0], c2[1], c2[2], 1]
    e = P1_d.dot(C_1)
    print(f"e:[{e[0]}, {e[1]}, {e[2]}]")
    e = e / e[2]
    print(f"e:[{e[0]}, {e[1]}, {e[2]}]")

    e_1 = P2_d.dot(C)
    print(f"e':[{e_1[0]}, {e_1[1]}, {e_1[2]}]")
    e_1 = e_1 / e_1[2]
    print(f"e':[{e_1[0]}, {e_1[1]}, {e_1[2]}]")
    return e, e_1


def four_b():
    e_1_x = np.array([[0, -e_1[2], e_1[1]], [e_1[2], 0, -e_1[0]], [-e_1[1], e_1[0], 0]])
    P_plus = P1_d.T.dot(np.linalg.inv(P1_d.dot(P1_d.T)))
    F = e_1_x.dot(P2_d.dot(P_plus))
    print(F.tolist())
    return F


def l1(l2):
    # l = F.T x
    return np.dot(F.T, np.dot(e_1_x, l2))


def l2(x1):
    # l' = F x'
    return np.dot(F, x1)


def show_4_c():
    lego1 = plt.imread("CVassignment3_files/lego1.jpg")
    lego2 = plt.imread("CVassignment3_files/lego2.jpg")

    colours = [
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for x in range(0, 15)
    ]
    h, w, _ = np.shape(lego1)

    i = 0
    for y in range(0, w, int(w / (len(colours) - 1))):
        l_2 = l2(np.array([0, y, 1]))
        l_1 = l1(l_2)

        y1_1 = int((-1 * l_1[2]) / l_1[1])
        y2_1 = int((-1 * l_2[2]) / l_2[1])

        y1_2 = int((-1 * l_1[2] - l_1[0] * h) / l_1[1])
        y2_2 = int((-1 * l_2[2] - l_2[0] * h) / l_2[1])

        cv2.line(lego1, (0, y1_1), (w, y1_2), colours[i], 10)
        cv2.line(lego2, (0, y2_1), (w, y2_2), colours[i], 10)
        i += 1
    plt.imshow(lego1)
    plt.show()
    plt.imshow(lego2)
    plt.show()
