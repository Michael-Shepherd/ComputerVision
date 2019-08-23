import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from math import cos, sin, radians, floor, ceil
from numpy import dot, array


def apply_homography(a, h):

    r, c, col = np.shape(a)

    # Determine size of output image by forward-transforming the four corners of A
    p1_h, p2_h, p3_h, p4_h = array([[0], [0], [1]], dtype=np.float), \
                             array([[c - 1], [0], [1]], dtype=np.float), \
                             array([[0], [r - 1], [1]], dtype=np.float), \
                             array([[c - 1], [r - 1], [1]], dtype=np.float)

    # Apply homography on each point
    p1, p2, p3, p4 = dot(h, p1_h), dot(h, p2_h), dot(h, p3_h), dot(h, p4_h)
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

    points = []

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
                b[y, x, :] = ((xpc - xp) * (ypc - yp) * a[ypf, xpf, :]
                              + (xpc - xp) * (yp - ypf) * a[ypc, xpf, :]
                              + (xp - xpf) * (ypc - yp) * a[ypf, xpc, :]
                              + (xp - xpf) * (yp - ypf) * a[ypc, xpc, :])
                points.append([y, x])

    return b, points, int(min_x), int(min_y)


if __name__ == "__main__":

    a = np.copy(mpimg.imread('question1b.jpg'))

    # Show original
    plt.imshow(a.astype("uint8"))
    plt.show()

    r, c, col = np.shape(a)

    # Apply homography of 3 different s and theta values
    s1 = 0.5
    s2 = 1
    s3 = 2
    t1 = 30
    t2 = 90
    t3 = 175

    h1 = np.array([[s1*cos(t1), -s1*sin(t1), c],
                  [s1*sin(t1), s1*cos(t1), r],
                  [0, 0, 1]])
    h2 = np.array([[s2*cos(t2), -s2*sin(t2), c],
                   [s2*sin(t2), s2*cos(t2), r],
                   [0, 0, 1]])
    h3 = np.array([[s3*cos(t3), -s3*sin(t3), c],
                   [s3*sin(t3), s3*cos(t3), r],
                   [0, 0, 1]])

    b, _, _, _ = apply_homography(a, h1)
    plt.imshow(b.astype("uint8"))
    plt.show()

    b, _, _, _ = apply_homography(a, h2)
    plt.imshow(b.astype("uint8"))
    plt.show()

    b, _, _, _ = apply_homography(a, h3)
    plt.imshow(b.astype("uint8"))
    plt.show()
