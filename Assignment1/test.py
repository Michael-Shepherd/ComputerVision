import numpy as np


def bl_resize(base_image, scale):
    print("Starting Bilinear Interpolation")
    x_scale = int(scale * base_image.shape[0])
    y_scale = int(scale * base_image.shape[1])
    resized = np.zeros((x_scale, y_scale, 3), dtype=np.uint8)
    for x in range(0, x_scale):
        for y in range(0, y_scale):
            x_old = x / scale
            if x_old >= base_image.shape[0]:
                x_old = base_image.shape[0] - 1
            y_old = y / scale
            if y_old >= base_image.shape[1]:
                y_old = base_image.shape[1] - 1

            xf = floor(x_old)
            xc = ceil(x_old)
            yf = floor(y_old)
            yc = ceil(y_old)
            c_ff = base_image[xf, yf, :].astype(dtype=np.uint8)
            c_cf = base_image[xc, yc, :].astype(dtype=np.uint8)
            c_fc = base_image[xf, yc, :].astype(dtype=np.uint8)
            c_cc = base_image[xc, yc, :].astype(dtype=np.uint8)

            resized[x, y, :] = (
                (yc - y_old) * (((xc - x_old) * c_ff) + (x_old - xf) * (c_cf))
                + (y_old - yf) * (((xc - x_old) * c_fc) + ((x_old - xf) * c_cc))
            ).astype(dtype=np.uint8)

    print("Done resizing")
    return resized

