import numpy as np
import os


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def rgb2ycbcr(x):
    y = (24.966 * x[:, :, :, 2] + 128.553 * x[:, :, :, 1] + 65.481 * x[:, :, :, 0] + 16) / 255
    cb = (112 * x[:, :, :, 2] - 74.203 * x[:, :, :, 1] - 37.797 * x[:, :, :, 0] + 128) / 255
    cr = (-18.214 * x[:, :, :, 2] - 93.789 * x[:, :, :, 1] + 112 * x[:, :, :, 0] + 128) / 255
    y = np.stack([y, cb, cr], axis=3)
    return y


def ycbcr2rgb(x):
    r = 1.16438356 * (x[:, :, :, 0] - 16 / 255) + 1.59602715 * (x[:, :, :, 2] - 128 / 255)
    g = 1.16438356 * (x[:, :, :, 0] - 16 / 255) - 0.3917616 * (x[:, :, :, 1] - 128 / 255) - 0.81296805 * (
            x[:, :, :, 2] - 128 / 255)
    b = 1.16438356 * (x[:, :, :, 0] - 16 / 255) + 2.01723105 * (x[:, :, :, 1] - 128 / 255)
    y = np.stack([r, g, b], axis=3)
    return y


def metric(x, y, border_cut):
    if border_cut > 0:
        x = x[border_cut:-border_cut, border_cut:-border_cut, :]
        y = y[border_cut:-border_cut, border_cut:-border_cut, :]
    else:
        x = x
        y = y

    mse = np.mean((x - y) ** 2)
    return 20 * np.log10(1 / np.sqrt(mse))