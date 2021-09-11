import cv2
import numpy as np


def get_gaussian_kernel(sigma=1., kernel_size=7):
    line = np.linspace(-kernel_size // 2, kernel_size // 2, kernel_size)
    x, y = np.meshgrid(line, line)
    gaussian_kernel = np.exp(-(x ** 2 + y ** 2) / sigma ** 2 / 2)
    return gaussian_kernel / gaussian_kernel.sum()


def point_to_gaussian(center, height, width, sigma=3, kernel_size=5):
    image = np.zeros((height, width), dtype=np.float32)
    kernel = get_gaussian_kernel(sigma, kernel_size)
    kernel = kernel / kernel.max()
    image = np.pad(image, kernel_size)
    center = np.array(center) + kernel_size
    half = kernel_size // 2
    image[center[0] - half:center[0] + half + 1, center[1] - half:center[1] + half + 1] = kernel
    image = image[kernel_size:-kernel_size, kernel_size:-kernel_size]
    return image


def soft_arg_max(image, beta=100.):
    image = image - image.max()
    height, width = image.shape
    position_map_x = np.arange(height)
    position_map_y = np.arange(width)
    y, x = np.meshgrid(position_map_y, position_map_x)
    x_index = np.sum(np.e ** (beta * image) * x) / np.sum(np.e ** (beta * image))
    y_index = np.sum(np.e ** (beta * image) * y) / np.sum(np.e ** (beta * image))

    return np.array([y_index, x_index])


def get_gaussian_heatmap(height, width, point, sigma=15.):
    line_x = np.linspace(-point[1], height - point[1], height, dtype=np.float64)
    line_y = np.linspace(-point[0], width - point[0], width, dtype=np.float64)

    x, y = np.meshgrid(line_y, line_x)
    gaussian_kernel = np.exp(-(x ** 2 + y ** 2) / sigma ** 2 / 2)
    return gaussian_kernel.astype(np.float32)


def wrap_box_region(heatmap, boxes, kernel=None):
    if kernel is None:
        kernel = get_gaussian_kernel(9, 51)
    src_box = np.array([[0, 0], [kernel.shape[1], 0], [kernel.shape[1], kernel.shape[0]], [0, kernel.shape[0]]],
                       dtype=np.float32)
    for box in boxes:
        matrix = cv2.getPerspectiveTransform(src_box, box.astype(np.float32))
        warped = cv2.warpPerspective(kernel, matrix, (heatmap.shape[1], heatmap.shape[0]))
        heatmap += warped


def get_triangle_center(box):
    center = box.mean(0)
    center1 = (box[0] + box[1] + center) / 3
    center2 = (box[2] + box[3] + center) / 3
    return center1, center2


def wrap_box_affine(heatmap, boxes, kernel=None):
    if kernel is None:
        kernel = get_gaussian_kernel(9, 51)
    src_box = np.array([[0, 0], [kernel.shape[1], 0], [kernel.shape[1], kernel.shape[0]], [0, kernel.shape[0]]],
                       dtype=np.float32)
    for i in range(len(boxes) - 1):
        box1, box2 = boxes[i:i + 2]
        center1_top, center1_bottom = get_triangle_center(box1)
        center2_top, center2_bottom = get_triangle_center(box2)
        box = np.array([center1_top, center2_top, center2_bottom, center1_bottom])
        matrix = cv2.getPerspectiveTransform(src_box, box.astype(np.float32))
        warped = cv2.warpPerspective(kernel, matrix, (heatmap.shape[1], heatmap.shape[0]))
        heatmap += warped
