from functools import partial

import numpy as np
import torch as t
import torch.nn.functional as F
from scipy.optimize import leastsq


def _bezier(i, n, _t):
    return np.math.comb(n, i) * _t ** i * (1 - _t) ** (n - i)


def bezier_prime(i, n, _t):
    if i == 0:
        return -n * (1 - _t) ** (n - 1)
    elif i == n:
        return n * _t ** (n - 1)
    else:
        return np.math.comb(n, i) * _t ** (i - 1) * (1 - _t) ** (n - i - 1) * (i - n * _t)


def _get_bezier_curve_torch(control_points, interval=100, expand=1.):
    order = len(control_points[0]) - 1
    relative_distance = t.linspace(1 - expand, expand, interval, dtype=t.float32, device=control_points.device)
    bezier_matrix = t.stack([_bezier(i, order, relative_distance) for i in range(order + 1)], 1)
    bezier_matrix = bezier_matrix.repeat(len(control_points), 1, 1)
    fitted = t.einsum('bij,bjk->bik', bezier_matrix, control_points)
    return fitted


def _bezier_grid(bezier_pts, height, width, dest_height, dest_width, h_expand, w_expand):
    batch_size = len(bezier_pts)
    device = bezier_pts.device
    top_line = bezier_pts[:, :len(bezier_pts[0]) // 2]
    bottom_line = t.flip(bezier_pts[:, len(bezier_pts[0]) // 2:], (1,))
    top_curve = _get_bezier_curve_torch(top_line, dest_width, w_expand).unsqueeze(1).repeat(1, dest_height, 1, 1)
    bottom_curve = _get_bezier_curve_torch(bottom_line, dest_width, w_expand).unsqueeze(1).repeat(1, dest_height, 1,
                                                                                                  1)
    percent = t.linspace(h_expand, 1 - h_expand, dest_height, device=device).reshape(1, -1, 1, 1).repeat(batch_size, 1,
                                                                                                         dest_width, 2)

    grid = top_curve * percent + bottom_curve * (1 - percent)
    grid = grid / t.tensor([width, height], dtype=t.float32, device=device).reshape(1, 1, 1, 2) * 2 - 1
    return grid


def bezier_align(image, bezier_pts, dest_height, dest_width, h_expand=1., w_expand=1.):
    grid = _bezier_grid(bezier_pts, image.shape[2], image.shape[3], dest_height, dest_width, h_expand, w_expand)
    new_image = F.grid_sample(image, grid, align_corners=False)
    return new_image


def get_bezier_curve(control_points, interval=100, ratio=1., func=_bezier):
    if isinstance(ratio, (int, float)):
        ratio = (1 - ratio, ratio)
    order = len(control_points) - 1
    relative_distance = np.linspace(ratio[0], ratio[1], interval)
    bezier_matrix = np.stack([func(i, order, relative_distance) for i in range(order + 1)], 1)
    fitted = bezier_matrix @ control_points
    return fitted


def get_bezier_polygon(control_points, interval=100, width_ratio=1., height_ratio=1.):
    if isinstance(width_ratio, (int, float)):
        width_ratio = (width_ratio, width_ratio)
    top_line, bottom_line = control_points[:len(control_points) // 2], control_points[len(control_points) // 2:]
    top_curve = get_bezier_curve(top_line, interval, width_ratio[0])
    bottom_curve = get_bezier_curve(bottom_line, interval, width_ratio[1])
    if height_ratio != 1.:
        top_curve, bottom_curve = top_curve * height_ratio + bottom_curve * (1 - height_ratio), bottom_curve * (
                1 - height_ratio) + top_curve * height_ratio
    polygon = np.concatenate([top_curve, bottom_curve])
    return polygon


def _fit_bezier_torch(source_point, order=3):
    if len(source_point) == 1:
        raise ValueError("to fit a bezier curve, you should pass at last 2 points!")
    while len(source_point) < order + 1:
        temp = [source_point[0]]
        for i in range(len(source_point) - 1):
            temp.append((source_point[i] + source_point[i + 1]) / 2)
            temp.append(source_point[i + 1])
        source_point = t.tensor(temp)

    source_point = source_point.float()
    relative_distance = t.linalg.norm(source_point[1:] - source_point[:-1], dim=1)
    relative_distance = relative_distance / relative_distance.sum()
    relative_distance = t.cat([t.tensor([0], device=source_point.device), t.cumsum(relative_distance, 0)])
    bezier_matrix = t.stack([_bezier(i, order, relative_distance) for i in range(order + 1)], 1)

    para = (t.inverse(bezier_matrix.T @ bezier_matrix) @ bezier_matrix.T @ source_point)

    return para


def _fit_bezier(source_point, order=3):
    if len(source_point) == 1:
        raise ValueError("to fit a bezier curve, you should pass at last 2 points!")
    while len(source_point) < order + 1:
        temp = [source_point[0]]
        for i in range(len(source_point) - 1):
            temp.append((source_point[i] + source_point[i + 1]) / 2)
            temp.append(source_point[i + 1])
        source_point = np.array(temp)

    def optim_func(p, x, x0, x3, y0, y3):
        parameter = np.array([[x0, y0], *np.array(p).reshape(-1, 2), [x3, y3]])
        return (x @ parameter).reshape(-1)

    def error(p, x, y, func):
        return func(p, x).reshape(-1) - y

    b_x0, b_y0 = source_point[0]
    b_x3, b_y3 = source_point[-1]
    relative_distance = np.linalg.norm(source_point[1:] - source_point[:-1], axis=1)
    relative_distance = relative_distance / relative_distance.sum()
    relative_distance = np.concatenate([np.array([0]), np.cumsum(relative_distance)])
    bezier_matrix = np.stack([_bezier(i, order, relative_distance) for i in range(order + 1)], 1)

    y = source_point
    weight = np.array([np.linspace(1, 0, order + 1)[1:-1], np.linspace(0, 1, order + 1)[1:-1]]).transpose(1, 0)
    p0 = (weight @ np.array([source_point[0], source_point[-1]])).reshape(-1)
    func = partial(optim_func, x0=b_x0, x3=b_x3, y0=b_y0, y3=b_y3)
    error_func = partial(error, func=func)
    para = leastsq(error_func, p0, args=(bezier_matrix, y.reshape(-1)))[0]  # 进行拟合

    return np.array([[b_x0, b_y0], *np.array(para).reshape(-1, 2), [b_x3, b_y3]])


def _smart_split(polygon):
    angle_index = []
    for i in range(len(polygon) - 3):
        p1, p2, p3, p4 = polygon[i:i + 4]
        angle = (p2 - p1).dot(p4 - p3) / np.linalg.norm(p2 - p1) / np.linalg.norm(p4 - p3)
        angle_index.append((angle, i))
    angle_index.sort()
    index = angle_index[0][1]
    return polygon[:index + 2], polygon[index + 2:]


def get_bezier_control_point(polygon, order=3, smart_split=False, smart_reverse=False):
    if smart_split:
        top_line, bottom_line = _smart_split(polygon)
    else:
        top_line, bottom_line = polygon[:len(polygon) // 2], polygon[len(polygon) // 2:]
    if smart_reverse and top_line[-1][0] < top_line[0][0]:
        top_line = top_line[::-1]
        bottom_line = bottom_line[::-1]
    top_control = _fit_bezier(top_line, order)
    bottom_control = _fit_bezier(bottom_line, order)
    control = np.concatenate([top_control, bottom_control], 0)
    return control


def _adjust_corner(a, b, c, d):
    angle1 = (a - b).dot(c - b) / np.linalg.norm(a - b) / np.linalg.norm(c - b)
    angle2 = (d - c).dot(b - c) / np.linalg.norm(d - c) / np.linalg.norm(b - c)
    # both angle are rectangle
    if abs(angle1) < 0.1 and abs(angle2) < 0.1:
        return -1, None
    index = 0
    if angle1 > angle2:
        index = 1
        a, b, c, d = d, c, b, a
    x1, y1 = a
    x2, y2 = b
    x0, y0 = c
    # k = |AP| / |AB|
    k = ((x0 - x1) * (x2 - x1) + (y0 - y1) * (y2 - y1)) / ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    x = x1 + k * (x2 - x1)
    y = y1 + k * (y2 - y1)
    return index, np.array([x, y])


def adjust_control_point(control_points):
    polygons = get_bezier_polygon(control_points, 10)
    top_curve, bottom_curve = polygons[:len(polygons) // 2], polygons[len(polygons) // 2:]
    top_control, bottom_control = control_points[:len(control_points) // 2], control_points[len(control_points) // 2:]
    a, b, c, d = top_curve[1], top_curve[0], bottom_curve[-1], bottom_curve[-2]
    index, new_point = _adjust_corner(a, b, c, d)
    if index == 0:
        top_control[0] = new_point.reshape(-1, 2)
    elif index == 1:
        bottom_control[-1] = new_point.reshape(-1, 2)

    a, b, c, d = top_curve[-2], top_curve[-1], bottom_curve[0], bottom_curve[1]
    index, new_point = _adjust_corner(a, b, c, d)
    if index == 0:
        top_control[-1] = new_point.reshape(-1, 2)
    elif index == 1:
        bottom_control[0] = new_point.reshape(-1, 2)

    return np.concatenate([top_control, bottom_control])
