import cv2
import numpy as np
from skimage.draw import polygon
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree
from functools import reduce
import operator
import math
import alphashape
import shapely


def get_center_point(contour):
    M = cv2.moments(contour)
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])
    point = [center_x, center_y]
    return point

def get_extreme_points(contour):
    """
    计算轮廓四个极值点
    """
    contour = np.array(contour)
    top_most = list(contour[contour[:, 1].argmin()])
    left_most = list(contour[contour[:, 0].argmin()])
    bottom_most = list(contour[contour[:, 1].argmax()])
    right_most = list(contour[contour[:, 0].argmax()])
    return [top_most, left_most, bottom_most, right_most]


def get_contour_with_mask(mask):
    # opencv加载图片并提取边界[边界检测,绘制边界,极值点等](https://blog.csdn.net/sunny2038/article/details/12889059)
    ret, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def sort_points(coords):
    """coords=[[x,y],[x,y],...]"""
    center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
    return sorted(coords, key=lambda coord: (-135 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360, reverse=True)


def get_contour_from_points_with_delaunay(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)

    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle

    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        # print('circum_r', circum_r)

        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def calc_iou_with_polygon(polygon_1, polygon_2):
    """
    计算两个多边形的IoU,polygon=np.array([[x,y],[x,y],....])
    [参考](https://blog.csdn.net/qq_40081208/article/details/105309319)
    """
    rr1, cc1 = polygon(polygon_2[:, 0], polygon_2[:, 1])
    rr2, cc2 = polygon(polygon_1[:, 0], polygon_1[:, 1])
    if len(rr1) == 0 or len(cc1) == 0 or len(rr2) == 0 or len(cc2) == 0:
        return 0
    r_max = max(rr1.max(), rr2.max()) + 1
    c_max = max(cc1.max(), cc2.max()) + 1
    canvas = np.zeros((r_max, c_max))
    canvas[rr1, cc1] += 1
    canvas[rr2, cc2] += 1
    union = np.sum(canvas > 0)
    intersection = np.sum(canvas == 2)
    if union == 0:
        return 0
    return round(intersection / union, 4)


# opencv图片包括轮廓图,img:png图片,points:边界点数组
def show_contour(img, points):
    for i in range(len(points)):
        if i < len(points) - 1:
            cv2.line(img, (points[i][0], points[i][1]), (points[i + 1][0], points[i + 1][1]), (0, 0, 255), 1)
        else:
            cv2.line(img, (points[i][0], points[i][1]), (points[0][0], points[0][1]), (0, 0, 255), 1)
    return img


# cv2显示图像
def show_cv_img(img):
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def add_box_padding(size, box, padding):
    left, top = max(box[0][0] - padding, 0), max(box[0][1] - padding, 0)
    right, bottom = min(box[1][0] + padding, size[1] - 1), min(box[1][1] + padding, size[0] - 1)
    return [[left, top], [right, bottom]]


def pil_to_cv2(image):
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    # test calc_iou_with_polygon
    triangle_1 = np.array([
        [200, 100],
        [180, 180],
        [220, 180]
    ])
    rect_1 = np.array([
        [100, 100],
        [100, 200],
        [200, 200],
        [200, 100]
    ])
    print(calc_iou_with_polygon(rect_1, triangle_1))
