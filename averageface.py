# coding=utf8

import os
import math
import cv2
import glob
import dlib
import numpy as np

# 初始化路径
predictor_path = 'shape_predictor_68_face_landmarks.dat'
path = 'images'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def main():

    # 输出人脸的分辨率
    w = 400
    h = 600

    # 读取图片并计算特征点
    images, all_points = read(path)

    eyecorner_dst = [(np.int(0.3 * w), np.int(h / 3)),
                     (np.int(0.7 * w), np.int(h / 3))]

    images_norm = []
    points_norm = []

    boundary_pts = np.array([(0, 0), (w / 2, 0), (w - 1, 0), (w - 1, h / 2),
                             (w - 1, h - 1), (w / 2, h - 1), (0, h - 1),
                             (0, h / 2)])

    # 初始化用于记录特征点平均值的数组
    points_avg = np.array([(0, 0)] * (68 + len(boundary_pts)), np.float32())

    num_images = len(images)

    # 遍历之前读入的图像及其特征点
    for i in range(0, num_images):
        points1 = all_points[i]

        eyecorner_src = [all_points[i][36], all_points[i][45]]

        tform = similarity_transform(eyecorner_src, eyecorner_dst)

        img = cv2.warpAffine(images[i], tform, (w, h))

        points2 = np.reshape(np.array(points1), (68, 1, 2))
        points = cv2.transform(points2, tform)
        points = np.float32(np.reshape(points, (68, 2)))

        points = np.append(points, boundary_pts, axis=0)

        points_avg = points_avg + points / num_images

        points_norm.append(points)
        images_norm.append(img)

    rect = (0, 0, w, h)
    tri = calculate_triangles(rect, np.array(points_avg))

    output = np.zeros((h, w, 3), np.float32())

    for i in range(0, len(images_norm)):
        img = np.zeros((h, w, 3), np.float32())
        for j in range(0, len(tri)):
            t_in = []
            t_out = []

            for k in range(0, 3):
                p_in = points_norm[i][tri[j][k]]
                p_in = constrain_point(p_in, w, h)

                p_out = points_avg[tri[j][k]]
                p_out = constrain_point(p_out, w, h)

                t_in.append(p_in)
                t_out.append(p_out)

            warp_triangle(images_norm[i], img, t_in, t_out)

        output = output + img

    output = output / num_images

    # 输出结果图像
    cv2.imshow('AverageFace', output)
    cv2.waitKey(0)


# 图像读入和特征点计算函数


def read(path):
    # 初始化数组
    points_array = []
    images_array = []

    # 遍历path下的图片
    for f in glob.glob(os.path.join(path, "*.jpg")):
        print("Processing file: {}".format(f))
        img = cv2.imread(f)
        cv2.imshow(f, img)
        cv2.waitKey(1)
        fimg = np.float32(img) / 255.0
        # 将图片用浮点化表示，方便之后的运算
        dets = detector(img, 1)
        print("Number of faces detected: {}".format(len(dets)))

        for k, d in enumerate(dets):
            images_array.append(fimg)
            points = []
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))

            shape = predictor(img, d)
            print("Part 0: {}, Part 1: {} ...".format(
                shape.part(0), shape.part(1)))
            # 着色特征点
            for n in range(0, 68):
                points.append((int(shape.part(n).x), int(shape.part(n).y)))
                cv2.circle(img, (shape.part(n).x, shape.part(n).y), 2,
                           (0, 255, 0), -1)
            points_array.append(points)
        cv2.imshow(f, img)
        cv2.waitKey(1)

    # 函数返回
    return images_array, points_array


# 对图像特征点进行近似变化
def similarity_transform(in_points, out_points):
    s60 = math.sin(60 * math.pi / 180)
    c60 = math.cos(60 * math.pi / 180)

    in_pts = np.copy(in_points).tolist()
    out_pts = np.copy(out_points).tolist()

    xin = c60 * (in_pts[0][0] - in_pts[1][0]) - s60 * \
        (in_pts[0][1] - in_pts[1][1]) + in_pts[1][0]
    yin = s60 * (in_pts[0][0] - in_pts[1][0]) + c60 * \
        (in_pts[0][1] - in_pts[1][1]) + in_pts[1][1]

    in_pts.append([np.int(xin), np.int(yin)])

    xout = c60 * (out_pts[0][0] - out_pts[1][0]) - s60 * \
        (out_pts[0][1] - out_pts[1][1]) + out_pts[1][0]
    yout = s60 * (out_pts[0][0] - out_pts[1][0]) + c60 * \
        (out_pts[0][1] - out_pts[1][1]) + out_pts[1][1]

    out_pts.append([np.int(xout), np.int(yout)])

    tform = cv2.estimateRigidTransform(
        np.array([in_pts]), np.array([out_pts]), False)

    return tform


# 判断点是否在范围


def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# 辅助计算函数


def calculate_triangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)

    for p in points:
        subdiv.insert((p[0], p[1]))

    triangle_list = subdiv.getTriangleList()
    delaunay_tri = []

    for t in triangle_list:
        pt = []

        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(rect, pt1) and rect_contains(
                rect, pt2) and rect_contains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if abs(pt[j][0] - points[k][0]) < 1.0 and abs(
                            pt[j][1] - points[k][1]) < 1.0:
                        ind.append(k)
            if len(ind) == 3:
                delaunay_tri.append((ind[0], ind[1], ind[2]))

    return delaunay_tri


def constrain_point(p, w, h):
    p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))

    return p


# 变换处理


def apply_affine_transform(src, src_tri, dst_tri, size):

    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))

    dst = cv2.warpAffine(
        src,
        warp_mat, (size[0], size[1]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101)

    return dst


def warp_triangle(img1, img2, t1, t2):

    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(0, 3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)
    img2_rect = img2_rect * mask

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] +
         r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
             (1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] +
         r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect


if __name__ == '__main__':
    main()
