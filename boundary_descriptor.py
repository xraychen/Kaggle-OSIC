import cv2
import numpy as np
'''
    This code edit by YU-SHUN from NTUT IEM
    oringal code is from NTUT IEM Ph.D TIEN

    All the function has a variable call: feature_list,
    this variable is from function: calc_contour_feature(),
    if user want to use the code, this function must be import.
'''


def get_threshold_mask(imgray, THRESH_VALUE=170):
    ret, thresh = cv2.threshold(
        imgray, THRESH_VALUE, 255, cv2.THRESH_BINARY)

    return thresh


def get_contours_binary(img, THRESH_VALUE=170, whiteGround=True):
    if len(img.shape) > 2:
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        imgray = img

    thresh = get_threshold_mask(imgray, THRESH_VALUE)
    if whiteGround:
        thresh_white = thresh
    else:
        thresh_white = 255 - thresh

    # if your python-cv version is lower than 4.0 the cv2.findContours will return 3 variable,
    # upper 4.0 : contours, hierarchy = cv2.findContours(XXX)
    # lower 4.0 : _, contours, hierarchy = cv2.findContours(XXX)
    contours, hierarchy = cv2.findContours(thresh_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def calc_contour_feature(img, contours):
    feature_list = list()
    for cont in contours:
        area = cv2.contourArea(cont)
        perimeter = cv2.arcLength(cont, closed=True)
        bbox = cv2.boundingRect(cont)
        bbox2 = cv2.minAreaRect(cont)
        circle = cv2.minEnclosingCircle(cont)
        if len(cont) > 5:
            ellipes = cv2.fitEllipse(cont)
        else:
            ellipes = None
        M = cv2.moments(cont)
        if area != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)
        else:
            center = (None, None)
        feature = [center, area, perimeter, bbox, bbox2, circle, ellipes]
        feature_list.append(feature)

    return feature_list


def draw_center(img, feature_list, color=(255, 255, 255), radius=5, isShow=True):
    img_center = img.copy()
    for f in feature_list:
        if f[0][0] is not None:
            img_center = cv2.circle(
                img_center, f[0], radius, color, -1)
            if isShow:
                cv2.imshow("image with center", img_center)
    return img_center


def draw_bbox(img, feature_list, color=(0, 255, 0), width=2, isShow=True):
    img_bbox = img.copy()
    for f in feature_list:
        (x, y, w, h) = f[3]
        img_bbox = cv2.rectangle(img_bbox, (x, y), (x + w, y + h), color, width)

    if isShow:
        cv2.imshow("image with bbox", img_bbox)
        cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
    return img_bbox


def get_crop_img_ls(img, feature_list, extra_W=0, extra_H=0, isShow=False):
    crop_img_ls = []
    for f in feature_list:
        (x, y, w, h) = f[3]
        x -= extra_W
        y -= extra_H
        w += extra_W * 2
        h += extra_H * 2

        new_position = [x, y]
        for i in range(len(new_position)):
            if new_position[i] < 0:
                new_position[i] = 0
        [x, y] = new_position

        if x + w > img.shape[1]:
            w = img.shape[1] - x
        if y + h > img.shape[0]:
            h = img.shape[0] - y

        crop_img = img[y: y + h, x: x + w]
        crop_img_ls.append(crop_img)

    if isShow:
        for crop_img in crop_img_ls:
            cv2.imshow("crop_img_bbox", crop_img)
            cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                pass
    return crop_img_ls


def draw_bbox2(img, feature_list, color=(0, 255, 0), width=2, isShow=True):
    img_bbox2 = img.copy()
    for f in feature_list:
        box = np.int0(cv2.boxPoints(f[4]))
        img_bbox = cv2.drawContours(img_bbox2, [box], -1, color, width)
    if isShow:
        cv2.imshow("image with bbox", img_bbox2)
    return img_bbox2


def draw_minSCircle(img, feature_list, color=(0, 255, 0), width=2, isShow=True):
    img_circle = img.copy()
    for f in feature_list:
        ((x, y), radius) = f[5]
        img_circle = cv2.circle(
            img_circle, (int(x), int(y)), int(radius), color, width)
    if isShow:
        cv2.imshow("image with bbox", img_circle)
    return img_circle


def get_signature_info(contours, center, isShow=False):
    cX, cY = center
    contours = np.array(contours)

    points = contours[0, :, 0, :]

    start_point = 0
    for i in range(points.shape[0]):
        pX = points[i, 0]
        pY = points[i, 1]
        if cY == pY:
            if cX < pX:
                print(pX, pY)
                print("start")
                start_point = i
                break

    center_dist_infos = {}
    vecter_0 = np.array([points[start_point, 0] - cX, 0])
    for i in range(points.shape[0]):
        pX = points[i, 0]
        pY = points[i, 1]

        center_dist = (abs(cX-pX)**2 + abs(cY-pY)**2)**(1/2)
        vecter_a = np.array([points[i, 0] - cX, cY - points[i, 1]])

        theta = np.degrees(
            np.arccos(np.dot(vecter_0, vecter_a) / (vecter_0[0] * center_dist)))
        if pY > cY:
            theta = 360 - theta

        center_dist_infos.update({theta: center_dist})

    center_dist_infos_ls = sorted(
        center_dist_infos.items(), key=lambda x: x[0])
    center_dist_infos = {}
    for item in center_dist_infos_ls:
        center_dist_infos.update({item[0]: item[1]})

    if isShow is True:
        import matplotlib.pyplot as plt

        plt.clf()
        plt.xlim([0, 360])
        plt.ylim([0, max(center_dist_infos.values()) + 20])
        a = list(center_dist_infos.keys())
        b = list(center_dist_infos.values())
        plt.plot(a, b)
        plt.show()

    return center_dist_infos
