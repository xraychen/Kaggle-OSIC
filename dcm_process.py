import cv2
import pydicom
import numpy as np
from boundary_descriptor import *


def transform_ctdata(ct_dcm, windowWidth, windowCenter, CONVERT_DCM2GRAY=True):
    ct_slope = float(ct_dcm.RescaleSlope)
    ct_intercept = float(ct_dcm.RescaleIntercept)
    ct_img = ct_intercept + ct_dcm.pixel_array * ct_slope

    minWindow = - float(abs(windowCenter)) + 0.5 * float(abs(windowWidth))
    new_img = (ct_img - minWindow) / float(windowWidth)

    if np.average(new_img) > 1:
        try:
            minWindow = - float(abs(ct_dcm.WindowCenter)) + 0.5 * float(abs(windowWidth))
        except TypeError:
            minWindow = - float(abs(ct_dcm.WindowCenter[0])) + 0.5 * float(abs(windowWidth))
        new_img = (ct_img - minWindow) / float(windowWidth)

    new_img[new_img < 0] = 0
    new_img[new_img > 1] = 1
    if CONVERT_DCM2GRAY is True:
        new_img = (new_img * 255).astype(np.uint8)
    return new_img


def remove_img_nosie(img, contours, isShow=False):
    '''
        Only save contours part, else place become back.
        ===
        create a np.zeros array(black),
        use cv2.drawContours() make contours part become 255 (white),
        final, use cv2.gitwise_and() to remove noise for img
    '''
    crop_img = np.zeros(img.shape, np.uint8)
    crop_img = cv2.drawContours(
        crop_img.copy(), contours, -1, 255, thickness=-1)
    crop_img = cv2.bitwise_and(img, crop_img)

    if isShow is True:
        cv2.imshow('remove_img_nosie', crop_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pass
    return crop_img


def remove_black_frame(img, contour, isShow=False):
    feature = calc_contour_feature(img, contour)
    x, y, w, h = feature[0][3]
    img_center = [int(img.shape[0] / 2) + 1, int(img.shape[1] / 2) + 1]

    if img_center[0] > y:
        w = (img_center[1] - x) * 2 - 2
        h = (img_center[0] - y) * 2 - 2
        feature[0][3] = (x, y, w, h)
    else:
        x += w
        y += h
        w = (x - (img_center[1])) * 2 - 2
        h = (y - (img_center[0])) * 2 - 2
        feature[0][3] = (2 * img_center[1] - x, 2 * img_center[0] - y, w, h)

    img = get_crop_img_ls(
        img, feature, extra_W=-1, extra_H=-1, isShow=False)[0]
    new_img = np.ones(
        (img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8) * 255
    new_img[1:-1, 1:-1] = img

    return new_img


def get_biggest_countour(img, isShow=False):
    contours = get_contours_binary(img, THRESH_VALUE=100, whiteGround=False)
    new_contours = []
    contour_area_ls = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > (img.size * 0.05) and contour_area < (img.size * 0.9) and contour.size > 8:
            contour_area_ls.append(contour_area)
            new_contours.append(contour)

    if len(contour_area_ls) != 0:
        biggest_contour = [new_contours[contour_area_ls.index(max(contour_area_ls))]]
    else:
        biggest_contour = []

    return biggest_contour


def get_lung_img(img, isShow=False):
    lung_contour = get_biggest_countour(img, isShow=False)

    if len(lung_contour) != 0 and np.average(img[0:10, 0:10]) < 50:
        img = remove_black_frame(img, lung_contour, isShow=False)
        lung_contour = get_biggest_countour(img)

    lung_img = remove_img_nosie(img, lung_contour, isShow=False)
    features = calc_contour_feature(lung_img, lung_contour)
    lung_img = get_crop_img_ls(lung_img, features)[0]

    return lung_img


def get_square_img(img):
    square_size = max(img.shape[:])
    square_center = int(square_size / 2)
    output_img = np.zeros(
        (square_size, square_size), dtype='uint8')
    start_point_x = square_center - int(img.shape[0]/2)
    start_point_y = square_center - int(img.shape[1]/2)
    output_img[start_point_x: start_point_x + img.shape[0], start_point_y: start_point_y + img.shape[1]] = img

    return output_img
