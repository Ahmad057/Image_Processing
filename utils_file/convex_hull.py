import os
import cv2
import json
import numpy as np




def convex_hull_detection(img):
    
    gblur_filter = 5
    gblur_std = 0

    eliptic_kernel = 5
    binary_thresh =   (0, 255)

    edges_thresh1 = 30
    egdes_thresh2 = 100

    convexHull_preset = {
        "gblur_filter": gblur_filter,
        "gblur_std": gblur_std,
        "eliptic_kernel": eliptic_kernel,
        "binary_thresh": binary_thresh,
        "edges_thresh1": edges_thresh1,
        "egdes_thresh2": egdes_thresh2
    }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred_image = cv2.GaussianBlur(gray, (convexHull_preset["gblur_filter"], convexHull_preset["gblur_filter"]), convexHull_preset["gblur_std"])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (convexHull_preset["eliptic_kernel"], convexHull_preset["eliptic_kernel"]))

    opening = cv2.morphologyEx(blurred_image, cv2.MORPH_OPEN, kernel)
    _, thresh = cv2.threshold(opening, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    edges = cv2.Canny(thresh, threshold1=convexHull_preset["edges_thresh1"], threshold2=convexHull_preset["egdes_thresh2"])
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(thresh)


    areas = [cv2.contourArea(contour) for contour in contours]
    # area_thresh = ((max(areas) - min(areas))/2)*1.5
    area_thresh = 1000
    largest_indices = sorted(areas, reverse=True)[:1]

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area in largest_indices and area > area_thresh:
            hull = cv2.convexHull(contour)
            cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)

            x, y, w, h = cv2.boundingRect(contour)

            object_image = img[y:y + h, x:x + w]
            cv2.imwrite(f'hull-object/object_{i}.jpg', object_image)
            mask[:] = 0