import matplotlib.pyplot as plt
from skimage import data,filters
import scipy.ndimage as ndimage
import numpy as np
from PIL import Image
from math import floor, ceil
import pandas as pd
import cv2
import os
import math

image = plt.imread("greenscreen.jpg")
cat =plt.imread("cat.jpg")
cat_background = cat[130:505,200:700,:]
orb = cv2.ORB_create()
minHessian = 400
surf = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
cathedral = cv2.imread('cathedral.jpg')
cathedral = cv2.cvtColor(cathedral, cv2.COLOR_BGR2RGB)
cathedral_1 = cv2.imread('cathedral1.jpg')
cathedral_1 = cv2.cvtColor(cathedral_1, cv2.COLOR_BGR2RGB)
semper0 = plt.imread("semper0.jpg")
semper1 = plt.imread("semper1.jpg")

# Question 1
# Get mask where green is in the threshold and also
# where greens are smaller than either of the other colours
mask = (image[:, :, 1] < 175) | (image[:, :, 2] > image[:, :, 1]) | (image[:, :, 0] > image[:, :, 1])
# Get Inverse mask
un_mask = np.invert(mask)
green = np.zeros_like(image, np.uint8)
green[mask] = image[mask]
def display_1():
    green[un_mask] = cat_background[un_mask]
    plt.axis('off')
    plt.imshow(green)
    
# Question2
grey= Image.open('lena_gray.gif').convert("L")
grey = np.asarray(grey).astype(float)
# Blur image
smoothed_grey = ndimage.gaussian_filter(grey, sigma=(2))
mask = grey - smoothed_grey
masked = grey + mask
smoothed_grey = smoothed_grey.astype(np.uint8)
grey = grey.astype(np.uint8)
mask = mask.astype(np.uint8)
masked_clipped = np.clip(masked, 0, 255)
masked_scaled = ((masked*1.0)/(np.max(masked)) * 255).astype(np.uint8)
masked = masked.astype(np.uint8)

# display question 2
def display_2():
    plt.axis('off')
    print("Original picture:")
    plt.imshow(grey, cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.axis('off')
    print("Smoothed with Gaussian filter:")
    plt.imshow(smoothed_grey,cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.axis('off')
    print("Mask:")
    plt.imshow(mask,cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.axis('off')
    print("Final Result Truncated:")
    plt.imshow(masked,cmap='gray')
    plt.show()
    plt.axis('off')
    print("Final Result Scaled:")
    plt.imshow(masked_scaled,cmap='gray')
    plt.show()
    plt.axis('off')
    print("Final Result Clipped:")
    plt.imshow(masked_clipped,cmap='gray')
    plt.show()


# Question 3
def nn_resize(base_image, scale):
    print("Starting Nearest Neighbour")
    x_scale = int(scale*base_image.shape[0])
    y_scale = int(scale*base_image.shape[1])
    resized = np.zeros((x_scale,y_scale,3),  dtype=np.uint8)
    for x in range(0, x_scale):
        for y in range(0, y_scale):
            x_old = int(round(x/scale))
            if x_old >= base_image.shape[0] - 1:
                x_old = base_image.shape[0] - 1
            y_old = int(round(y/scale))
            if y_old >= base_image.shape[1] - 1:
                y_old = base_image.shape[1] - 1
                
            resized[x, y, :] =  base_image[x_old,y_old,:]
            
    print("Done resizing")
    return resized

def display_3_nn():
    # Enlarge
    print("Scale = 3.1")
    large_nn = nn_resize(cat, 3.1)
    cat_nn = large_nn
    plt.imshow(large_nn)
    plt.show()
    plt.imshow(cat)
    plt.show()
    # Enlarge more
    print("Scale = 5")
    large_nn = nn_resize(cat, 5)
    plt.imshow(large_nn)
    plt.show()
    plt.imshow(cat)
    plt.show()
    # Shrink
    print("Scale = 0.9")
    small_nn = nn_resize(cat, 0.9)
    plt.imshow(small_nn)
    plt.show()
    plt.imshow(cat)
    plt.show()
    # Shrink more
    print("Scale = 0.1")
    small_nn = nn_resize(cat, 0.1)
    plt.imshow(small_nn)
    plt.show()
    plt.imshow(cat)
    plt.show()
    return cat_nn

def bl_resize(base_image, scale):
    print("Starting Bilinear Interpolation")
    x_scale = int(scale * base_image.shape[0])
    y_scale = int(scale * base_image.shape[1])
    resized = np.zeros((x_scale, y_scale, 3), dtype=np.uint8)
    for x in range(0, x_scale):
        for y in range(0, y_scale):
            x_old = x / scale
            if x_old >= base_image.shape[0] - 1:
                x_old = base_image.shape[0] - 1
            y_old = y / scale
            if y_old >= base_image.shape[1] - 1:
                y_old = base_image.shape[1] - 1
            xf = int(floor(x_old))
            xc = int(ceil(x_old))
            yf = int(floor(y_old))
            yc = int(ceil(y_old))
            c_ff = base_image[xf, yf, :].astype(dtype=int)
            c_cf = base_image[xc, yf, :].astype(dtype=int)
            c_fc = base_image[xf, yc, :].astype(dtype=int)
            c_cc = base_image[xc, yc, :].astype(dtype=int)
            if y_old == yc and x_old == xc:
                # color is the same        
                resized[x, y, :] = base_image[xc,yc,:]
            elif y_old == yc:
                resized[x, y, :] = (
                    0.5*
                    ((xc-x_old)*c_ff + (x_old - xf)*c_cf)
                    +
                    0.5*
                    ((xc-x_old)*c_fc + (x_old - xf)*c_cc)
                ).astype(np.uint8)
            elif x_old == xc:
                # Same x
                resized[x, y, :] = (
                    (yc - y_old)*
                    (0.5*c_ff + 0.5*c_cf)
                    +
                    (y_old -yf)*
                    (0.5*c_fc + 0.5*c_cc)
                ).astype(np.uint8)
            else:
                # Same y
                resized[x, y, :] = (
                    (yc - y_old)*
                    ((xc-x_old)*c_ff + (x_old - xf)*c_cf)
                    +
                    (y_old -yf)*
                    ((xc-x_old)*c_fc + (x_old - xf)*c_cc)
                ).astype(np.uint8)
    print("Done resizing")
    return resized

def display_3_bl():
    # Enlarge
    print("Scale = 3.1")
    large_bl = bl_resize(cat, 3.1)
    cat_bl = large_bl
    plt.imshow(large_bl)
    plt.show()
    plt.imshow(cat)
    plt.show()
    # Enlarge more
    print("Scale = 5")
    large_bl = bl_resize(cat, 5)
    plt.imshow(large_bl)
    plt.show()
    plt.imshow(cat)
    plt.show()
    # Shrink
    print("Scale = 0.9")
    small_bl = bl_resize(cat, 0.9)
    plt.imshow(small_bl)
    plt.show()
    plt.imshow(cat)
    plt.show()
    # Shrink more
    print("Scale = 0.1")
    small_bl = bl_resize(cat, 0.1)
    plt.imshow(small_bl)
    plt.show()
    plt.imshow(cat)
    plt.show()
    return cat_bl

def compare_nn_bl():
    print("Nearest Neighbour Interpolation:")
    plt.imshow(cat_nn[600:610, 2100:2110, :])
    plt.show()
    print("Bilinear Interpolation:")
    plt.imshow(cat_bl[600:610, 2100:2110, :])
    plt.show()
    print("Open CV resize:")
    plt.imshow(cv2.resize(cat,None,fx=3.1,fy=3.1)[600:610, 2100:2110, :])
    plt.show()

# Question 4

# Sources
# https://www.kaggle.com/wesamelshamy/tutorial-image-feature-extraction-and-matching
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
# https://docs.opencv.org/3.4.2/d5/dde/tutorial_feature_description.html
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html

def detect_colour(detector, image):
    key_points, descriptors = detector.detectAndCompute(image, None)
    return image, key_points, descriptors
    
def get_matches_orb(detector, image1, image2, nmatches=2000):
    image1, key_points_1, descriptors_1 = detect_colour(detector, image1)
    image2, key_points_2, descriptors_2 = detect_colour(detector, image2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)
    print("Matching features: {}".format(len(matches)))
    matches = sorted(matches, key = lambda x: x.distance)
    image_matches = image1
    for i in range(0, len(matches[:nmatches])):
        start = key_points_1[matches[i].queryIdx].pt;
        start = (int(round(start[0])), int(round(start[1])))
        end = key_points_2[matches[i].trainIdx].pt;
        end = (int(round(end[0])), int(round(end[1])))
        cv2.circle(image_matches, start, 2, (255, 0, 0), thickness=1, lineType=8, shift=0)
        cv2.line(image_matches, start, end, (0, 255, 0), thickness=2, lineType=8)
    plt.axis("off")
    plt.imshow(image_matches)  
    plt.show()

def get_matches_surf(detector, image1, image2, nmatches=5000):
    image1, key_points_1, descriptors_1 = detect_colour(detector, image1)
    image2, key_points_2, descriptors_2 = detect_colour(detector, image2)
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
    matches = matcher.match(descriptors_1, descriptors_2)
    print("Matching features: {}".format(len(matches)))
    matches = sorted(matches, key = lambda x: x.distance)
    image_matches = image1
    for i in range(0, len(matches[:nmatches])):
        start = key_points_1[matches[i].queryIdx].pt;
        start = (int(round(start[0])), int(round(start[1])))
        end = key_points_2[matches[i].trainIdx].pt;
        end = (int(round(end[0])), int(round(end[1])))
        cv2.circle(image_matches, start, 2, (255, 0, 0), thickness=1, lineType=8, shift=0)
        cv2.line(image_matches, start, end, (0, 255, 0), thickness=2, lineType=8)
    plt.axis("off")
    plt.imshow(image_matches)    
    plt.show()
    
def diff_algs():
    semper0 = plt.imread("semper0.jpg")
    semper1 = plt.imread("semper1.jpg")
    print("SURF, min hessian = {}".format(minHessian))
    get_matches_surf(surf, semper0, semper1, 300)
    print("ORB")
    semper0 = plt.imread("semper0.jpg")
    semper1 = plt.imread("semper1.jpg")
    get_matches_orb(orb, semper0, semper1, 300)

def change_hessian(show_vecs=100):
    minHessian = 400
    semper0 = plt.imread("semper0.jpg")
    semper1 = plt.imread("semper1.jpg")
    surf = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    print("SURF, min hessian = {}".format(minHessian))
    get_matches_surf(surf, semper0, semper1, show_vecs)

    minHessian = 300
    semper0 = plt.imread("semper0.jpg")
    semper1 = plt.imread("semper1.jpg")
    surf = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    print("SURF, min hessian = {}".format(minHessian))
    get_matches_surf(surf, semper0, semper1, show_vecs)

    minHessian = 200
    semper0 = plt.imread("semper0.jpg")
    semper1 = plt.imread("semper1.jpg")
    surf = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    print("SURF, min hessian = {}".format(minHessian))
    get_matches_surf(surf, semper0, semper1, show_vecs)

    minHessian = 100
    semper0 = plt.imread("semper0.jpg")
    semper1 = plt.imread("semper1.jpg")
    surf = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    print("SURF, min hessian = {}".format(minHessian))
    get_matches_surf(surf, semper0, semper1, show_vecs)

    minHessian = 50
    semper0 = plt.imread("semper0.jpg")
    semper1 = plt.imread("semper1.jpg")
    surf = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    print("SURF, min hessian = {}".format(minHessian))
    get_matches_surf(surf, semper0, semper1, show_vecs)

def display_4_1():
    semper0 = plt.imread("semper0.jpg")
    semper1 = plt.imread("semper1.jpg")
    get_matches_surf(surf, semper0, semper1, 500)

def display_4_2():
    semper0 = plt.imread("semper0.jpg")
    semper1 = plt.imread("semper1.jpg")
    diff_algs()
    
# Question 5

def get_matches_surf_scaled(detector, image1, image2, nmatches=5000, scale=1):
    threshold = max(image1.shape)/100.0
    correct_matches = 0
    image1, key_points_1, descriptors_1 = detect_colour(detector, image1)
    image2, key_points_2, descriptors_2 = detect_colour(detector, image2)
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
    
    # Match one-to-one
    if len(key_points_2) < len(key_points_1):
        matches = matcher.match(descriptors_2, descriptors_1)
        matches = sorted(matches, key = lambda x: x.distance)
        image_matches = image1
        
        for i in range(0, len(matches[:nmatches])):
            start = key_points_1[matches[i].trainIdx].pt;
            start = (int(round(start[0])), int(round(start[1])))
            end = key_points_2[matches[i].queryIdx].pt;
            org_x = end[0]/scale
            org_y = end[1]/scale
            end = (int(round(org_x)), int(round(org_y)))
            dist = math.sqrt((start[0] - end[0]*1.0)**2 + (start[1] - end[1]*1.0)**2)
            if dist <= threshold:
                correct_matches += 1
            cv2.circle(image_matches, start, 2, (255, 0, 0), thickness=1, lineType=8, shift=0)
            cv2.line(image_matches, start, end, (0, 255, 0), thickness=2, lineType=8)
    else:
        matches = matcher.match(descriptors_1, descriptors_2)
        matches = sorted(matches, key = lambda x: x.distance)
        image_matches = image1
        for i in range(0, len(matches[:nmatches])):
            start = key_points_1[matches[i].queryIdx].pt;
            start = (int(round(start[0])), int(round(start[1])))
            end = key_points_2[matches[i].trainIdx].pt;
            org_x = end[0]/scale
            org_y = end[1]/scale
            end = (int(round(org_x)), int(round(org_y)))
            dist = math.sqrt((start[0] - end[0]*1.0)**2 + (start[1] - end[1]*1.0)**2)
            if dist <= threshold:
                correct_matches += 1
            cv2.circle(image_matches, start, 2, (255, 0, 0), thickness=1, lineType=8, shift=0)
            cv2.line(image_matches, start, end, (0, 255, 0), thickness=2, lineType=8)
    repeatability = float(len(matches))/len(key_points_1)
    plt.axis("off")
    plt.imshow(image_matches)
    plt.show()
    accuracy = 0.0
    accuracy = (correct_matches*1.0)/len(matches)
    print("1% distance threshold: {}".format(threshold))
    print("Matching features: {}".format(len(matches)))
    print("Correct Matches: {}".format(correct_matches))
    return len(matches), repeatability, accuracy

def resize_5():
    minHessian = 100
    arc = plt.imread("arc.jpg")
    arc_scale = []
    scales = np.arange(0.1, 3.1, 0.1)
    print("Resizing:")
    for scale in scales:
        arc_scale.append(cv2.resize(arc,None,fx=scale,fy=scale))
    placeholder = []
    for scale in arc_scale:
        placeholder.append(scale)
    return placeholder, scales

def display_5():
    placeholder, scales = resize_5()
    arc_scale = []
    repeatability_list = []
    accuracy_list = []
    for scale in placeholder:
        arc_scale.append(scale)
    for i in range(0, len(scales)):
        base = plt.imread("arc.jpg")
        new = arc_scale[i]
        matches, repeatability, accuracy = get_matches_surf_scaled(surf, base, new, scale=scales[i])
        print("Scale: {}".format(scales[i]))
        print("Repeatability:{}%\nAccuracy: {}%\n**************".format(repeatability*100, accuracy*100))
        repeatability_list.append(repeatability)
        accuracy_list.append(accuracy)
        plt.show()
    return accuracy_list, repeatability_list, scales

def plot_5():
    plt.plot(scales, accuracy_list, label="Accuracy")
    plt.plot(scales, repeatability_list, label="Repeatability")
    plt.legend()
    plt.show()
    plt.plot(scales, accuracy_list, label="Accuracy")
    plt.legend()
    plt.show()
    plt.plot(scales, repeatability_list, label="Repeatability")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    display_1()
    display_2()
    cat_nn = display_3_nn()
    cat_bl = display_3_bl()
    compare_nn_bl()
    display_4_2()
    change_hessian(1000)
    accuracy_list, repeatability_list, scales = display_5()
    plot_5()