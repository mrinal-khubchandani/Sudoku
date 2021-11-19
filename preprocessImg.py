import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess(img):
    
    img1 = np.zeros([img.shape[0]+10,img.shape[1]+10])
    img1[5:img1.shape[0]-5,5:img1.shape[1]-5] = img
    ret, thresh = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY)
    
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = angle + 90

    M = cv2.getRotationMatrix2D((thresh.shape[0] / 2, thresh.shape[1] / 2), -angle, 1)
    img1 = cv2.warpAffine(thresh, M, (thresh.shape[0], thresh.shape[1])) 
    ret, img1 = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY)
    
    coor = np.where(img1 > 0)
    x_min = np.min(coor[0])
    y_min = np.min(coor[1])
    
    img1 = img1[x_min:,y_min:]
    img1 = img1[:img1.shape[0]-x_min,:img1.shape[1]-y_min]
    
    x_left = img1.shape[0]%9
    y_left = img1.shape[1]%9
    
    img1 = img1[x_left:,y_left:]
    
    img1 = cv2.resize(img1, (333, 333),interpolation = cv2.INTER_NEAREST)
    blurred = cv2.GaussianBlur(img1, (7, 7), 3)
    img1 = 255 - img1
    return img1