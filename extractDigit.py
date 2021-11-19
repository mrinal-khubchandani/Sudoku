import numpy as np
import cv2

def getBlocks(img):
    
    s = int(img.shape[0]/9)
    X = []
    Y = []
    for i in range(9):
        for j in range(9):
            img1 = img[s*i:s*(i+1),j*s:(j+1)*s]
            ret, img1 = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY)
            img1 = img1[4:,4:]
            img1 = img1[:img1.shape[0]-5,:img1.shape[1]-5]
            if np.count_nonzero(img1 == 255) / (img1.shape[0]*img1.shape[1]) < 0.04:
                Y.append(-1)
            else:
                Y.append(1)

            X.append(img1)
            
    
    return X,Y